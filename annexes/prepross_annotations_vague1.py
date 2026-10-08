# Empile les deux tours d'annotation de la vague 1 (CSV de la première annotation + parquet
# de la seconde) et exporte un fichier d'entrée pour le pipeline de codification.
# Repart de annexes/benchmarking/prepross_annotations.py.

#%%
import os
import re

import duckdb
from codif_common.s3 import connect_secret

S3_PREFIX = "s3://projet-budget-famille/data/annotations_vague1_2026"
SECOND_ROUND = (
    f"{S3_PREFIX}/seconde_annotations_vague1_2026/"
    "BDF_data_saisie_vague_1_a_codif_predictions_verif_henri.parquet"
)
S3_WORKFLOW_INPUTS = "s3://projet-budget-famille/data/workflow_inputs"
EXPORT_PATH = f"{S3_WORKFLOW_INPUTS}/bdf_vague1_annotation_2ndround.parquet"
# Dans un sous-dossier : le glob('*.csv') de S3_PREFIX ne descend pas dedans, le rapport
# de contradictions ne risque donc pas d'être empilé comme une annotation.
CONTRADICTIONS_PATH = f"{S3_PREFIX}/contradictions/contradictions_annotations_vague1.parquet"
# Table `code -> code_parent_equivalent` de l'étape prune-codes (pruning des hiérarchies
# linéaires). Elle ne dépend que de la nomenclature COICOP : n'importe quel run convient.
MAPPING_PATH = (
    "s3://projet-budget-famille/data/workflow_runs/2026-09-08/codif-c9vjm/"
    "prune-codes/mapping_lvl4.parquet"
)
PATTERNS = ("carnets_papier", "p1")

con = connect_secret()

#%%
# Première annotation : on garde les fichiers dont le nom contient l'un des PATTERNS
# (un fichier qui correspond à plusieurs motifs n'est listé qu'une fois).
all_files = [
    row[0]
    for row in con.execute(
        f"select file from glob('{S3_PREFIX}/*.csv') order by file"
    ).fetchall()
]
files = [f for f in all_files if any(p in os.path.basename(f) for p in PATTERNS)]

print(f"{len(files)}/{len(all_files)} fichiers sélectionnés :")
for file in files:
    print(f"  {os.path.basename(file)}")

#%%
# union_by_name : les carnets et les tickets n'ont pas exactement les mêmes colonnes
# (DATE_SAISIE / JOUR_COLL / NUM_CARN vs SAISIE / ID_TICK / code_mag), les colonnes
# absentes sont remplies par NULL. filename garde l'origine de chaque ligne.
# nullstr : MONT_DEP (110 lignes) et INTERNET (104) contiennent la chaîne "NA", qui
# typerait les deux colonnes en VARCHAR et casserait l'agrégation du budget. La chaîne
# vide doit être listée explicitement : nullstr *remplace* la liste par défaut au lieu
# de l'étendre, donc nullstr='NA' seul cesserait de lire les champs vides comme NULL.
first_round = con.execute(
    """
    select * from read_csv(
        $files, union_by_name = true, filename = true, nullstr = ['NA', '']
    )
    """,
    {"files": files},
).df()

# `source_saisie` n'existe que dans la seconde annotation : on la déduit du nom de
# fichier pour la première, avec le vocabulaire du fichier natif BDF_data_saisie_vague1
# (tick_appli / tick_pap / carn_pap).
assert "source_saisie" not in first_round.columns
first_round = con.execute(
    """
    select *, case
        when filename like '%tickets_appli%' then 'tick_appli'
        when filename like '%tickets_papier%' then 'tick_pap'
        when filename like '%carnets_papier%' then 'carn_pap'
    end as source_saisie
    from first_round
    """
).df()
assert first_round["source_saisie"].notna().all(), "fichier de source inconnue"

print(f"Première annotation : {first_round.shape[0]} lignes, {first_round.shape[1]} colonnes")
print(
    con.execute(
        "select source_saisie, count(*) as n from first_round group by all order by n desc"
    ).df().to_string(index=False)
)

#%%
# Seconde annotation. Un même `id` a pu être annoté plusieurs fois à des dates
# différentes (parfois avec des codes différents) : on garde l'annotation la plus
# récente, considérée comme la correction finale.
second_raw = con.execute(
    f"select * from read_parquet('{SECOND_ROUND}')"
).df()
second_round = con.execute(
    f"""
    select *
    from read_parquet('{SECOND_ROUND}')
    qualify row_number() over (partition by id order by timestamp desc) = 1
    """
).df()
print(
    f"Seconde annotation : {len(second_raw)} lignes lues, "
    f"{len(second_raw) - len(second_round)} doublons d'id écartés, "
    f"{len(second_round)} lignes gardées"
)

#%%
# Harmonisation des types sur ceux de la première annotation : DuckDB type en BOOLEAN
# les colonnes entièrement vides du parquet (NUM_CARN, JOUR_COLL, annee, comment), et
# ORD_DEP / SAISIE n'ont pas le même type d'un jeu à l'autre. Seules les colonnes
# présentes dans les deux jeux sont converties.
first_types = dict(
    con.execute(
        "select column_name, column_type from (describe select * from first_round)"
    ).fetchall()
)
casts = ", ".join(
    f'"{col}"::{first_types[col]} as "{col}"'
    for col in second_round.columns
    if col in first_types
)
second_round = con.execute(
    f"select * replace ({casts}), '{SECOND_ROUND}' as filename from second_round"
).df()

#%%
# Empilement. Les deux jeux sont disjoints sur `id` (vérifié) : le contrôle ci-dessous
# le garantit au cas où les fichiers sources changeraient.
annotations = con.execute(
    """
    select * from first_round
    union all by name
    select * from second_round
    """
).df()

duplicated_ids = con.execute(
    "select count(*) from (select id from annotations group by id having count(*) > 1)"
).fetchone()[0]
assert duplicated_ids == 0, f"{duplicated_ids} id présents plusieurs fois après empilement"

print(f"{annotations.shape[0]} lignes, {annotations.shape[1]} colonnes")
print(
    con.execute(
        """
        select
            coalesce(annotation, 'TOTAL') as annotation,
            case when grouping(source_saisie) = 1 then 'tous' else source_saisie end
                as source_saisie,
            count(*) as n
        from (
            select
                case when filename like '%seconde_annotations%' then 'seconde' else 'première' end
                    as annotation,
                source_saisie
            from annotations
        )
        group by rollup (annotation, source_saisie)
        order by grouping(annotation), annotation, grouping(source_saisie), source_saisie
        """
    ).df().to_string(index=False)
)

#%%
# Valeurs manquantes dans les deux colonnes de code. Au-delà des vrais NULL, les étapes
# LLM/regex écrivent des chaînes sentinelles ("N/A", "Reprise manuelle") que pandas et
# DuckDB traitent comme des valeurs ordinaires : tout ce qui ne respecte pas la forme
# COICOP 99[.9]* compte aussi comme manquant.
COICOP_RE = "^[0-9]{2}([.][0-9])*$"


def missing_counts(column: str) -> str:
    """Comptes de NULL / sentinelles pour une colonne de code, sous forme de select SQL."""
    return f"""
        select
            '{column}' as column_name,
            count(*) as n,
            count(*) filter ({column} is null) as nulls,
            count(*) filter (
                {column} is not null
                and not regexp_matches({column}, '{COICOP_RE}')
            ) as sentinels,
            round(100.0 * count(*) filter (
                {column} is null
                or not regexp_matches({column}, '{COICOP_RE}')
            ) / count(*), 3) as pct_missing
        from annotations
    """


missing = con.execute(
    f"{missing_counts('code')} union all {missing_counts('predicted_code')}"
).df()
print(missing.to_string(index=False))

#%%
# Doublons et contradictions, détectés sur la forme canonique des codes : troncature à 4
# positions puis pruning des hiérarchies linéaires, comme `trunc_and_prune_lvl4`
# (prune-codes/src/prune_codes/pruning.py). Beaucoup d'écarts entre deux annotations d'un
# même produit ne tiennent qu'au niveau 5 (01.1.1.3.1 contre 01.1.1.3) : ce ne sont ni
# des doublons à garder ni des contradictions.
con.execute(f"create or replace table mapping_lvl4 as select * from read_parquet('{MAPPING_PATH}')")
# Même règle que codif_common.codes.truncate_code(code, 4).
con.execute("create or replace macro trunc4(c) as array_to_string(list_slice(string_split(c, '.'), 1, 4), '.')")
annotations_canon = con.execute(
    """
    select a.*, coalesce(m.code_parent_equivalent, trunc4(a.code)) as canon
    from annotations a
    left join mapping_lvl4 m on m.code = trunc4(a.code)
    """
).df()
assert annotations_canon["canon"].notna().all(), "code sans forme canonique"

# La comparaison porte sur le libellé et le magasin en minuscules, sans espaces en
# bordure ; un magasin absent vaut la chaîne vide.

# Contradictions : même libellé × magasin annoté avec plusieurs codes canoniques.
# Calculées AVANT le dédoublonnage pour garder les effectifs réels, et exportées à part
# pour relecture. `profondeur` : les codes s'emboîtent (06.1.1 et 06.1.1.1), seul le niveau
# de détail diffère. `desaccord` : les codes divergent vraiment. `codes_origine` donne les
# codes tels qu'ils ont été annotés.
contradictions = con.execute(
    """
    with k as (
        select
            lower(trim(product)) as libelle,
            lower(trim(coalesce(MAG_DEP, ''))) as magasin,
            canon as code,
            string_agg(distinct code, ' | ') as codes_origine,
            count(*) as n_lignes,
            round(sum(MONT_DEP), 2) as montant_total,
            string_agg(distinct source_saisie, '+') as sources
        from annotations_canon
        group by libelle, magasin, canon
    )
    select
        libelle, magasin, code, codes_origine, n_lignes, montant_total, sources,
        case
            when bool_and(starts_with(longest, code)) over (partition by libelle, magasin)
                then 'profondeur'
            else 'desaccord'
        end as type_contradiction
    from (
        select *, arg_max(code, length(code)) over (partition by libelle, magasin) as longest
        from k
        qualify count(*) over (partition by libelle, magasin) > 1
    )
    order by type_contradiction desc, libelle, magasin, n_lignes desc
    """
).df()
n_paires = contradictions.groupby(["libelle", "magasin"]).ngroups
print(f"{n_paires} libellé × magasin avec plusieurs codes canoniques ({len(contradictions)} lignes)")
print(
    contradictions.groupby("type_contradiction")[["libelle", "magasin"]]
    .apply(lambda d: len(d.drop_duplicates()))
    .to_string()
)

# Doublons : même libellé, même magasin, même code canonique. Un libellé répété ne porte
# aucune information de plus pour le pipeline, et le découpage train/test se fait par
# ligne : des lignes identiques tomberaient des deux côtés. On garde la ligne annotée le
# plus récemment. Quand les codes bruts du groupe diffèrent (le doublon n'apparaît que
# grâce à la canonisation), la ligne gardée reçoit le code canonique ; sinon, comme pour
# toute ligne sans doublon, le code d'origine est conservé.
#
# Seuls les doublons *concordants* sont retirés : un même libellé × magasin codé
# différemment au niveau canonique est conservé, ce n'est pas un doublon.
#
# Attention : la ligne retirée emporte son MONT_DEP. Deux articles identiques sur un
# même ticket comptent donc pour un seul dans toute somme de budget faite sur ce fichier.
n_avant = len(annotations)
annotations = con.execute(
    """
    select * exclude (canon, codes_bruts_differents)
        replace (case when codes_bruts_differents then canon else code end as code)
    from (
        select *,
            min(code) over w <> max(code) over w as codes_bruts_differents
        from annotations_canon
        window w as (
            partition by lower(trim(product)), lower(trim(coalesce(MAG_DEP, ''))), canon
        )
    )
    qualify row_number() over (
        partition by lower(trim(product)), lower(trim(coalesce(MAG_DEP, ''))), canon
        order by timestamp desc nulls last
    ) = 1
    """
).df()
print(f"{n_avant - len(annotations)} doublons retirés ({n_avant} -> {len(annotations)} lignes)")
print(
    con.execute(
        """
        select
            case when filename like '%seconde_annotations%' then 'seconde' else 'première' end
                as annotation,
            count(*) as n
        from annotations group by all order by all
        """
    ).df().to_string(index=False)
)

#%%
# Renommage des colonnes pour le pipeline.
#
# « Ajoutée par la codification » est déduit, pas codé en dur : l'union des noms de
# colonnes des fichiers sources de data/workflow_inputs/ est le schéma natif de la BDF,
# donc tout le reste a été produit par la codification ou par la vérification manuelle
# qui a suivi, et est supprimé (sauf `code`, l'étiquette).
#
# `product` est l'exception : il contient le libellé NAT_DEP d'origine, il redevient
# donc NAT_DEP, la colonne texte que le pipeline lit par défaut.

# Noms canoniques que build-datasets/main.py crée ou écrase en mode prédiction. Un de
# ces noms resté dans le fichier d'entrée serait lu comme vérité terrain (`code`, `coicop`
# ne sont posés par défaut que s'ils sont absents) ou écrasé en silence (`id`).
PIPELINE_RESERVED = {
    "raw_product", "l_pr_product", "s_pr_product", "source", "annee", "code",
    "coicop", "shop", "shop_type_name", "budget", "id", "n_obs",
    "_source_input_file",
}

native_columns = set(
    con.execute(
        f"""
        select column_name
        from (describe from read_csv_auto(
            '{S3_WORKFLOW_INPUTS}/*_a_codif.csv',
            delim = ';', nullstr = 'NA',
            types = {{'ID_SABIANE': 'VARCHAR'}}, union_by_name = true
        ))
        """
    ).df()["column_name"]
)

# Le fichier exporté est un simple input du pipeline : on retire tout ce qui vient de
# l'ancienne codification ou de son annotation. Seule `code` est gardée, sans suffixe :
# c'est l'étiquette (annotation manuelle) que lit l'étape `evaluate`.
COLUMNS_TO_DROP = [
    # prédiction du pipeline
    "predicted_code", "prediction_source", "llm_comment",
    # dérivées par le pipeline, recalculées depuis NAT_DEP et MAG_DEP
    "shop_type_name", "s_pr_product_orig",
    # métadonnées de l'annotation
    "comment", "timestamp", "annee", "coicop",
    # traçabilité
    "id", "filename",
]

# Garde-fou : toute colonne non native doit être traitée explicitement ci-dessus.
non_native = set(annotations.columns) - native_columns
unexpected = non_native - {"product", "code"} - set(COLUMNS_TO_DROP)
assert not unexpected, f"colonnes non natives non traitées : {sorted(unexpected)}"

export = annotations.drop(columns=COLUMNS_TO_DROP).rename(columns={"product": "NAT_DEP"})
print(f"{len(COLUMNS_TO_DROP)} colonnes supprimées, {export.shape[1]} restantes")

#%%
# Compatibilité des noms de colonnes avec le contrat d'entrée du pipeline
# (build-datasets/main.py : build_observations / load_input_file).
invalid_names = [c for c in export.columns if not re.fullmatch(r"\w+", c)]
# `code` est réservé, mais c'est voulu : build-datasets le lit comme vérité terrain
# (l'étiquette manuelle) et ne le pose par défaut que s'il est absent.
reserved_left = sorted((set(export.columns) & PIPELINE_RESERVED) - {"code"})

checks = {
    "colonne texte NAT_DEP présente (-p text_column=NAT_DEP)": "NAT_DEP" in export.columns,
    "colonne magasin MAG_DEP présente (-p shop_column=MAG_DEP)": "MAG_DEP" in export.columns,
    "colonne budget MONT_DEP présente (-p budget_column=MONT_DEP)": "MONT_DEP"
    in export.columns,
    "étiquette code présente (-p label-column=code)": "code" in export.columns,
    "aucun nom en collision avec les noms canoniques du pipeline": not reserved_left,
    "noms = identifiants simples (ni espace, ni accent, ni point)": not invalid_names,
    "aucun nom en double": not export.columns.duplicated().any(),
}
for label, ok in checks.items():
    print(f"  [{'OK' if ok else 'KO'}] {label}")
if reserved_left:
    print(f"  -> en collision : {reserved_left}")
if invalid_names:
    print(f"  -> invalides : {invalid_names}")

print(
    con.execute("select column_name, column_type from (describe select * from export)")
    .df()
    .to_string(index=False)
)

#%%
# Export vers S3, à lancer après validation des contrôles ci-dessus.
# ID_SABIANE est forcé en VARCHAR quand le pipeline lit un CSV ; le parquet porte ses
# propres types, le cast doit donc être explicite ici.
EXPORTER = False

assert all(checks.values()), "noms de colonnes incompatibles avec l'entrée du pipeline"

if EXPORTER:
    con.execute(
        f"""
        copy (select * replace (ID_SABIANE::varchar as ID_SABIANE) from export)
        to '{EXPORT_PATH}' (format parquet)
        """
    )
    print(f"{len(export)} lignes exportées vers {EXPORT_PATH}")

    # Rapport de contradictions, en parquet.
    con.execute(
        f"""
        copy (select * from contradictions)
        to '{CONTRADICTIONS_PATH}' (format parquet)
        """
    )
    print(f"{len(contradictions)} lignes de contradictions exportées vers {CONTRADICTIONS_PATH}")
else:
    print(f"Export désactivé (EXPORTER = False) : {len(export)} lignes -> {EXPORT_PATH}")
    print(f"  et {len(contradictions)} lignes de contradictions -> {CONTRADICTIONS_PATH}")
