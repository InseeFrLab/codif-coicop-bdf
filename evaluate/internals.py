"""Décomposition interne des classifieurs.

L'accuracy globale dit *combien* un classifieur se trompe. Elle ne dit pas
*où* : un RAG à 70 % peut être un retriever qui ne ramène jamais la bonne
réponse, ou un générateur qui la voit et choisit autre chose. Les deux
appellent des corrections opposées — réindexer d'un côté, retoucher le prompt
de l'autre — et rien dans le chiffre agrégé ne les distingue.

Ces indicateurs vivaient dans les étapes de classification, chacune loguant les
siens dans sa propre expérience MLflow, sur son propre périmètre et avec sa
propre convention. Ils sont rapatriés ici, calculés sur le **même** jeu de
lignes et la **même** vérité canonique que le reste du rapport : c'est ce qui
les rend enfin comparables entre classifieurs.

Ce module est importé par `evaluation_report.qmd` (qui en fait des tableaux) et
par `main.py` (qui en loggue les scalaires). Un seul calcul, deux sorties : le
rapport et MLflow ne peuvent pas diverger.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import pandas as pd

from rag_annotations.eval import (
    codable_reliability,
    confidence_reliability,
    distribution_distortion,
)
from rag_notices.eval.metrics import compute_hierarchical_metrics

# Les codes canoniques n'excèdent jamais 4 segments : au-delà, les tableaux
# seraient structurellement vides.
LEVELS: Sequence[int] = (1, 2, 3, 4)
TARGET_LEVEL = 4

# Seuil de confiance du régime `threshold`, repris de `eval.threshold_confidence`
# des deux configs RAG, qui valent 0.7 toutes les deux.
CONFIDENCE_THRESHOLD = 0.7

# Grille du balayage de seuils. Pas de 0,05 et non 0,1 : c'est la table qui
# instruit une décision de relecture, et un pas de 10 points laisse choisir
# entre deux compromis très éloignés.
CONFIDENCE_SWEEP = tuple(round(0.5 + 0.05 * i, 2) for i in range(9))

# Les cinq régimes de réponse, du plus permissif au plus strict. Lire la colonne
# `n` autant que l'accuracy : une accuracy qui monte de régime en régime sur une
# population qui fond n'est pas une amélioration, c'est une sélection.
REGIMES = [
    ("all_raw", "Toutes les lignes"),
    ("all_parsed", "Réponse JSON exploitable"),
    ("codable_only", "`codable` = vrai"),
    ("parsed_and_codable", "Exploitable **et** codable"),
    ("threshold", f"… et confiance ≥ {CONFIDENCE_THRESHOLD}"),
]

# Confiances de l'échelle [0, 1], donc directement comparables entre elles.
# `llm_confiance` (entier 1-5) et `sirus_proba` en sont exclus : ils ont chacun
# leur propre section de calibration, et les mélanger ici produirait des seuils
# qui ne veulent rien dire.
UPSTREAM_CONFIDENCES = [
    ("RAG", "rag_code", "rag_confidence"),
    ("RAG-annot", "ragann_code", "ragann_confidence"),
    ("TTC", "ttc_code_1", "ttc_conf_1"),
]


# ---------------------------------------------------------------------------
# Mise en forme des entrées
# ---------------------------------------------------------------------------

def widen_retrieved(retrieved: pd.DataFrame) -> pd.DataFrame:
    """`retrieved_codes.parquet` est en format large : une colonne par rang
    récupéré, nommée "0", "1", … Le rassembler en une liste par ligne.

    Le nombre de colonnes n'est pas supposé connu : il vaut `retrieval.size` de
    la config au moment de l'indexation, qui peut avoir changé depuis.
    """
    rank_cols = [c for c in retrieved.columns if str(c).isdigit()]
    if not rank_cols or "id" not in retrieved.columns:
        return pd.DataFrame(columns=["id", "list_retrieved_codes"])
    rank_cols.sort(key=int)
    out = retrieved[["id"]].copy()
    out["list_retrieved_codes"] = [
        [c for c in row if isinstance(c, str) and c]
        for row in retrieved[rank_cols].to_numpy()
    ]
    return out


def build_records(
    scorable: pd.DataFrame,
    truth_col: str,
    predicted_col: str,
    confidence_col: Optional[str] = None,
    flags: Optional[pd.DataFrame] = None,
    retrieved: Optional[pd.DataFrame] = None,
) -> List[Dict]:
    """Assemble les enregistrements qu'attendent les bibliothèques de mesure.

    La prédiction et la vérité viennent du parquet de conciliation — les mêmes
    colonnes que le reste du rapport score, donc les chiffres se raccordent.
    `parsed` / `codable` et les codes récupérés, eux, ne survivent pas à la
    fusion : ils viennent des parquets de l'étape elle-même, joints sur `id`.

    `parsed` vaut vrai par défaut : un classifieur sans notion de parsing (TTC,
    LCS) a toujours « répondu », et `confidence_reliability` écarte les lignes
    non parsées.
    """
    cols = ["id", truth_col, predicted_col]
    if confidence_col and confidence_col in scorable.columns:
        cols.append(confidence_col)
    base = scorable[[c for c in cols if c in scorable.columns]].copy()

    base = base.rename(columns={truth_col: "code", predicted_col: "code_predict"})
    if confidence_col and confidence_col in base.columns:
        base = base.rename(columns={confidence_col: "confidence"})
    else:
        base["confidence"] = None

    for frame, keep in ((flags, ["id", "parsed", "codable"]),
                        (retrieved, ["id", "list_retrieved_codes"])):
        if frame is None or "id" not in frame.columns:
            continue
        present = [c for c in keep if c in frame.columns]
        if len(present) > 1:
            base = base.merge(frame[present].drop_duplicates("id"), how="left", on="id")

    if "parsed" not in base.columns:
        base["parsed"] = True
    else:
        base["parsed"] = base["parsed"].fillna(True).astype(bool)
    if "codable" not in base.columns:
        base["codable"] = None
    if "list_retrieved_codes" not in base.columns:
        base["list_retrieved_codes"] = None

    records = base.to_dict("records")
    for r in records:
        # DuckDB rend une colonne de listes en `numpy.ndarray`, pas en `list` :
        # un test `isinstance(..., list)` la rejetterait en silence, et le
        # classifieur perdrait tout son retrieval sans qu'aucune erreur ne sorte.
        raw = r.get("list_retrieved_codes")
        if raw is None or isinstance(raw, float):
            r["list_retrieved_codes"] = []
            continue
        try:
            r["list_retrieved_codes"] = [
                str(c) for c in raw if isinstance(c, str) and c
            ]
        except TypeError:
            r["list_retrieved_codes"] = []
    return records


def _read(con, path: Optional[str]) -> Optional[pd.DataFrame]:
    """Lecture tolérante : un artefact absent rend `None`, pas une exception.

    Ces parquets sont des compléments. Un run partiel, ou relancé étape par
    étape, peut n'en avoir aucun — le rapport doit rester lisible et l'étape ne
    doit pas échouer pour un tableau manquant. Ce qui, lui, ne se rattrape pas
    (la vérité canonique absente), échoue en tête de `main.py`.
    """
    if not path:
        return None
    try:
        return con.sql(f"SELECT * FROM read_parquet('{path}')").df()
    except Exception as exc:  # noqa: BLE001 — cf. docstring
        print(f"[evaluate] artefact illisible, ignoré : {path} ({exc})", flush=True)
        return None


def _count(con, path: Optional[str], where: str = "") -> Optional[int]:
    """Nombre de lignes d'un parquet, sans le charger. None si illisible.

    `SELECT count(*)` et non `_read` : ces artefacts font la taille du jeu à
    coder, et on n'en veut qu'un entier. Même tolérance que `_read`.
    """
    if not path:
        return None
    try:
        clause = f" WHERE {where}" if where else ""
        return int(con.sql(f"SELECT count(*) FROM read_parquet('{path}'){clause}").fetchone()[0])
    except Exception as exc:  # noqa: BLE001 — cf. `_read`
        print(f"[evaluate] artefact illisible, ignoré : {path} ({exc})", flush=True)
        return None


def input_counts(con, path: Optional[str]) -> Optional[dict]:
    """Décompte du fichier d'entrée, écrit par `build-datasets`. None si absent.

    Absent veut dire : run antérieur à l'introduction de cet artefact, ou run
    lancé sans `--input-file`. Même contrat tolérant que `_read` — le rapport se
    dégrade, il n'échoue pas.
    """
    frame = _read(con, path)
    if frame is None or not len(frame):
        return None
    # pandas remonte en NaN ce que l'écrivain avait mis à None (`n_labelled` est
    # nullable) : on le ramène à None pour que l'appelant teste une seule chose.
    return {
        k: (None if pd.isna(v) else v)
        for k, v in frame.iloc[0].to_dict().items()
    }


def run_metadata_rows(counts: Optional[dict]) -> List[tuple]:
    """Lignes « identité » du tableau de métadonnées.

    Les VOLUMES n'y sont plus : ils vivent dans `funnel_rows`, qui les présente
    en entonnoir. Deux tableaux plutôt qu'un parce qu'ils répondent à deux
    questions — « quel run, sur quel fichier ? » et « qu'est devenu ce fichier ? ».
    """
    if counts is None:
        return [(
            "Fichier d'entrée",
            "_(décompte absent : run antérieur à `build-datasets/input_counts.parquet`, "
            "ou lancé sans `--input-file`)_",
        )]
    return [("Fichier d'entrée", f"`{counts['input_file']}`")]


def funnel_rows(
    counts: Optional[dict],
    *,
    n_regex: Optional[int] = None,
    n_chain: Optional[int] = None,
    n_conciliation: Optional[int] = None,
    n_scorable: Optional[int] = None,
    n_deliverable: Optional[int] = None,
) -> Optional[List[tuple]]:
    """L'entonnoir : ce que devient le fichier d'entrée, étape par étape.

    Répond à « combien de lignes sont écartées d'emblée, combien sont tranchées
    par la regex, combien entrent vraiment dans la chaîne de codification ». Ces
    trois volumes vivaient dans trois endroits différents du pipeline et nulle
    part ensemble.

    Renvoie des triplets `(étape, lignes, part)`. La part se rapporte au fichier
    d'entrée ; sur un run antérieur à `input_counts`, elle se rapporte aux
    observations retenues, et l'entonnoir démarre simplement plus bas.

    Chaque ligne est omise si son compte manque : un artefact absent réduit
    l'entonnoir, il ne le fait pas échouer.
    """
    def fmt(value: int) -> str:
        return f"{int(value):,}".replace(",", " ")

    base = counts["n_input_rows"] if counts else n_deliverable
    if not base:
        return None

    def part(value: Optional[int]) -> str:
        return "—" if value is None else f"{value / base:.1%}"

    rows: List[tuple] = []
    n_retenues = counts["n_observations"] if counts else n_deliverable

    if counts:
        rows.append(("**Fichier d'entrée**", f"**{fmt(counts['n_input_rows'])}**",
                     part(counts["n_input_rows"])))
        rows.append(("— écartées : libellé vide après nettoyage",
                     fmt(counts["n_dropped_empty_label"]), part(counts["n_dropped_empty_label"])))
        rows.append(("— écartées : **produit non codable**",
                     fmt(counts["n_dropped_uncodable"]), part(counts["n_dropped_uncodable"])))
        inexplique = (
            counts["n_input_rows"] - counts["n_dropped_empty_label"]
            - counts["n_dropped_uncodable"] - counts["n_observations"]
        )
        if inexplique:
            rows.append(("— écartées : **cause non identifiée**",
                         f"**{fmt(inexplique)}**", part(inexplique)))

    rows.append(("**Observations retenues** = lignes du fichier livré",
                 f"**{fmt(n_retenues)}**", part(n_retenues)))

    # `export-results` part de TOUTES les observations et fusionne en `left` :
    # les deux nombres sont égaux par construction, et la ligne ci-dessus
    # l'affirme. Le jour où une jointure duplique, il faut le dire plutôt que de
    # laisser l'entonnoir mentir avec assurance.
    if counts and n_deliverable is not None and n_deliverable != counts["n_observations"]:
        rows.append(("⚠️ lignes réellement dans le livrable",
                     f"**{fmt(n_deliverable)}**", part(n_deliverable)))

    if n_regex is not None:
        rows.append(("— **codées par la regex**, hors chaîne", fmt(n_regex), part(n_regex)))
    if n_chain is not None:
        rows.append(("— **entrées dans la chaîne de codification**", fmt(n_chain), part(n_chain)))

    # Ce qui n'est ni capté par la regex ni entré dans la chaîne a été retiré par
    # `--sample-observations` (échantillonnage centralisé à classify-regex). Ne
    # s'affiche que sur un run échantillonné, où c'est l'explication du reste.
    if n_regex is not None and n_chain is not None and n_retenues is not None:
        reste = n_retenues - n_regex - n_chain
        if reste:
            rows.append(("— retirées par l'échantillonnage", fmt(reste), part(reste)))

    if n_conciliation is not None:
        rows.append(("parvenues à la conciliation", fmt(n_conciliation), part(n_conciliation)))
    if n_scorable is not None:
        rows.append(("dont portant un code de référence (mesurables)",
                     fmt(n_scorable), part(n_scorable)))
    return rows


def group_summary(
    data: pd.DataFrame,
    group_col: str,
    *,
    truth_col: str,
    final_col: str,
    levels: Sequence[int] = (1, 2, 4),
    min_n: int = 1,
) -> Optional[pd.DataFrame]:
    """Accuracy de la conciliation ventilée par modalité de ``group_col``.

    Conçue pour la provenance du produit (ticket de caisse, carnet papier,
    saisie assistée), où la question n'est pas « quelle brique se trompe » mais
    « quelle source de saisie coûte de la qualité ». D'où les colonnes :

    - le **volume** et sa part, sans quoi un écart d'accuracy sur 12 lignes se
      lit comme un écart sur 12 000 ;
    - la **couverture** : une source peut faire chuter l'accuracy en produisant
      des libellés que la chaîne refuse de coder, ce qui n'est pas le même
      défaut que de les coder faux ;
    - l'accuracy à plusieurs niveaux : une saisie pauvre dégrade souvent le
      niveau 4 en laissant le niveau 1 intact (on sait que c'est de
      l'alimentaire, pas quel poste) ;
    - l'**écart au niveau 4 par rapport à l'ensemble**, en points : c'est lui
      qui désigne où porter l'effort.

    Renvoie None si la colonne est absente ou entièrement vide.
    """
    from codif_common.metrics import accuracy, answer_mask

    if group_col not in data.columns or not data[group_col].notna().any():
        return None

    ref_level = levels[-1]
    _, _, acc_ensemble = accuracy(data[truth_col], data[final_col], ref_level)

    rows = []
    for value, sub in data.groupby(group_col, dropna=False):
        row = {
            group_col: value,
            "n": len(sub),
            "part": len(sub) / len(data),
            "couverture": float(answer_mask(sub[final_col]).mean()),
        }
        for k in levels:
            _, _, acc = accuracy(sub[truth_col], sub[final_col], k)
            row[f"accuracy niv{k}"] = acc
        row["écart / ensemble"] = (
            row[f"accuracy niv{ref_level}"] - acc_ensemble
            if pd.notna(row[f"accuracy niv{ref_level}"]) and pd.notna(acc_ensemble)
            else None
        )
        rows.append(row)

    out = pd.DataFrame(rows)
    out = out[out["n"] >= min_n].sort_values("n", ascending=False)
    return out if len(out) else None


def division_labels(con, path: Optional[str]) -> Dict[str, str]:
    """Libellés des divisions COICOP, lus dans la nomenclature du run.

    ``{"01": "Produits alimentaires et boissons non alcoolisées", …}``. Un code
    de division est un code à **un seul segment** : on filtre là-dessus plutôt
    que sur la colonne ``type``, dont les libellés dépendent de la version du
    fichier source.

    Renvoie un dictionnaire vide si la nomenclature est absente ou n'a pas les
    colonnes attendues — les tableaux affichent alors le code seul. Un libellé
    manquant ne doit jamais faire échouer un rapport de mesure.
    """
    frame = _read(con, path)
    if frame is None or not {"code", "label_fr"} <= set(frame.columns):
        return {}
    codes = frame["code"].astype("string")
    divisions = frame[codes.notna() & ~codes.str.contains(".", regex=False)]
    return {
        str(r["code"]): str(r["label_fr"])
        for _, r in divisions.iterrows()
        if pd.notna(r["label_fr"])
    }


def label_division(code: str, labels: Dict[str, str], *, width: int = 48) -> str:
    """``"01"`` → ``"01 — Produits alimentaires et boissons non alcoolisées"``."""
    if code is None:
        return "— (aucun code émis)"
    libelle = labels.get(str(code))
    if not libelle:
        return str(code)
    if len(libelle) > width:
        libelle = libelle[: width - 1].rstrip() + "…"
    return f"{code} — {libelle}"


def accuracy_by_predicted_division(
    data: pd.DataFrame,
    *,
    truth_col: str,
    final_col: str,
    labels: Optional[Dict[str, str]] = None,
    target_level: int = 4,
) -> Optional[pd.DataFrame]:
    """« Quand le pipeline prédit de l'alimentaire, a-t-il le bon code ? »

    Regroupe par la division **prédite**, et non par la division vraie. Les deux
    tableaux se ressemblent et répondent à des questions opposées :

    - par division **vraie** : « parmi les vrais produits alimentaires, combien
      sont bien codés ? » — c'est ce qu'on veut savoir pour juger la couverture
      d'un domaine ;
    - par division **prédite** (ici) : « parmi les produits que la chaîne dit
      alimentaires, combien le sont vraiment, et combien ont le bon code
      complet ? » — c'est la question opérationnelle, celle qu'on se pose devant
      un fichier livré, quand la vérité n'est pas connue.

    Deux taux, dans cet ordre de lecture : la division prédite est-elle la bonne,
    puis le code complet l'est-il. Le second ne peut pas dépasser le premier —
    avoir le bon code de niveau 4 suppose la bonne division.

    Les lignes sans code émis forment leur propre groupe : elles ne sont pas une
    division, mais leur volume fait partie de la lecture.
    """
    from codif_common.metrics import accuracy, code_parts

    labels = labels or {}
    div = data[final_col].map(lambda c: code_parts(c)[0] if code_parts(c) else None)
    work = data.assign(_div=div)

    rows = []
    for value, sub in work.groupby("_div", dropna=False):
        value = None if pd.isna(value) else value
        n_ok1, _, acc1 = accuracy(sub[truth_col], sub[final_col], 1)
        n_ok4, _, acc4 = accuracy(sub[truth_col], sub[final_col], target_level)
        rows.append({
            "division prédite": label_division(value, labels),
            "n prédits": len(sub),
            "part des prédictions": len(sub) / len(work),
            "bonne division": acc1,
            f"bon code niv{target_level}": acc4,
        })
    if not rows:
        return None
    return pd.DataFrame(rows).sort_values("n prédits", ascending=False)


def distortion_level1(
    data: pd.DataFrame, *, truth_col: str, final_col: str
) -> Optional[dict]:
    """Distorsion entre la distribution VRAIE et la distribution PRÉDITE au niveau 1.

    Question différente de l'accuracy : celle-ci demande « chaque produit est-il
    bien codé ? », celle-là « la répartition par division ressemble-t-elle à la
    vraie ? ». Les deux se séparent dès que les erreurs se **compensent** — mille
    produits à tort en 01 et mille à tort hors de 01 laissent la distribution
    intacte et l'accuracy au sol. C'est la différence entre un agrégat
    exploitable et une codification individuelle juste.

    Réutilise ``rag_annotations.eval.distribution_distortion`` (le TV et la KL y
    sont déjà implémentés et testés) et lui ajoute ce qui rend le chiffre
    lisible : les erreurs brutes, et la part d'entre elles qui se compensent.

    Renvoie None si aucune ligne n'est mesurable.
    """
    from codif_common.metrics import accuracy
    from rag_annotations.eval import distribution_distortion

    records = [
        {"code": t, "code_predict": p}
        for t, p in zip(data[truth_col], data[final_col])
    ]
    dist = distribution_distortion(records, level=1)
    if not dist["n"]:
        return None

    n_ok, n, _ = accuracy(data[truth_col], data[final_col], 1)
    erreurs_brutes = n - n_ok
    # TV × n : le nombre de produits qu'il faudrait DÉPLACER d'une division à
    # l'autre pour que la distribution prédite coïncide avec la vraie. C'est la
    # lecture exacte de la distance en variation totale, pas une approximation.
    deplacements = dist["tv_distance"] * dist["n"]
    return {
        **dist,
        "n_scored": n,
        "erreurs_brutes": erreurs_brutes,
        "deplacements": deplacements,
        # Part des erreurs de division qui s'annulent entre elles. 0 % : toutes
        # les erreurs vont dans le même sens, l'agrégat est biaisé d'autant.
        # 90 % : la distribution est presque juste malgré des erreurs
        # individuelles nombreuses.
        "compensation": (
            1 - deplacements / erreurs_brutes if erreurs_brutes else None
        ),
    }


def load_classifier_records(
    con,
    scorable: pd.DataFrame,
    truth_col: str,
    ragnotices_path: Optional[str] = None,
    retrieved_path: Optional[str] = None,
    ragann_path: Optional[str] = None,
) -> Dict[str, List[Dict]]:
    """Les enregistrements par classifieur, prêts pour tous les indicateurs.

    Appelé par le rapport **et** par `main.py` : les tableaux rendus et les
    scalaires logués dans MLflow décrivent ainsi exactement les mêmes lignes.

    Un classifieur absent du parquet de conciliation est simplement omis, de
    même qu'un artefact complémentaire manquant — le classifieur reste alors
    présent, sans ses colonnes de retrieval ni ses drapeaux.
    """
    rag_flags = _read(con, ragnotices_path)
    retrieved_wide = _read(con, retrieved_path)
    ragann = _read(con, ragann_path)

    retrieved = widen_retrieved(retrieved_wide) if retrieved_wide is not None else None

    ragann_retrieved = None
    if ragann is not None and "list_retrieved_codes" in ragann.columns:
        ragann_retrieved = ragann[["id", "list_retrieved_codes"]]

    specs = [
        ("RAG", "rag_code", "rag_confidence", rag_flags, retrieved),
        ("RAG-annot", "ragann_code", "ragann_confidence", ragann, ragann_retrieved),
        ("TTC", "ttc_code_1", "ttc_conf_1", None, None),
        ("LCS", "lcs_code", None, None, None),
    ]

    out: Dict[str, List[Dict]] = {}
    for name, pred_col, conf_col, flags, retr in specs:
        if pred_col not in scorable.columns:
            continue
        out[name] = build_records(
            scorable, truth_col, pred_col, conf_col, flags=flags, retrieved=retr
        )
    return out


# ---------------------------------------------------------------------------
# Les indicateurs
# ---------------------------------------------------------------------------

def has_response_flags(records: List[Dict]) -> bool:
    """La brique porte-t-elle un drapeau de codabilité ?

    `build_records` pose `codable = None` pour TTC et LCS, qui n'ont pas cette
    notion. Le découpage par régime n'a alors aucun sens : les régimes
    `codable_only`, `parsed_and_codable` et `threshold` sont **vides**.

    Ce test remplace une heuristique fausse qui vivait dans le rapport
    (« tous les régimes ont le même effectif »). Ils n'ont pas le même effectif,
    ils sont vides : la colonne `n` vaut `[N, N, 0, 0, 0]`, donc deux valeurs
    distinctes, donc le garde ne se déclenchait jamais et TTC et LCS avaient
    droit à un sous-tableau de zéros et de tirets.
    """
    return any(r.get("codable") is not None for r in records)


def retrieval_size(records: List[Dict]) -> Optional[tuple[int, int]]:
    """``(min, max)`` du nombre de codes récupérés par produit, ou ``None``.

    Mesuré sur les données du run et non lu dans `retrieval.size` des configs
    RAG : c'est la valeur qui a servi à *ce* run qu'il faut afficher, pas celle
    du fichier de configuration d'aujourd'hui. Sans elle, le recall se lit sans
    échelle — « recall de 80 % » ne veut rien dire tant qu'on ignore si le
    retriever ramenait 5 candidats ou 50.

    Le minimum peut être inférieur au maximum : Qdrant renvoie *au plus*
    `limit` points, et une collection réduite en rend moins.
    """
    tailles = [len(r["list_retrieved_codes"]) for r in records if r["list_retrieved_codes"]]
    if not tailles:
        return None
    return min(tailles), max(tailles)


def hierarchical(records: List[Dict]) -> Dict:
    """Accuracy, recall de retrieval et accuracy conditionnelle, par régime et
    par niveau. Une seule passe sur les données, réutilisée par trois tableaux."""
    return compute_hierarchical_metrics(
        records,
        predicted_col="code_predict",
        label_col="code",
        confidence_col="confidence",
        codable_col="codable",
        parsed_col="parsed",
        retrieved_col="list_retrieved_codes",
        threshold=CONFIDENCE_THRESHOLD,
        by_product_type=False,
    )["overall"]


def retrieval_table(overall: Dict, regime: str = "all_raw") -> pd.DataFrame:
    """Ce que le retriever ramène, et ce que le générateur en fait.

    Trois colonnes par niveau, et c'est leur écart qui informe :

    - **recall** : part des lignes dont la bonne réponse figure dans la liste
      récupérée. C'est la matière dont dispose le générateur. Ce n'est **pas**
      un plafond strict : la consigne de reprendre un candidat à l'identique est
      incitative, pas structurelle, et un modèle peut sortir un code juste hors
      liste. Une accuracy au-dessus du recall mesure exactement cette part ;
    - **accuracy si récupéré** : sur ces lignes-là, à quelle fréquence le LLM
      choisit effectivement la bonne. C'est la qualité de la **génération**,
      débarrassée des échecs du retriever ;
    - **accuracy** : le produit des deux, le chiffre habituel.

    Un recall bas et une accuracy-si-récupéré haute : réindexer, augmenter `k`.
    L'inverse : retoucher le prompt. Le chiffre agrégé seul ne permet pas de
    choisir.
    """
    g = overall.get(regime, {})
    rows = []
    for k in LEVELS:
        recall = g.get(f"level_{k}_retrieval_accuracy")
        cond = g.get(f"level_{k}_generation_accuracy_when_retrieved")
        rows.append({
            "niveau": k,
            "recall du retriever": recall,
            "accuracy si récupéré": cond,
            "accuracy": g.get(f"level_{k}"),
            "manque au retriever": None if recall is None else 1.0 - recall,
        })
    return pd.DataFrame(rows).set_index("niveau")


def regime_table(
    overall: Dict, level: int = TARGET_LEVEL, *, has_retrieval: bool = True
) -> pd.DataFrame:
    """Accuracy par régime de réponse, au niveau demandé.

    Un RAG peut échouer de quatre façons distinctes : ne pas rendre de JSON
    exploitable, déclarer la ligne non codable, rendre un code avec une
    confiance basse, ou se tromper franchement. Les quatre sont comptées
    ensemble dans l'accuracy globale.
    """
    rows = []
    for key, label in REGIMES:
        g = overall.get(key)
        if not g:
            continue
        n = g.get("n_samples", 0)
        # `_metrics_for_group` renvoie 0.0 — et non None — quand aucun code n'a
        # été récupéré. Pour une brique sans retriever cela se lit « le
        # retriever ne ramène jamais rien », ce qui décrit un échec là où il n'y
        # a pas de retriever du tout. D'où `has_retrieval`, que l'appelant sait
        # et que le dictionnaire de métriques ne dit pas.
        recall = g.get(f"level_{level}_retrieval_accuracy") if (n and has_retrieval) else None
        rows.append({
            "régime": label,
            "n": n,
            "part": None,
            f"accuracy niv{level}": g.get(f"level_{level}") if n else None,
            f"recall niv{level}": recall,
        })
    out = pd.DataFrame(rows)
    if len(out):
        total = out.loc[out["régime"] == REGIMES[0][1], "n"]
        base = int(total.iloc[0]) if len(total) and total.iloc[0] else 0
        out["part"] = out["n"] / base if base else None
    return out.set_index("régime")


def confidence_table(records_by_classifier: Dict[str, List[Dict]]) -> pd.DataFrame:
    """Une confiance est utile si elle sépare le juste du faux.

    L'AUROC répond exactement à ça : probabilité qu'une prédiction correcte
    porte une confiance plus haute qu'une prédiction fausse. 0,5 = la confiance
    n'apprend rien et ne doit servir à aucun seuil ; au-dessus de 0,7 elle est
    exploitable pour trier ce qui part en relecture.
    """
    rows = []
    for name, records in records_by_classifier.items():
        rel = confidence_reliability(records, TARGET_LEVEL)
        if not rel["n"]:
            continue
        row = {
            "classifieur": name,
            "n": rel["n"],
            "AUROC": rel["auroc"],
            "conf. moy. si juste": rel["mean_conf_correct"],
            "conf. moy. si faux": rel["mean_conf_incorrect"],
        }
        row["écart"] = (
            None
            if row["conf. moy. si juste"] is None or row["conf. moy. si faux"] is None
            else row["conf. moy. si juste"] - row["conf. moy. si faux"]
        )
        rows.append(row)
    return pd.DataFrame(rows).set_index("classifieur") if rows else pd.DataFrame()


def threshold_sweep_table(records: List[Dict]) -> pd.DataFrame:
    """Le compromis couverture / exactitude : si on ne garde que les prédictions
    au-dessus d'un seuil, combien en reste-t-il et sont-elles meilleures ?

    C'est la table qui instruit une décision de relecture — pas l'AUROC, qui
    dit seulement si un seuil peut exister.
    """
    # Grille passée explicitement : le défaut de `confidence_reliability` est
    # (0.5 … 0.9) par pas de 0,1, et ce défaut est partagé avec `rag-annotations`
    # qui s'en sert pour ses propres mesures. Le resserrer ici, pas là-bas.
    rel = confidence_reliability(records, TARGET_LEVEL, thresholds=CONFIDENCE_SWEEP)
    if not rel["threshold_sweep"]:
        return pd.DataFrame()
    return pd.DataFrame(rel["threshold_sweep"]).rename(columns={
        "threshold": "seuil",
        "coverage": "couverture",
        "n_kept": "n gardées",
        "accuracy_kept": "accuracy gardées",
        "accuracy_dropped": "accuracy écartées",
    }).set_index("seuil")


def codable_table(records_by_classifier: Dict[str, List[Dict]]) -> pd.DataFrame:
    """Le drapeau `codable` est un refus déclaré : le modèle annonce qu'aucun
    candidat ne convient. Il n'a de valeur que s'il est plus souvent juste que
    l'ensemble — c'est le `lift`. Un lift ≈ 0 signifie que le modèle refuse au
    hasard, et que le drapeau ne doit pas servir de filtre.
    """
    rows = []
    for name, records in records_by_classifier.items():
        if not any(r.get("codable") is not None for r in records):
            continue
        rel = codable_reliability(records, TARGET_LEVEL)
        rows.append({
            "classifieur": name,
            "accuracy globale": rel["overall_accuracy"],
            "n codable=vrai": rel["true"]["n"],
            "accuracy si codable": rel["true"]["accuracy"],
            "couverture codable": rel["true"]["coverage"],
            "accuracy si non codable": rel["false"]["accuracy"],
            "lift": rel["lift"],
        })
    return pd.DataFrame(rows).set_index("classifieur") if rows else pd.DataFrame()


def distortion_table(records_by_classifier: Dict[str, List[Dict]], level: int) -> pd.DataFrame:
    """Un classifieur peut avoir une accuracy correcte tout en déformant la
    répartition des postes — sur-prédire une catégorie fréquente, en ignorer une
    rare. Invisible dans l'accuracy, visible ici.

    `TV` = distance en variation totale (0 = distributions identiques, 1 =
    disjointes) ; `KL` = divergence de Kullback-Leibler de la vérité vers la
    prédiction.
    """
    rows = []
    for name, records in records_by_classifier.items():
        usable = [r for r in records if r.get("codable") is not False]
        if not usable:
            continue
        d = distribution_distortion(usable, level)
        rows.append({
            "classifieur": name,
            f"TV niv{level}": d["tv_distance"],
            f"KL niv{level}": d["kl_divergence"],
        })
    return pd.DataFrame(rows).set_index("classifieur") if rows else pd.DataFrame()


def worst_distorted(records: List[Dict], level: int, top: int = 8) -> pd.DataFrame:
    """Les catégories les plus sur- et sous-prédites, celles qui font la
    distorsion. `diff` positif = le classifieur sur-prédit cette catégorie."""
    usable = [r for r in records if r.get("codable") is not False]
    if not usable:
        return pd.DataFrame()
    per_cat = pd.DataFrame(distribution_distortion(usable, level)["per_category"])
    if not len(per_cat) or "diff" not in per_cat.columns:
        return pd.DataFrame()
    per_cat = per_cat.reindex(per_cat["diff"].abs().sort_values(ascending=False).index)
    return per_cat.head(top).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Bout en bout : le chiffre métier
# ---------------------------------------------------------------------------

def end_to_end(
    con,
    deliverable_path: Optional[str],
    observations_path: Optional[str],
    mapping_path: Optional[str],
) -> Optional[Dict]:
    """Accuracy du fichier livré, **lignes captées par la regex comprises**.

    C'est le seul chiffre qui décrive ce que reçoit l'utilisateur : tous les
    autres tableaux du rapport partent du parquet de conciliation, où les lignes
    tranchées par la regex n'entrent jamais — elles sortent du circuit avant les
    classifieurs.

    PÉRIMÈTRE — le point qui rendait ce tableau faux. `export-results` construit
    le livrable sur la **totalité** de `observations.parquet`, alors que
    l'échantillonnage a lieu en aval, à `classify-regex`. Sur un run
    `sample-observations=100`, le livrable compte donc toujours ses ~16 000
    lignes, dont ~15 900 sans aucune prédiction — et une prédiction absente est
    comptée comme une erreur. Le tableau annonçait « 16 000 observations » avec
    une accuracy diluée d'autant, en la présentant comme le chiffre métier.

    Le tri se fait sur `prediction_source`, que le livrable porte déjà
    (`regex | consensus | llm | sirus`, vide si le run n'a pas codé la ligne) et
    que cette étape ne lisait pas. Les deux volumes sont renvoyés pour que
    l'écart reste visible plutôt qu'absorbé.

    Trois lectures parce que le livrable ne se suffit pas : `export-results`
    retire `code` et `code_lvl4` (ils sont dans son `PIPELINE_COLS`), la vérité
    est donc rejointe depuis `observations`, puis rendue canonique par le
    mapping — sans quoi une prédiction canonique juste serait comptée fausse.
    """
    if not (deliverable_path and observations_path and mapping_path):
        return None

    deliverable = _read(con, deliverable_path)
    observations = _read(con, observations_path)
    mapping = _read(con, mapping_path)
    if deliverable is None or observations is None or mapping is None:
        return None
    if "predicted_code" not in deliverable.columns or "id" not in deliverable.columns:
        return None
    if "code" not in observations.columns or "id" not in observations.columns:
        return None

    from prune_codes.pruning import trunc_and_prune_lvl4

    truth = observations[["id", "code"]].dropna(subset=["code"])
    truth = truth[truth["code"].astype(str).str.len() > 0]
    if not len(truth):
        return None
    truth = trunc_and_prune_lvl4(truth.copy(), mapping, code_name="code")
    truth_col = "code_tpruned" if "code_tpruned" in truth.columns else "code"

    keep = ["id", "predicted_code"]
    # Absente des runs antérieurs à `prediction_source` : on retombe alors sur
    # « une ligne décidée est une ligne portant un code », qui est la même chose
    # à ceci près qu'elle ne permet pas la ventilation par source.
    has_source = "prediction_source" in deliverable.columns
    if has_source:
        keep.append("prediction_source")
    livre = deliverable[keep].copy()

    decided = (
        livre[livre["prediction_source"].notna()]
        if has_source
        else livre[livre["predicted_code"].notna()]
    )

    merged = decided.merge(truth[["id", truth_col]], how="inner", on="id")
    if not len(merged):
        return None

    from codif_common.metrics import accuracy

    def _rows(frame: pd.DataFrame) -> Dict[int, tuple]:
        return {k: accuracy(frame[truth_col], frame["predicted_code"], k) for k in LEVELS}

    ensemble = _rows(merged)
    hors_regex = (
        _rows(merged[merged["prediction_source"] != "regex"]) if has_source else None
    )

    levels = pd.DataFrame(
        [
            {
                "niveau": k,
                "n": ensemble[k][1],
                "justes": ensemble[k][0],
                "accuracy livrée": ensemble[k][2],
                **(
                    {
                        "n hors regex": hors_regex[k][1],
                        "accuracy hors regex": hors_regex[k][2],
                        # L'apport de la regex, en points : c'est la question
                        # « qu'est-ce que l'étape regex change au chiffre
                        # global ? », posée dans la seule unité qui y réponde.
                        "apport regex": (
                            ensemble[k][2] - hors_regex[k][2]
                            if pd.notna(ensemble[k][2]) and pd.notna(hors_regex[k][2])
                            else None
                        ),
                    }
                    if hors_regex
                    else {}
                ),
            }
            for k in LEVELS
        ]
    ).set_index("niveau")

    by_source = None
    if has_source:
        rows = []
        for src, sub in merged.groupby("prediction_source"):
            n_ok, n_app, acc = accuracy(sub[truth_col], sub["predicted_code"], TARGET_LEVEL)
            rows.append(
                {
                    "source": src,
                    "n livré": len(sub),
                    "part du livré": len(sub) / len(merged),
                    "n avec vérité": n_app,
                    f"accuracy niv{TARGET_LEVEL}": acc,
                }
            )
        by_source = (
            pd.DataFrame(rows).sort_values("n livré", ascending=False).set_index("source")
        )

    return {
        "levels": levels,
        "by_source": by_source,
        "has_source": has_source,
        # Volumes, pour rendre visible l'écart d'un run échantillonné.
        "n_deliverable": len(livre),
        "n_decided": len(decided),
        "n_scored": len(merged),
    }


# ---------------------------------------------------------------------------
# Vers MLflow
# ---------------------------------------------------------------------------

def flatten_internal(records_by_classifier: Dict[str, List[Dict]]) -> Dict[str, float]:
    """Les mêmes indicateurs, aplatis en scalaires MLflow.

    Nommage `<indicateur>/<classifieur>/…` pour que l'UI MLflow les regroupe et
    qu'une comparaison entre deux runs porte sur des séries alignées.
    """
    flat: Dict[str, float] = {}
    for name, records in records_by_classifier.items():
        slug = name.lower().replace("-", "_")
        overall = hierarchical(records)

        has_retrieval = any(r.get("list_retrieved_codes") for r in records)
        raw = overall.get("all_raw", {})
        for k in LEVELS:
            if has_retrieval:
                flat[f"retrieval/{slug}/recall_niv{k}"] = raw.get(
                    f"level_{k}_retrieval_accuracy"
                )
                flat[f"retrieval/{slug}/generation_when_retrieved_niv{k}"] = raw.get(
                    f"level_{k}_generation_accuracy_when_retrieved"
                )

        for key, _label in REGIMES:
            g = overall.get(key)
            if not g or not g.get("n_samples"):
                continue
            flat[f"regime/{slug}/{key}/n"] = float(g["n_samples"])
            flat[f"regime/{slug}/{key}/accuracy_niv{TARGET_LEVEL}"] = g.get(
                f"level_{TARGET_LEVEL}"
            )

        rel = confidence_reliability(records, TARGET_LEVEL)
        if rel["n"]:
            flat[f"confidence/{slug}/auroc"] = rel["auroc"]
            flat[f"confidence/{slug}/mean_conf_correct"] = rel["mean_conf_correct"]
            flat[f"confidence/{slug}/mean_conf_incorrect"] = rel["mean_conf_incorrect"]
            for row in rel["threshold_sweep"]:
                pct = int(round(row["threshold"] * 100))
                flat[f"confidence/{slug}/accuracy_at_{pct}"] = row["accuracy_kept"]
                flat[f"confidence/{slug}/coverage_at_{pct}"] = row["coverage"]

        if any(r.get("codable") is not None for r in records):
            cod = codable_reliability(records, TARGET_LEVEL)
            flat[f"codable/{slug}/accuracy_true"] = cod["true"]["accuracy"]
            flat[f"codable/{slug}/coverage_true"] = cod["true"]["coverage"]
            flat[f"codable/{slug}/lift"] = cod["lift"]

        usable = [r for r in records if r.get("codable") is not False]
        for lvl in (1, 2):
            if not usable:
                continue
            d = distribution_distortion(usable, lvl)
            flat[f"distortion/{slug}/level_{lvl}/tv_distance"] = d["tv_distance"]
            flat[f"distortion/{slug}/level_{lvl}/kl_divergence"] = d["kl_divergence"]

    return {k: float(v) for k, v in flat.items() if v is not None}
