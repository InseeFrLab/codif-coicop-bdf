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


def regime_table(overall: Dict, level: int = TARGET_LEVEL) -> pd.DataFrame:
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
        rows.append({
            "régime": label,
            "n": n,
            "part": None,
            f"accuracy niv{level}": g.get(f"level_{level}") if n else None,
            f"recall niv{level}": g.get(f"level_{level}_retrieval_accuracy") if n else None,
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
    rel = confidence_reliability(records, TARGET_LEVEL)
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
) -> Optional[pd.DataFrame]:
    """Accuracy du livrable, **lignes captées par la regex comprises**.

    C'est le seul chiffre qui décrive ce que reçoit l'utilisateur. Tous les
    autres tableaux de ce rapport partent du parquet de conciliation, où les
    lignes tranchées par la regex n'entrent jamais : elles sortent du circuit
    avant les classifieurs. Elles sont peu nombreuses mais très justes, donc
    l'accuracy de bout en bout est mécaniquement supérieure — et c'est celle
    qu'il faut annoncer.

    Trois lectures, parce que le livrable ne se suffit pas : `export-results`
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

    merged = deliverable[["id", "predicted_code"]].merge(
        truth[["id", truth_col]], how="inner", on="id"
    )
    if not len(merged):
        return None

    from codif_common.metrics import accuracy

    rows = []
    for k in LEVELS:
        n_ok, n_app, acc = accuracy(
            merged[truth_col], merged["predicted_code"], k, inclusive=True
        )
        rows.append({"niveau": k, "n": n_app, "justes": n_ok, "accuracy": acc})
    return pd.DataFrame(rows).set_index("niveau")


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
