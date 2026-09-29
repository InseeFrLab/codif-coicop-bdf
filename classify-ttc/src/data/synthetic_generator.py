"""Génération de données de synthèse COICOP (produits « unseen »).

Génère des libellés produits de synthèse pour les codes COICOP, ancrés dans la
nomenclature RMES, et valide chaque item avec une copie scalaire de la chaîne
de nettoyage du training (``src.preprocessing.data_preparation.preprocess_text``).

Point d'entrée : ``python main.py generate-synthetic`` (voir le README). Les
dépendances LLM sont optionnelles : ``uv sync --extra synth``.

Contrat de sortie (consommé par ``src.data.build_training_data``):
    header: product;code;libelle
    reader: pd.read_csv(..., sep=';', skiprows=1, header=None, usecols=[0,1])
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import unidecode
from pydantic import BaseModel, Field, field_validator

# langchain-openai's structured-output path triggers a harmless pydantic
# serializer warning on ParsedChatCompletion.parsed (upstream issue).
warnings.filterwarnings(
    "ignore",
    message=r"Pydantic serializer warnings",
    category=UserWarning,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_COICOP_PATH = "data/coicop-2018_envoi_rmes_20251022.csv"
DEFAULT_STOPWORDS_PATH = "data/text/stopwords.json"
DEFAULT_OUTPUT_CSV = "data/synthetic_data.csv"
DEFAULT_RAW_DIR = "data/synthetic_raw"
DEFAULT_MANIFEST_PATH = "data/synthetic_manifest.json"
DEFAULT_REFERENCE_PATH = "data/20260130-coicop_et_codes_techniques.csv"
DEFAULT_EXAMPLES = 300
DEFAULT_LEVEL = 4
DEFAULT_TEMPERATURE = 0.8
DEFAULT_MAX_TOKENS = 8192
DEFAULT_RETRIES = 3
DEFAULT_MODEL = "gemma4-26b-moe"

logger = logging.getLogger(__name__)

# Band table (plan): DDC rows per level-4 code -> synthetic examples.
# >=1000 -> 0, 500-999 -> 100, 100-499 -> 200, 1-99 -> 300, 0/absent -> 400.
ALLOCATION_BANDS: list[tuple[int, int]] = [
    (1000, 0),
    (500, 100),
    (100, 200),
    (1, 300),
]
# Zero / absent codes get this many examples.
ALLOCATION_ZERO_BAND = 400


# ---------------------------------------------------------------------------
# Cleaning chain (scalar copy of src/preprocessing/data_preparation.py:preprocess_text)
#
# Scalar equivalent of: unidecode -> lower -> remove_noise ->
# tokenize_and_clean -> remove_empty_and_strip -> remove_stopwords.
# A row the pipeline drops (empty/whitespace) is returned as None.
# Kept scalar for speed (called per item, several times); equivalence with
# preprocess_text is pinned by tests/test_synthetic_generator.py.
# ---------------------------------------------------------------------------

_RIEN_RE = re.compile(r"\brien\b|\rien du tout\b")
_PUNCT_RE = re.compile(r"[^\w\s]+")
_DIGIT_PLUS_RE = re.compile(r"[\d+]")
_MULTI_SPACE_RE = re.compile(r"\s\s+")


def clean_product(text: str, stopwords: list[str]) -> str | None:
    """Nettoie un libellé produit avec la chaîne de training exacte.

    Retourne la chaîne nettoyée, ou ``None`` si la chaîne devient vide
    (ligne supprimée par le pipeline de training).

    Steps (ordre strict, identique à ``src.data_preparation.preprocess_text``):
        1. unidecode
        2. lower
        3. re.sub "\\brien\\b|\\rien du tout\\b" (regex copié tel quel)
        4. re.sub "[^\\w\\s]+" -> " " (ponctuation)
        5. mêmes regex "rien" (la chaîne les exécute vraiment 2 fois)
        6. re.sub "[\\d+]" -> " " (classe: chiffre ou '+', tel quel)
        7. drop mots d'une seule lettre
        8. collapse "\\s\\s+" -> " "
        9. dédup des tokens en conservant l'ordre d'apparition
       10. strip; vide/blanc -> None (check AVANT suppression des stop-words)
       11. drop des tokens présents dans ``stopwords``
    """
    t = unidecode.unidecode(text)
    t = t.lower()
    t = _RIEN_RE.sub("", t)
    t = _PUNCT_RE.sub(" ", t)
    t = _RIEN_RE.sub("", t)
    t = _DIGIT_PLUS_RE.sub(" ", t)
    t = " ".join(w for w in t.split() if len(w) > 1)
    t = _MULTI_SPACE_RE.sub(" ", t)
    tokens = t.split()
    tokens = sorted(set(tokens), key=tokens.index)  # dédup, ordre d'apparition
    t = " ".join(tokens)
    t = t.strip()
    if not t:  # vide/blanc uniquement -> None (check AVANT stop-words)
        return None
    t = " ".join(w for w in t.split() if w not in stopwords)
    return t


def _item_is_valid(product: str, stopwords: list[str]) -> bool:
    """Un item est valide s'il survit à la chaîne de nettoyage, mesure
    au plus 80 caractères après nettoyage (contrat du prompt v3) et ne contient
    pas de ``;`` ni de saut de ligne (le CSV de sortie est délimité par ``;``,
    une ligne par produit)."""
    if ";" in product or "\n" in product or "\r" in product:
        return False
    cleaned = clean_product(product, stopwords)
    return cleaned is not None and len(cleaned) <= 80


# ---------------------------------------------------------------------------
# Line parser (port of the legacy generator's line parser — tier 3)
# ---------------------------------------------------------------------------

_SKIP_PATTERNS = [
    "voici",
    "voilà",
    "exemples",
    "catégorie",
    "coicop",
    "produits pour",
    "produits de",
    "ci-dessous",
    "suivants",
]
_PREFIXES = ["- ", "• ", "* ", "– "]


def parse_response(response: str) -> list[str]:
    """Parse une réponse LLM en liste de libellés produits (tier 3, dernier).

    Port du parser de l'ancien générateur LangChain : skip des lignes
    d'introduction (skip_patterns), cutoff >80 caractères, stripping des
    puces et des numéros, conservation de l'ordre.
    """
    lines = response.strip().split("\n")
    products: list[str] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        line_lower = line.lower()
        if any(pattern in line_lower for pattern in _SKIP_PATTERNS):
            continue
        if len(line) > 80:
            continue
        for prefix in _PREFIXES:
            if line.startswith(prefix):
                line = line[len(prefix) :]
                break
        if len(line) > 2 and line[0].isdigit() and line[1] in [".", ")", ":"]:
            line = line[2:].strip()
        elif (
            len(line) > 3
            and line[0].isdigit()
            and line[1].isdigit()
            and line[2] in [".", ")", ":"]
        ):
            line = line[3:].strip()
        if line:
            products.append(line)
    return products


# ---------------------------------------------------------------------------
# JSON tier (tier 2)
# ---------------------------------------------------------------------------

_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


def _parse_json_response(text: str) -> list[str] | None:
    """Extrait le premier bloc tableau JSON de ``text``.

    Retourne une ``list[str]`` si le bloc est un tableau de chaînes non-vides,
    sinon ``None``. Utilise d'abord ``json.loads`` direct, puis un fallback
    regex sur le premier bloc ``[...]``.
    """
    if not text or not text.strip():
        return None
    block: str | None = None
    try:
        parsed = json.loads(text)
        block = text
    except (json.JSONDecodeError, ValueError):
        m = _JSON_ARRAY_RE.search(text)
        if m is None:
            return None
        block = m.group(0)
    if block is None:
        return None
    try:
        data = json.loads(block)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(data, list):
        return None
    if not all(isinstance(x, str) and x.strip() for x in data):
        return None
    return data


# ---------------------------------------------------------------------------
# Prompt v3 (style ticket brut — majuscules, sans accents, sans prix, léger bruit OCR)
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE = """Tu es un expert en classification des produits et services selon la
nomenclature COICOP (Classification of Individual Consumption According to Purpose).

Génère exactement {num_examples} libellés de produits réalistes, FRANÇAIS, pour la
catégorie COICOP suivante:

Code COICOP: {code}
Libellé: {libelle}
{comprend_section}
{ne_comprend_section}

STYLE EXACT — libellé D'UN TICKET DE CAISSE / BALISE EAN (le but est de générer des
libellés qui ressemblent à du libellé brut, PAS à un nom propre bien coiffé):
- TOUT EN MAJUSCULES, comme l'inscription sur un ticket de caisse français.
- SANS ACCENTS (libellé brut OCR/ticket) : ex. SUCRE, EPI, TILDA, NA, DUCRO, OEUFS.
- Marque FR réaliste quand c'est une marque connue (ex: ST JACQUES, GARDEL, TILDA,
  NA, DUCRO, EPI, KIRI, BONGRAIN, PRESIDENT, NESTLE, CARREFOUR) sinon produit générique.
- Produit + unité de poids/volume typique d'un ticket (500GRS, 750G, 1KG, 200G,
  3X20CL, X12, 6 X, 3X, 6 PORTION, LE KG).
- Quantité écrite comme multiplicateur (X12, 6 X, 3X) quand pertinent ; jamais un
  chiffre seul.
- Abréviations courantes possibles (ex: CFR, CP, NA).
- PAS DE PRIX, PAS DE SYMBOLE € : UN SEUL produit désigné, jamais de montant.
- UNE ligne sur vingt (±5%) seulement avec une petite imperfection OCR (DUCRO,
  HUIL au lieu d'HUILE, une lettre manquante) ; le produit doit rester identifiable.
- Aucune numérotation, aucune puce, pas de conclusion. Un seul produit par ligne.
- 3 à 80 caractères APRÈS nettoyage (ponctuation, chiffres et mots d'une lettre retirés).

Exemples de bonnes lignes (style ticket brut, sans prix) :
- ST JACQUES OEUFS FRAIS X12
- TILDA RIZ BASMATI 500GRS
- GARDEL SUCRE 750GRS
- TOMATE GRAPPE 500G
- EPI CURRY FORT 45G DUCRO
- KIRI 6 PORTION 100G

Maintenant, génère les {num_examples} libellés, un par ligne, SANS rien d'autre:"""

PROMPT_VERSION = "v3-" + hashlib.sha1(PROMPT_TEMPLATE.encode("utf-8")).hexdigest()[:8]


def build_prompt(
    code: str,
    libelle: str,
    num_examples: int,
    comprend: str | None = None,
    ne_comprend_pas: str | None = None,
) -> str:
    """Rend le prompt v3 pour une catégorie (sections RMES incluses si présentes)."""
    comprend_section = f"Cette catégorie comprend : {comprend}" if comprend else ""
    ne_comprend_section = (
        f"Cette catégorie NE comprend PAS : {ne_comprend_pas}"
        if ne_comprend_pas
        else ""
    )
    return PROMPT_TEMPLATE.format(
        num_examples=num_examples,
        code=code,
        libelle=libelle,
        comprend_section=comprend_section,
        ne_comprend_section=ne_comprend_section,
    )


# ---------------------------------------------------------------------------
# Technical codes 98.* / 99.* (saisie / hors champ COICOP)
#
# Not products: payment lines, discounts, unreadable lines, lump-sum entries...
# They get their own prompt with hand-written hints (examples taken from
# data/annotated/*.csv) and are never LLM-verified.
# ---------------------------------------------------------------------------

TECHNICAL_PREFIXES = ("98", "99")

# code -> (what the line is, example lines)
TECHNICAL_HINTS: dict[str, tuple[str, list[str]]] = {
    "98.1": (
        "ligne globale d'un panier de courses, sans détail des produits",
        ["COURSES", "COURSES LECLERC", "ACHATS MAGASIN", "TICKET COURSES"],
    ),
    "98.1.1": (
        "montant global de courses alimentaires saisi sans détail",
        ["ALIMENTATION", "ALIMENTAIRE", "COURSES ALIMENTAIRES", "NOURRITURE SEMAINE"],
    ),
    "98.1.1.1": (
        "achat global de fruits et légumes sans détail (marché, primeur)",
        ["FRUITS ET LEGUMES", "PRIMEUR", "MARCHE FRUITS LEGUMES", "FRUITS LEGUMES VRAC"],
    ),
    "98.1.2": (
        "courses globales faites en supermarché, hypermarché ou drive, sans détail",
        ["COURSES SUPERMARCHE", "HYPER U", "DRIVE LECLERC", "COURSES INTERMARCHE"],
    ),
    "98.1.3": (
        "courses non alimentaires globales (droguerie, bazar, bricolage), sans détail",
        ["COURSES NON ALIMENTAIRES", "DROGUERIE", "HYGIENE ENTRETIEN", "ACHATS ACTION"],
    ),
    "98.1.4": (
        "autres courses globales dont la nature n'est pas précisée",
        ["COURSES DIVERSES", "ACHATS DIVERS", "PETITES COURSES", "COURSES MARCHE"],
    ),
    "98.2": (
        "ligne trop générique pour être classée",
        ["DIVERS", "ARTICLE DIVERS", "RAYON DIVERS", "AUTRES ARTICLES"],
    ),
    "98.3": (
        "ligne de paiement par carte bancaire sur le ticket (pas un produit)",
        ["CARTE BANCAIRE", "CB", "PAIEMENT CB SANS CONTACT", "VISA", "REGLEMENT CARTE"],
    ),
    "98.4": (
        "ligne de ticket d'un PRODUIT dont le libellé est tellement abrégé ou "
        "tronqué qu'on ne peut pas savoir quel produit c'est (suites "
        "d'abréviations avec points), ou code interne du magasin. N'écris PAS de "
        "méta-libellés du type ARTICLE NON IDENTIFIE ; ILLISIBLE seul reste "
        "possible mais rare. Pas de réduction ni de paiement",
        ["ILLISIBLE", "ASS. BROCH EX. T NAT", "MAT CPL SSA", "PED.RNCH.BAR.PLT CAR",
         "SUPPLEMENTS FRAIS"],
    ),
    "98.5": (
        "réduction, remise, bon de réduction ou avantage fidélité sur le ticket",
        ["BON REDUCTION", "BON IMMEDIAT", "BRD SCANACHAT", "REMISE FIDELITE",
         "AVANTAGE CARTE"],
    ),
    "98.6": (
        "paiement avec le solde d'une carte de fidélité, cagnotte ou carte cadeau",
        ["UTILISATION CAGNOTTE", "SOLDE CARTE FIDELITE", "DEDUCTION CAGNOTTE",
         "PAIEMENT CARTE CADEAU"],
    ),
    "98.8": (
        "argent de poche donné aux enfants",
        ["ARGENT DE POCHE", "ARGENT POCHE ENFANTS", "ARGENT DE POCHE SEMAINE"],
    ),
    "99.1": (
        "cadeau offert ou don à une association, arrondi solidaire",
        ["DON CROIX ROUGE", "CADEAU ANNIVERSAIRE", "ARRONDI SOLIDAIRE",
         "DON RESTOS DU COEUR"],
    ),
    "99.2": (
        "opération bancaire : retrait d'espèces, virement, prélèvement",
        ["RETRAIT DAB", "VIREMENT", "PRELEVEMENT", "RETRAIT ESPECES BANQUE"],
    ),
    "99.3": (
        "gros travaux dans le logement (rénovation lourde, extension, toiture)",
        ["REFECTION TOITURE", "EXTENSION MAISON", "ACOMPTE MACON",
         "RENOVATION SALLE DE BAIN COMPLETE"],
    ),
    "99.4": (
        "impôts, taxes, cotisations syndicales ou politiques",
        ["TAXE FONCIERE", "IMPOT SUR LE REVENU", "COTISATION SYNDICALE",
         "ADHESION PARTI"],
    ),
    "99.5": (
        "somme prélevée directement sur le salaire par l'employeur",
        ["PRELEVEMENT EMPLOYEUR", "RETENUE SUR SALAIRE", "MUTUELLE ENTREPRISE"],
    ),
    "99.9": (
        "autre dépense hors du champ de la consommation (remboursement, caution, "
        "prêt entre particuliers)",
        ["REMBOURSEMENT AMI", "CAUTION", "PRET FAMILLE", "AUTRE"],
    ),
}

TECHNICAL_PROMPT_TEMPLATE = """Tu aides à entraîner un classifieur de lignes de dépenses de ménages
français (tickets de caisse et saisies manuelles dans un carnet de comptes).

Certaines lignes ne sont PAS des produits mais relèvent d'un code technique.
Génère exactement {num_examples} libellés réalistes, FRANÇAIS, pour le code suivant:

Code: {code}
Libellé: {libelle}
Ce que c'est: {description}

Exemples de lignes réelles pour ce code:
{examples}

STYLE:
- Mélange lignes de ticket de caisse (TOUT EN MAJUSCULES, SANS ACCENTS, abréviations)
  et saisies manuelles courtes d'un ménage (minuscules possibles).
- Enseigne ou banque FR réaliste quand c'est pertinent.
- PAS DE PRIX, PAS DE SYMBOLE €, pas de date.
- Aucune numérotation, aucune puce, pas de conclusion. Un seul libellé par ligne.
- 3 à 80 caractères APRÈS nettoyage (ponctuation, chiffres et mots d'une lettre retirés).
- Varie les formulations : ne recopie pas les exemples.

Maintenant, génère les {num_examples} libellés, un par ligne, SANS rien d'autre:"""

TECHNICAL_PROMPT_VERSION = (
    "t1-" + hashlib.sha1(TECHNICAL_PROMPT_TEMPLATE.encode("utf-8")).hexdigest()[:8]
)


def is_technical(code: str) -> bool:
    return str(code).split(".")[0] in TECHNICAL_PREFIXES


def build_technical_prompt(code: str, libelle: str, num_examples: int) -> str:
    """Rend le prompt pour un code technique 98/99 (hints de ``TECHNICAL_HINTS``)."""
    description, examples = TECHNICAL_HINTS.get(code, (libelle, []))
    return TECHNICAL_PROMPT_TEMPLATE.format(
        num_examples=num_examples,
        code=code,
        libelle=libelle,
        description=description,
        examples="\n".join(f"- {e}" for e in examples) or "- (aucun)",
    )


def render_prompt(
    code: str,
    libelle: str,
    num_examples: int,
    comprend: str | None = None,
    ne_comprend_pas: str | None = None,
) -> str:
    """Choisit le prompt technique (98/99) ou le prompt produit v3."""
    if is_technical(code):
        return build_technical_prompt(code, libelle, num_examples)
    return build_prompt(code, libelle, num_examples, comprend, ne_comprend_pas)


# ---------------------------------------------------------------------------
# Allocation
# ---------------------------------------------------------------------------


def allocate(
    ddc: pd.Series | dict | None, default: int = DEFAULT_EXAMPLES
) -> dict[str, int]:
    """Mappe les codes niveau-4 vers le nombre d'exemples de synthèse.

    Band table (``ALLOCATION_BANDS``):
        count >= 1000 -> 0
        500 <= count < 1000 -> 100
        100 <= count < 500 -> 200
        1 <= count < 100 -> 300
        count == 0 / absent -> 400

    Args:
        ddc: ``pd.Series`` (code -> nb lignes DDC) DICT {code: count}.
        default: valeur utilisée par l'appelant pour les codes sans ``--ddc``.

    Returns:
        ``dict[str, int]``: un entry ``{code: valeur}`` par code du Series/dict.
        ``default`` n'est pas stocké ici — l'appelant (CLI/générateur) le
        consomme via ``allocated.get(code, default)`` pour les codes sans DDC.
    """
    result: dict[str, int] = {}
    counts: dict[str, int]
    if isinstance(ddc, pd.Series):
        counts = ddc.to_dict()
    elif isinstance(ddc, dict):
        counts = ddc
    else:
        counts = {}
    for code, count in counts.items():
        n = int(count)
        if n == 0:
            result[code] = ALLOCATION_ZERO_BAND
            continue
        for upper, value in ALLOCATION_BANDS:
            if n >= upper:
                result[code] = value
                break
    return result


# ---------------------------------------------------------------------------
# Nomenclature loader
# ---------------------------------------------------------------------------


def load_coicop(path: str | Path) -> pd.DataFrame:
    """Chargement de la nomenclature RMES (CSV ``sep=";"`` ou ``.parquet``) avec renames.

    Renames: ``label_fr`` → ``libelle``, ``contenu_central_fr`` → ``comprend``,
    ``note_exclusion_fr`` → ``ne_comprend_pas``, ``note_generale_fr`` →
    ``description``. Gère aussi l'ancien format simple (libelle, code).
    """
    if str(path).lower().endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, sep=";", encoding="utf-8")
    if "label_fr" in df.columns:
        df = df.rename(
            columns={
                "label_fr": "libelle",
                "contenu_central_fr": "comprend",
                "note_exclusion_fr": "ne_comprend_pas",
                "note_generale_fr": "description",
            }
        )
    elif "comprend" not in df.columns:
        df.columns = ["libelle", "code"]
        df["comprend"] = None
        df["ne_comprend_pas"] = None
        df["description"] = None
    df["code"] = df["code"].astype(str)
    return df


def get_categories_by_level(
    df: pd.DataFrame, level: int, exclude_technical: bool = True
) -> pd.DataFrame:
    """Filtre les catégories à un niveau donné.

    Niveau par ``code.str.count(r'\\.') == level - 1``; exclut ``98.*``/``99.*``
    si ``exclude_technical`` (défaut True). Retourne le subset trié par code.
    """
    out = df.copy()
    mask = out["code"].str.count(r"\.") == (level - 1)
    out = out[mask].copy()
    if exclude_technical:
        out = out[~out["code"].str.startswith(("98", "99"))].copy()
    out = out.sort_values("code", kind="stable").reset_index(drop=True)
    return out


def load_technical_codes(path: str | Path) -> pd.DataFrame:
    """Codes techniques 98.*/99.* (hors racines ``98``/``99``) du fichier
    ``"Libelle";"Code"`` (ex: ``20260216-coicop_et_codes_techniques.csv``)."""
    df = pd.read_csv(path, sep=";", encoding="utf-8", dtype=str)
    df = df.rename(columns={"Libelle": "libelle", "Code": "code"})
    df = df[df["code"].map(is_technical) & df["code"].str.contains(".", regex=False)]
    df = df[["code", "libelle"]].copy()
    df["comprend"] = None
    df["ne_comprend_pas"] = None
    return df.sort_values("code", kind="stable").reset_index(drop=True)


def complete_from_reference(
    nomenclature: pd.DataFrame, reference_path: str | Path | None, level: int
) -> pd.DataFrame:
    """Ajoute à la nomenclature les codes du niveau ``level`` présents dans la
    liste de référence mais absents de la nomenclature.

    Le parquet élagué supprime les codes ``X.0`` fils uniques (ex: ``01.2.1.0``,
    et même ``08.2.0``/``08.2.0.0``): leur définition RMES est identique à celle
    du parent conservé. Chaque code manquant reçoit le libellé de la référence
    et ``comprend``/``ne_comprend_pas``/``description`` de sa propre ligne dans
    la nomenclature RMES complète (``DEFAULT_COICOP_PATH``) si elle existe,
    sinon de son plus proche ancêtre présent.
    """
    fields = ("comprend", "ne_comprend_pas", "description")
    if reference_path is None or not Path(reference_path).exists():
        logger.warning("reference %s not found; nomenclature not completed", reference_path)
        return nomenclature
    ref = pd.read_csv(reference_path, sep=";", encoding="utf-8", dtype=str)
    ref = ref.rename(columns={"Libelle": "libelle", "Code": "code"}).dropna(subset=["code"])
    ref = ref[
        (ref["code"].str.count(r"\.") == level - 1) & ~ref["code"].map(is_technical)
    ]
    by_code = nomenclature.set_index("code", drop=False)
    by_code = by_code[~by_code.index.duplicated()]
    full = None
    if Path(DEFAULT_COICOP_PATH).exists():
        full = load_coicop(DEFAULT_COICOP_PATH).set_index("code", drop=False)
        full = full[~full.index.duplicated()]
    rows = []
    for code, libelle in zip(ref["code"], ref["libelle"]):
        if code in by_code.index:
            continue
        row: dict[str, Any] = {"code": code, "libelle": libelle}
        parts = code.split(".")
        sources = [(full, code)] if full is not None else []
        sources += [(by_code, ".".join(parts[:i])) for i in range(len(parts) - 1, 0, -1)]
        for df, key in sources:
            if key not in df.index:
                continue
            src = df.loc[key]
            if all(pd.isna(src.get(col)) for col in fields):
                continue  # e.g. pruned parent whose notes were dropped
            for col in fields:
                row[col] = src.get(col)
            break
        rows.append(row)
    if not rows:
        return nomenclature
    logger.info(
        "%d level-%d codes absent from the nomenclature added from %s: %s",
        len(rows),
        level,
        reference_path,
        ", ".join(r["code"] for r in rows),
    )
    return pd.concat([nomenclature, pd.DataFrame(rows)], ignore_index=True)


def select_categories(
    nomenclature: pd.DataFrame,
    level: int,
    technical: str = "none",
    reference_path: str | Path | None = None,
    only_codes: list[str] | None = None,
    max_categories: int | None = None,
    exclude_technical: bool = True,
) -> pd.DataFrame:
    """Catégories à générer: niveau ``level`` de la nomenclature, et/ou codes
    techniques (``technical``: ``none`` | ``add`` | ``only``), filtrées par
    ``only_codes`` puis tronquées à ``max_categories``."""
    parts = []
    if technical != "only":
        parts.append(get_categories_by_level(nomenclature, level, exclude_technical))
    if technical in ("add", "only"):
        parts.append(load_technical_codes(reference_path or DEFAULT_REFERENCE_PATH))
    categories = pd.concat(parts, ignore_index=True)
    if only_codes:
        unknown = set(only_codes) - set(categories["code"])
        if unknown:
            raise ValueError(
                f"--codes not found (level {level}, technical={technical}): "
                f"{sorted(unknown)}"
            )
        categories = categories[categories["code"].isin(only_codes)]
    if max_categories is not None:
        categories = categories.head(max_categories)
    return categories.reset_index(drop=True)


# ---------------------------------------------------------------------------
# LLM factory (copied get_llm_from_env, extended with explicit max_tokens)
# ---------------------------------------------------------------------------


def get_llm(
    model_name: str | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
):
    """Crée un ``ChatOpenAI`` depuis les variables d'environnement.

    Env:
        OPENAI_API_KEY (requis), OPENAI_BASE_URL, OPENAI_MODEL (défaut
        ``DEFAULT_MODEL``). ``temperature`` et ``max_tokens`` sont explicites.
    """
    from langchain_openai import ChatOpenAI  # extra `synth`

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable is required for a real (non --dry-run) run"
        )
    base_url = os.environ.get("OPENAI_BASE_URL")
    model = model_name or os.environ.get("OPENAI_MODEL", DEFAULT_MODEL)
    logger.info(
        "Configuring LLM: base_url=%s model=%s temperature=%s max_tokens=%s",
        base_url,
        model,
        temperature,
        max_tokens,
    )
    return ChatOpenAI(
        api_key=api_key,
        base_url=base_url,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        # Bound a stuck gateway request (SDK default is 600 s x 3 tries = 30 min).
        # A 300-item batch legitimately takes ~5 min at ~13 tok/s, so keep 600 s
        # but retry only once; the tier fallback / retry loop takes over.
        timeout=600,
        max_retries=1,
        # insee-Qwen3.8-27B is a Qwen3 *thinking* model: with thinking on, each
        # call spends most of its time on reasoning_content and outlasts the
        # AGAIN gateway read timeout (504) on this pod. This task (short product
        # labels) needs no chain-of-thought, so disable it — vLLM honours
        # chat_template_kwargs.enable_thinking.
        #
        # `extra_body` (NOT `model_kwargs`): `model_kwargs` is merged into the
        # top-level payload, which `Completions.create` rejects — it silently
        # broke tiers 2/3 + verify ("unexpected keyword argument
        # 'chat_template_kwargs'"). `extra_body` routes through
        # `client.chat.completions.create(..., extra_body=...)` for every call
        # path (see langchain_openai.chat_models.base).
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )


# ---------------------------------------------------------------------------
# DDC allocation reader
# ---------------------------------------------------------------------------


def load_ddc_counts(ddc_path: str) -> dict[str, int]:
    """Lecture du snapshot DDC et counts par code niveau-4 (``code.str[:8]``).

    Retourne ``{code_l4: nb_lignes}``. La lecture est paresseuse (duckdb pour
    S3, pandas pour local) — jamais de copie locale des données.
    """
    from .build_training_data import _read_parquet

    df = _read_parquet(ddc_path)
    if "code" not in df.columns:
        raise ValueError(
            f"DDC file {ddc_path} has no 'code' column; got {list(df.columns)}"
        )
    code_l4 = df["code"].astype(str).str[:8]
    counts: dict[str, int] = code_l4.value_counts().to_dict()
    return {str(k): int(v) for k, v in counts.items()}


# ---------------------------------------------------------------------------
# Pydantic schema (tier 1)
# ---------------------------------------------------------------------------


class COICOPBatch(BaseModel):
    """Schema pour la sortie structurée (tier 1): liste de libellés produits.

    Chaque item doit mesurer 3 à 80 caractères (contrat du prompt v3). La
    validation finale (chaîne de nettoyage) est faite par le générateur après
    réception du batch; ici on borne uniquement la longueur brute."""

    products: list[str] = Field(
        description="List of French product labels for the COICOP category",
        min_length=1,
        max_length=500,
    )

    @field_validator("products")
    @classmethod
    def _check_item_length(cls, v: list[str]) -> list[str]:
        for p in v:
            if not isinstance(p, str) or not (3 <= len(p) <= 80):
                raise ValueError(f"item {p!r} must be 3–80 chars")
        return v


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class COICOPUnseenGenerator:
    """Générateur de produits de synthèse "vus" au niveau choisi.

    Pour chaque catégorie: tiers de sortie (structuré → JSON strict → line
    parser), validation par la chaîne de nettoyage, retry avec re-prompt,
    option ``verify`` (un appel LLM par catégorie), JSONL de provenance et
    manifest. Le LLM est fourni par injection (``llm``) — jamais construit au
    moment du dry-run.
    """

    def __init__(
        self,
        llm: Any = None,
        coicop_path: str | Path = DEFAULT_COICOP_PATH,
        examples_per_category: int = DEFAULT_EXAMPLES,
        stopwords_path: str | Path = DEFAULT_STOPWORDS_PATH,
        output_csv: str | Path = DEFAULT_OUTPUT_CSV,
        raw_dir: str | Path = DEFAULT_RAW_DIR,
        manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
        level: int = DEFAULT_LEVEL,
        verify: bool = True,
        retries: int = DEFAULT_RETRIES,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        exclude_technical: bool = True,
    ) -> None:
        self.llm = llm
        self.coicop_path = Path(coicop_path)
        self.examples_per_category = examples_per_category
        self.stopwords_path = Path(stopwords_path)
        self.output_csv = Path(output_csv)
        self.raw_dir = Path(raw_dir)
        self.manifest_path = Path(manifest_path)
        self.level = level
        self.verify_enabled = verify
        self.retries = retries
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.exclude_technical = exclude_technical
        # Loaded once (the tests may reassign gen.stopwords directly).
        with open(self.stopwords_path, encoding="utf-8") as f:
            self.stopwords: list[str] = json.load(f)
        self._coicop_df: pd.DataFrame | None = None
        self._manifest: dict[str, Any] = {}
        self._manifest_lock = threading.Lock()
        self._load_existing_manifest()

    @property
    def coicop_df(self) -> pd.DataFrame:
        """Lazy load of the nomenclature."""
        if self._coicop_df is None:
            self._coicop_df = load_coicop(self.coicop_path)
        return self._coicop_df

    # --- per-category -----------------------------------------------------

    def _get_row(self, code: str) -> pd.Series | None:
        row = self.coicop_df[self.coicop_df["code"] == code]
        if row.empty:
            return None
        return row.iloc[0]

    def generate_category(
        self,
        code: str,
        libelle: str,
        num: int | None = None,
        verify: bool | None = None,
        force: bool = False,
    ) -> list[str]:
        """Génère pour une catégorie: items valides et uniques (texte brut).

        Args:
            code: code COICOP (niveau ``self.level``).
            libelle: label FR de la catégorie.
            num: nombre d'exemples demandés (défaut ``examples_per_category``).
                Le nombre rendu peut dépasser ``num`` si l'appel LLM en fournit
                davantage et qu'ils sont valides/uniques — le retry re-prompt
                seulement pour combler un déficit.
            verify: surcharge pour vérifier (défaut ``self.verify_enabled``).
            force: régénérer même si le code est déjà dans le manifest.

        Returns:
            Liste des libellés produits valides et uniques (texte brut).
        """
        if num is None:
            num = self.examples_per_category
        verify = self.verify_enabled if verify is None else verify
        if is_technical(code):
            verify = False  # a category-membership check is meaningless for 98/99

        self.raw_dir.mkdir(parents=True, exist_ok=True)
        raw_file = self.raw_dir / f"{code}.jsonl"
        done = self._code_in_manifest(code)
        if done and not force:
            logger.info("resume: skipping %s (already in manifest)", code)
            return self._read_existing_raw(raw_file)

        row = self._get_row(code)
        comprend = None
        ne_comprend_pas = None
        if row is not None:
            c = row.get("comprend")
            comprend = (
                None if c is None or (isinstance(c, float) and pd.isna(c)) else str(c)
            )
            n = row.get("ne_comprend_pas")
            ne_comprend_pas = (
                None if n is None or (isinstance(n, float) and pd.isna(n)) else str(n)
            )

        # Resume: reuse previously-accepted items unless force regenerates from scratch.
        accepted: list[str] = [] if force else self._read_existing_raw(raw_file)
        t0 = time.monotonic()
        tier: str | None = None
        attempts_used = 0
        verify_dropped = 0

        max_attempts = self.retries + 1
        for attempt in range(1, max_attempts + 1):
            attempts_used = attempt
            items = self._call_with_tiers(
                code, libelle, num, comprend, ne_comprend_pas, accepted
            )
            valid = [p for p in items if _item_is_valid(p, self.stopwords)]
            unique = self._dedup_ordered(accepted, valid)
            if verify:
                verified = self._verify_items(
                    unique, code, libelle, comprend, ne_comprend_pas
                )
            else:
                verified = unique
            if verify:
                verify_dropped += len(unique) - len(verified)
            accepted = verified
            if self._last_tier is not None:
                tier = tier or self._last_tier
            if len(accepted) >= num:
                break
            if attempt < max_attempts:
                time.sleep(min(4, 2 ** (attempt - 1)))  # 1s, 2s, 4s backoff

        self._write_raw(raw_file, accepted, tier or "lines", attempts_used, verify)
        self._record_manifest(
            code, libelle, num, len(accepted), verify_dropped, tier or "lines"
        )
        logger.info(
            "%s: %d/%d accepted, %d attempt(s), tier=%s, %.0fs",
            code,
            len(accepted),
            num,
            attempts_used,
            tier or "lines",
            time.monotonic() - t0,
        )
        return accepted

    # --- tiers ------------------------------------------------------------

    def _call_with_tiers(
        self,
        code: str,
        libelle: str,
        num: int,
        comprend: str | None,
        ne_comprend_pas: str | None,
        already: list[str],
    ) -> list[str]:
        """Tries tiers 1 → 2 → 3; returns raw candidate items (unordered)."""
        self._last_tier = "structured"
        prompt = render_prompt(code, libelle, num, comprend, ne_comprend_pas)
        if already:
            header = (
                "Nouveaux produits, STRICTEMENT différents de ces "
                "exemples déjà acceptés:\n"
            )
            already_block = header + "\n".join(already[:50])
            prompt = f"{prompt}\n\n{already_block}"

        # Tier 1: structured output
        try:
            structured_llm = self.llm.with_structured_output(COICOPBatch)
            batch = structured_llm.invoke(prompt)
            if batch is None:
                raise ValueError("structured output returned None")
            products = batch.products if hasattr(batch, "products") else list(batch)
            if products and all(_item_is_valid(p, self.stopwords) for p in products):
                self._last_tier = "structured"
                return list(products)
            # batch returned but items are not all valid -> fall through to tier 2
        except Exception:  # NotImplementedError / ValidationError / any gateway error
            self._last_tier = "structured"
            logger.debug(
                "tier 1 (structured) unavailable for %s, falling through", code
            )

        # Tier 2: strict JSON
        self._last_tier = "json"
        json_prompt = (
            f"{prompt}\n\n"
            f"Réponds UNIQUEMENT avec un tableau JSON de {num} chaînes, rien d'autre. "
            "Pas d'introduction, pas de conclusion, pas de texte hors du tableau."
        )
        try:
            resp = self.llm.invoke(json_prompt)
            text = resp.content if hasattr(resp, "content") else str(resp)
            parsed = _parse_json_response(text)
            if parsed:
                self._last_tier = "json"
                return list(parsed)
        except Exception:
            pass

        # Tier 3: line parser
        self._last_tier = "lines"
        try:
            resp = self.llm.invoke(prompt)
            text = resp.content if hasattr(resp, "content") else str(resp)
            return parse_response(text)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("tier 3 failed for %s: %s", code, exc)
            return []

    # --- verify -----------------------------------------------------------

    def _verify_items(
        self,
        items: list[str],
        code: str,
        libelle: str,
        comprend: str | None,
        ne_comprend_pas: str | None,
    ) -> list[str]:
        """Un appel LLM batché pour vérifier l'appartenance de chaque item."""
        if not items:
            return []
        context = ""
        if comprend:
            context += f"Cette catégorie comprend : {comprend}\n"
        if ne_comprend_pas:
            context += f"Cette catégorie NE comprend PAS : {ne_comprend_pas}\n"
        prompt = (
            f"Categorie: {code} - {libelle}\n"
            f"{context}\n"
            f"Pour chaque produit ci-dessous (même ordre que la liste), réponds sur la ligne "
            "correspondante 1 (appartient à cette catégorie) ou 0 (n'y appartient pas).\n"
            "Une réponse par ligne, un seul chiffre, rien d'autre.\n\n"
            + "\n".join(items)
        )
        try:
            resp = self.llm.invoke(prompt)
            text = resp.content if hasattr(resp, "content") else str(resp)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("verify failed for %s (%s); keeping all items", code, exc)
            return list(items)
        bits = [ln.strip() for ln in text.strip().splitlines()]
        kept: list[str] = []
        for i, it in enumerate(items):
            bit = bits[i] if i < len(bits) else "1"
            keep = bit.startswith("1")
            if keep:
                kept.append(it)
        if not kept:
            # Rejecting a whole batch is far more likely a verifier failure
            # than every generated product being wrong: keep the batch.
            logger.warning(
                "verify rejected all %d items for %s; keeping them", len(items), code
            )
            return list(items)
        return kept

    # --- dedup / raw / manifest -------------------------------------------

    def _dedup_ordered(self, accepted: list[str], new_items: list[str]) -> list[str]:
        """Merges ``new_items`` into ``accepted``, keeping first occurrence on
        cleaned text, preserving order."""
        seen: set[str] = set()
        out: list[str] = []
        for p in list(accepted) + list(new_items):
            cleaned = clean_product(p, self.stopwords)
            key = (cleaned or "").strip()
            if not key:
                continue
            if key in seen:
                continue
            seen.add(key)
            out.append(p)
        return out

    def _read_existing_raw(self, raw_file: Path) -> list[str]:
        if not raw_file.exists():
            return []
        out: list[str] = []
        with raw_file.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue
                if "product" in obj:
                    # Items written before newlines were rejected may contain one.
                    product = " ".join(obj["product"].split())
                    if product:
                        out.append(product)
        return out

    def _code_in_manifest(self, code: str) -> bool:
        return self._manifest.get("per_code", {}).get(code) is not None

    def _write_raw(
        self,
        raw_file: Path,
        items: list[str],
        tier: str,
        attempts: int,
        verify: bool,
    ) -> None:
        with raw_file.open("w", encoding="utf-8") as f:
            for p in items:
                f.write(
                    json.dumps(
                        {
                            "product": p,
                            "mode": tier,
                            "attempts": attempts,
                            "verify": verify,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    def _record_manifest(
        self,
        code: str,
        libelle: str,
        requested: int,
        accepted: int,
        verify_dropped: int,
        tier: str,
    ) -> None:
        self._manifest.setdefault("per_code", {})[code] = {
            "requested": requested,
            "accepted": accepted,
            "verify_dropped": verify_dropped,
            "tier": tier,
        }
        self._manifest.setdefault("codes", {})[code] = libelle
        # Persist after each category so a killed run resumes without redoing codes.
        with self._manifest_lock:
            self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
            self.manifest_path.write_text(
                json.dumps(self._manifest, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    def _load_existing_manifest(self) -> None:
        if not self.manifest_path.exists():
            return
        try:
            self._manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("could not read manifest %s: %s", self.manifest_path, exc)
            self._manifest = {}


# ---------------------------------------------------------------------------
# Curation (deterministic, post-generation)
# ---------------------------------------------------------------------------


def curate(
    per_code: dict[str, list[str]],
    libelles: dict[str, str] | None = None,
    stopwords: list[str] | None = None,
    logger: logging.Logger | None = None,
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    """Curation globale déterministe (plan, post-génération).

    1. Drop des items invalides (``clean_product`` None ou >80 caractères).
    2. Dédup global sur le TEXTE NETTOYÉ, en gardant le premier en ordre de
       code trié; un conflit cross-code conserve le premier et est loggué au
       manifest (``cross_code_conflicts``).

    Args:
        per_code: ``{code: [produits bruts]}``.
        libelles: ``{code: label_fr}`` (optionnel, non utilisé pour le dedup
          mais conservé à des fins de traçabilité).
        stopwords: liste des stop-words (défaut: fichier par défaut).
        logger: logger à utiliser pour logguer les conflits.

    Returns:
        ``(cleaned_per_code, manifest_fragment)`` où
        ``manifest_fragment = {"cross_code_conflicts": [...]}``.
    """
    if stopwords is None:
        with open(DEFAULT_STOPWORDS_PATH, encoding="utf-8") as f:
            stopwords = json.load(f)
    log = logger or logging.getLogger(__name__)
    cleaned: dict[str, list[str]] = {}
    seen: dict[str, str] = {}  # cleaned text -> first code (sorted order)
    conflicts: list[dict[str, str]] = []
    for code in sorted(per_code):
        kept: list[str] = []
        for p in per_code[code]:
            c = clean_product(p, stopwords)
            if c is None or len(c) > 80:
                continue
            if c in seen:
                conflicts.append(
                    {
                        "product": p,
                        "cleaned": c,
                        "first_code": seen[c],
                        "dropped_code": code,
                    }
                )
                log.warning(
                    "cross-code conflict: %r kept under %s, dropped under %s",
                    c,
                    seen[c],
                    code,
                )
                continue
            seen[c] = code
            kept.append(p)
        cleaned[code] = kept
    return cleaned, {"cross_code_conflicts": conflicts}


# ---------------------------------------------------------------------------
# CSV writer (consumer contract: header product;code;libelle, raw text)
# ---------------------------------------------------------------------------


def write_csv(
    output_csv: str | Path,
    per_code: dict[str, list[str]],
    libelles: dict[str, str],
) -> Path:
    """Écrit le CSV final ``product;code;libelle`` (Avec header, ``;`` séparateur).

    Texte brut des produits (le consumer repasse sa propre préproc).
    Dédup exact (product, code) par catégorie avant écriture.
    Le reader consomme: ``pd.read_csv(..., sep=';', skiprows=1, header=None,
    usecols=[0,1])``.
    """
    path = Path(output_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["product;code;libelle"]
    for code in sorted(per_code):
        libelle = libelles.get(code, "")
        seen: set[str] = set()
        for p in per_code[code]:
            if p in seen:
                continue
            seen.add(p)
            lines.append(f"{p};{code};{libelle}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def generate_and_save(
    llm: Any | None = None,
    coicop_path: str | Path = DEFAULT_COICOP_PATH,
    examples_per_category: int = DEFAULT_EXAMPLES,
    stopwords_path: str | Path = DEFAULT_STOPWORDS_PATH,
    output_csv: str | Path = DEFAULT_OUTPUT_CSV,
    raw_dir: str | Path = DEFAULT_RAW_DIR,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    level: int = DEFAULT_LEVEL,
    verify: bool = True,
    retries: int = DEFAULT_RETRIES,
    max_categories: int | None = None,
    only_codes: list[str] | None = None,
    max_workers: int = 1,
    ddc_path: str | None = None,
    force: bool = False,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    exclude_technical: bool = True,
    technical: str = "none",
    reference_path: str | Path = DEFAULT_REFERENCE_PATH,
) -> Path:
    """Orchestration complète: allocation, génération par catégorie (workers
    ``max_workers``), curation, écriture du CSV + manifest + JSONL.

    Le LLM est créé en lazy (ici), après le dry-run — ``llm`` peut être injecté
    (tests) et ``get_llm`` n'est appelé que si ``llm is None``.
    """
    if llm is None:
        llm = get_llm(temperature=temperature, max_tokens=max_tokens)

    generator = COICOPUnseenGenerator(
        llm=llm,
        coicop_path=coicop_path,
        examples_per_category=examples_per_category,
        stopwords_path=stopwords_path,
        output_csv=output_csv,
        raw_dir=raw_dir,
        manifest_path=manifest_path,
        level=level,
        verify=verify,
        retries=retries,
        temperature=temperature,
        max_tokens=max_tokens,
        exclude_technical=exclude_technical,
    )
    generator._load_existing_manifest()

    # Allocation
    if ddc_path:
        logger.info("Loading DDC allocation from %s", ddc_path)
        counts = load_ddc_counts(ddc_path)
        allocation = allocate(pd.Series(counts), default=examples_per_category)
        alloc_kind = "ddc-bands"
    else:
        allocation = allocate({}, default=examples_per_category)
        alloc_kind = "uniform"

    nomenclature = complete_from_reference(generator.coicop_df, reference_path, level)
    generator._coicop_df = nomenclature  # so _get_row finds the added codes
    categories = select_categories(
        nomenclature,
        level,
        technical=technical,
        reference_path=reference_path,
        only_codes=only_codes,
        max_categories=max_categories,
        exclude_technical=exclude_technical,
    )

    codes = categories["code"].tolist()
    libelles = dict(zip(categories["code"], categories["libelle"].astype(str)))

    def _one(code: str) -> tuple[str, list[str]]:
        return code, generator.generate_category(
            code,
            libelles[code],
            allocation.get(code, examples_per_category),
            force=force,
        )

    per_code: dict[str, list[str]] = {}
    if max_workers <= 1:
        for code in codes:
            _, items = _one(code)
            per_code[code] = items
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            for code, items in ex.map(_one, codes):
                per_code[code] = items

    # Curation (deterministic global dedup + cross-code conflicts)
    cleaned, manifest_fragment = curate(
        per_code, libelles=libelles, stopwords=generator.stopwords
    )

    # Write CSV (raw product text, dedup exact per code)
    write_csv(output_csv, cleaned, libelles)

    # Manifest (provenance)
    manifest = dict(generator._manifest)
    model = (
        getattr(llm, "model_name", None) or getattr(llm, "model", None) or DEFAULT_MODEL
    )
    base_url = (
        getattr(llm, "openai_api_base", None)
        or getattr(llm, "base_url", None)
        or os.environ.get("OPENAI_BASE_URL", "")
    )
    manifest.update(
        {
            "model": model,
            "base_url": _host_only(base_url),
            "prompt_version": PROMPT_VERSION,
            "technical_prompt_version": TECHNICAL_PROMPT_VERSION,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "verify": verify,
            "allocation": alloc_kind,
        }
    )
    manifest["per_code"] = manifest.get("per_code", {})
    manifest["cross_code_conflicts"] = manifest_fragment.get("cross_code_conflicts", [])
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    n_rows = sum(len(v) for v in cleaned.values())
    logger.info("Wrote %d rows to %s (manifest %s)", n_rows, output_csv, manifest_path)
    return Path(output_csv)


def _host_only(url: str | None) -> str:
    """Retourne seulement le HOST de l'URL (jamais le chemin, jamais la clé)."""
    if not url:
        return ""
    u = url.split("//")[-1]
    return u.split("/")[0]


# ---------------------------------------------------------------------------
# Dry run (prompts only, no LLM call)
# ---------------------------------------------------------------------------


def dry_run(
    coicop_path: str | Path = DEFAULT_COICOP_PATH,
    examples_per_category: int = DEFAULT_EXAMPLES,
    level: int = DEFAULT_LEVEL,
    max_categories: int | None = None,
    only_codes: list[str] | None = None,
    technical: str = "none",
    reference_path: str | Path = DEFAULT_REFERENCE_PATH,
) -> int:
    """Affiche les prompts des premières catégories sans appeler le LLM.

    ``max_categories`` borne le nombre de prompts affichés (défaut 3).
    Retourne le nombre de prompts rendus. Lève ``ValueError`` si ``only_codes``
    contient un code inconnu.
    """
    nomenclature = complete_from_reference(
        load_coicop(coicop_path), reference_path, level
    )
    categories = select_categories(
        nomenclature,
        level,
        technical=technical,
        reference_path=reference_path,
        only_codes=only_codes,
    )
    n_show = max_categories if max_categories is not None else 3
    shown = 0
    for _, row in categories.head(n_show).iterrows():
        prompt = render_prompt(
            row["code"],
            str(row["libelle"]),
            examples_per_category,
            None if pd.isna(row.get("comprend")) else str(row.get("comprend")),
            None
            if pd.isna(row.get("ne_comprend_pas"))
            else str(row.get("ne_comprend_pas")),
        )
        print(f"===== PROMPT {shown + 1}/{n_show} — code {row['code']} =====")
        print(prompt)
        shown += 1
    logger.info("dry-run complete: %d prompts rendered (no LLM call)", shown)
    return shown
