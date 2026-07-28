"""Synthetic data generation for COICOP classification using LLM.

Refactored according to Code2Text report (Matéo Morin, SSP Lab, 2026):
- Multi-call generation (5 calls × N products = ~50 products/category)
- Few-shot from real BDF annotations (tickets, carnets, ajouts manuels)
- Explicit style constraints (uppercase, BDF units, no punctuation)
- Post-processing style enforcement (_style_enforce)
- S3 support for reading annotations and writing output
"""

from __future__ import annotations

import json
import logging
import os
import re
import unicodedata
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import duckdb
import pandas as pd
import unidecode
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)

STOPWORDS_PATH = "data/text/stopwords.json"
DEFAULT_OUTPUT_PREFIX = (
    "s3://travail/projet-ml-classification-bdf/"
    "confidentiel/personnel_sensible/synthetic-data"
)


class COICOPExample(BaseModel):
    """Schema for a synthetic COICOP product example."""

    product: str = Field(
        description="Product or service name/description in French"
    )
    code: str = Field(description="COICOP code (e.g., '01.1.1.1.1')")
    libelle: str = Field(description="COICOP category label in French")


# ── S3 helpers (same pattern as extract_ddc.py / predict.py) ──────────────────


def _configure_s3(con: duckdb.DuckDBPyConnection) -> None:
    """Configure DuckDB S3 secret from environment variables."""
    con.execute(f"""
        CREATE SECRET secret_s3 (
            TYPE S3,
            KEY_ID '{os.environ["AWS_ACCESS_KEY_ID"]}',
            SECRET '{os.environ["AWS_SECRET_ACCESS_KEY"]}',
            ENDPOINT '{os.environ["AWS_S3_ENDPOINT"]}',
            SESSION_TOKEN '{os.environ["AWS_SESSION_TOKEN"]}',
            REGION 'us-east-1',
            URL_STYLE 'path',
            SCOPE 's3://'
        );
    """)


def _is_s3_path(path: str) -> bool:
    """Check if a path is an S3 URI."""
    return path.strip().startswith("s3://")


def _read_csv_from_path(file_path: str) -> pd.DataFrame:
    """Read CSV from local path or S3 URL, auto-detecting separator."""
    if _is_s3_path(file_path):
        con = duckdb.connect()
        _configure_s3(con)
        # READ_CSV_AUTO with auto-detect of separator
        result = con.execute(
            f"SELECT * FROM READ_CSV_AUTO('{file_path}')"
        ).df()
        result.columns = [c.strip() for c in result.columns]
        return result

    # Auto-detect separator: try ';' first (majority of BDF files), fallback to ','
    with open(file_path, "r", encoding="utf-8") as f:
        first_line = f.readline()

    sep = ";" if first_line.count(";") > first_line.count(",") else ","

    try:
        return pd.read_csv(file_path, sep=sep, encoding="utf-8")
    except Exception:
        # Fallback: auto-detect with pandas
        return pd.read_csv(file_path, sep=None, engine="python")


def _write_parquet_to_path(df: pd.DataFrame, output_path: str) -> None:
    """Write DataFrame to parquet, local or S3."""
    if _is_s3_path(output_path):
        con = duckdb.connect()
        _configure_s3(con)
        con.register("__output_df", df)
        con.execute(f"COPY __output_df TO '{output_path}' (FORMAT PARQUET)")
    else:
        df.to_parquet(output_path, index=False)


# ── LLM setup ────────────────────────────────────────────────────────────────


def get_llm_from_env(model_name: str | None = None) -> ChatOpenAI:
    """Create LLM instance from environment variables.

    Args:
        model_name: Model name override. If None, uses OPENAI_MODEL env var.

    Environment variables:
        LLMLAB_API_KEY: API key for OpenAI-compatible endpoint
        LLMLAB_URL: Base URL for API (optional, defaults to OpenAI)
        OPENAI_MODEL: Model name (optional, defaults to gpt-oss:20b)

    Returns:
        Configured ChatOpenAI instance
    """
    api_key = os.environ.get("LLMLAB_API_KEY")
    if not api_key:
        msg = "LLMLAB_API_KEY environment variable is required"
        raise ValueError(msg)

    base_url = os.environ.get("LLMLAB_URL")
    model_name = model_name or os.environ.get("OPENAI_MODEL", "gpt-oss:20b")

    logger.info("Configuring LLM: base_url=%s model=%s", base_url, model_name)

    return ChatOpenAI(
        api_key=api_key,
        base_url=base_url,
        model=model_name,
        temperature=0.8,
    )


# ── Style enforcement ────────────────────────────────────────────────────────


class COICOPSyntheticGenerator:
    """Generator for synthetic COICOP classification training data.

    Refactored per Code2Text report to reduce covariate shift:
    - 5 calls per category with few-shot real examples
    - Explicit style constraints in prompts
    - Post-processing style enforcement (uppercase, BDF units, etc.)
    """

    def __init__(
        self,
        llm: BaseChatModel | None = None,
        coicop_path: str | Path = "data/coicop_et_codes_techniques.csv",
        rmes_path: str | Path | None = "data/coicop-2018_envoi_rmes_20251022.csv",
        annotations_s3_paths: dict[str, str] | None = None,
        examples_per_call: int = 10,
        calls_per_category: int = 5,
        fewshot_per_category: int = 6,
        style_enforce: bool = True,
    ) -> None:
        """Initialize the synthetic data generator.

        Args:
            llm: LangChain chat model (defaults to OpenAI from env vars)
            coicop_path: Path to COICOP definitions CSV (with 98/99 codes)
            rmes_path: Path to RMES COICOP file for enriched descriptions (optional)
            annotations_s3_paths: Mapping {source_name: path} for real annotations.
                Keys: 'tickets', 'carnets', 'ajouts'.
                Paths can be local or S3 URIs (s3://...).
            examples_per_call: Number of examples LLM generates per API call.
            calls_per_category: LLM calls per category (5 = ~50 products/cat).
            fewshot_per_category: Real examples injected per category in prompt.
            style_enforce: Apply BDF style post-processing (True by default).
        """
        self.llm = llm if llm is not None else get_llm_from_env()
        self.coicop_path = Path(coicop_path)
        self.rmes_path = Path(rmes_path) if rmes_path else None
        self.annotations_s3_paths = annotations_s3_paths or {}
        self.examples_per_call = examples_per_call
        self.calls_per_category = calls_per_category
        self.fewshot_per_category = fewshot_per_category
        self.style_enforce = style_enforce
        self._coicop_df: pd.DataFrame | None = None
        self._annotations: dict[str, list[str]] | None = None

    # ── CoICOP hierarchy ─────────────────────────────────────────────────

    @property
    def coicop_df(self) -> pd.DataFrame:
        """Lazy load the COICOP hierarchy."""
        if self._coicop_df is None:
            self._coicop_df = self._load_coicop()
        return self._coicop_df

    def _load_coicop(self) -> pd.DataFrame:
        """Load COICOP hierarchy from CSV, optionally enriched with RMES."""
        df = pd.read_csv(self.coicop_path, sep=";", encoding="utf-8")

        # Detect and normalize formats
        if "label_fr" in df.columns:
            df = df.rename(columns={
                "label_fr": "libelle",
                "contenu_central_fr": "comprend",
                "note_exclusion_fr": "ne_comprend_pas",
                "note_generale_fr": "description",
            })
        elif "comprend" not in df.columns:
            if "Libelle" in df.columns:
                df = df.rename(columns={"Libelle": "libelle", "Code": "code"})
            else:
                df.columns = ["libelle", "code"]
            df["comprend"] = None
            df["ne_comprend_pas"] = None
            df["description"] = None

        if self.rmes_path and self.rmes_path.exists():
            rmes_df = pd.read_csv(self.rmes_path, sep=";", encoding="utf-8")
            if "label_fr" in rmes_df.columns:
                rmes_df = rmes_df.rename(columns={
                    "contenu_central_fr": "comprend",
                    "note_exclusion_fr": "ne_comprend_pas",
                    "note_generale_fr": "description",
                })
                desc_cols = ["comprend", "ne_comprend_pas", "description"]
                rmes_descs = rmes_df[["code"] + desc_cols].drop_duplicates(subset="code")
                df = df.merge(rmes_descs, on="code", how="left", suffixes=("", "_rmes"))
                for col in desc_cols:
                    rmes_col = f"{col}_rmes"
                    if rmes_col in df.columns:
                        df[col] = df[col].fillna(df[rmes_col])
                        df = df.drop(columns=[rmes_col])

        return df

    def _extract_level4(self, code: str | None) -> str | None:
        """Extract level-4 prefix from COICOP code."""
        if not code or not isinstance(code, str):
            return None
        parts = code.strip().split(".")
        return ".".join(parts[:4]) if len(parts) >= 4 else ".".join(parts) + "." + "0"

    def _get_leaf_categories(self) -> pd.DataFrame:
        """Get only leaf-level COICOP categories."""
        df = self.coicop_df.copy()
        mask = df["code"].str.count(r"\.") == 4
        return df[mask].copy()

    def _get_categories_by_level(self, level: int) -> pd.DataFrame:
        """Get COICOP categories at a specific hierarchy level."""
        df = self.coicop_df.copy()
        mask = df["code"].str.count(r"\.") == (level - 1)
        return df[mask].copy()

    def _get_technical_leaf_nodes(self) -> pd.DataFrame:
        """Get leaf nodes among technical codes (98.x, 99.x)."""
        df = self.coicop_df.copy()
        technical = df[df["code"].str.startswith(("98", "99"))]
        tech_codes = set(technical["code"].values)
        is_leaf = technical["code"].apply(
            lambda c: not any(other.startswith(c + ".") for other in tech_codes)
        )
        return technical[is_leaf].copy()

    @staticmethod
    def _get_code_type(code: str) -> str:
        """Determine the type of COICOP code for prompt selection."""
        if code.startswith("99"):
            return "technical_99"
        if code.startswith("98"):
            return "technical_98"
        return "standard"

    # ── Annotations loading ──────────────────────────────────────────────

    def _load_annotations(self) -> dict[str, list[str]]:
        """Charge les 3 sources annotées et retourne {level4_code: [libelles]}.

        Sources :
          - tickets_application.csv : product → Code coicop
          - depenses_manuelles_carnets.csv : Nature de la dépense → coicop
          - ajouts_manuels_application.csv : product → code1
        """
        if self._annotations is not None:
            return self._annotations

        annotations_by_code: dict[str, list[str]] = {}

        # Default local paths if not specified
        annotations_map = {
            "tickets": "data/annotated/tickets_application.csv",
            "carnets": "data/annotated/depenses_manuelles_carnets.csv",
            "ajouts": "data/annotated/ajouts_manuels_application.csv",
        }

        # Override with S3 paths if provided
        if self.annotations_s3_paths:
            for key in annotations_map:
                if key in self.annotations_s3_paths:
                    annotations_map[key] = self.annotations_s3_paths[key]

        for source_name, file_path in annotations_map.items():
            try:
                logger.info("Loading annotations from %s...", file_path)
                df = _read_csv_from_path(file_path)
            except Exception as e:
                logger.warning("Could not load %s annotations (%s): %s",
                               source_name, file_path, e)
                continue

            if source_name == "tickets":
                text_col = "product"
                code_col = "Code coicop"
            elif source_name == "carnets":
                text_col = "Nature de la dépense"
                code_col = "coicop"
            elif source_name == "ajouts":
                text_col = "product"
                code_col = "code1"
            else:
                continue

            if text_col not in df.columns or code_col not in df.columns:
                logger.warning(
                    "Missing columns in %s: have %s, need %s/%s",
                    source_name, df.columns.tolist(), text_col, code_col,
                )
                continue

            for _, row in df.dropna(subset=[text_col, code_col]).iterrows():
                text = str(row[text_col]).strip()
                code = str(row[code_col]).strip()

                if not text or len(text) < 2:
                    continue

                l4 = self._extract_level4(code)
                if l4:
                    if l4 not in annotations_by_code:
                        annotations_by_code[l4] = []
                    if text not in annotations_by_code[l4]:
                        annotations_by_code[l4].append(text)

        self._annotations = annotations_by_code
        logger.info("Loaded annotations: %d categories, %d total examples",
                     len(annotations_by_code),
                     sum(len(v) for v in annotations_by_code.values()))
        return self._annotations

    # ── Style enforcement ────────────────────────────────────────────────

    def _style_enforce(self, raw_products: list[str]) -> list[str]:
        """Apply BDF ticket-style rules to LLM-generated products.

        Rules:
        1. unidecode + to UPPERCASE
        2. Units: 5g->GRS, 5kg->KG, 75cl->CL
        3. Remove non-alphanumeric punctuation
        4. Remove words < 2 characters
        5. Truncate to 40 characters max
        6. Remove stopwords (data/text/stopwords.json)
        7. Re-strip
        """
        # Load stopwords
        try:
            with open(STOPWORDS_PATH, "r", encoding="utf-8") as f:
                stopwords = set(json.load(f))
        except FileNotFoundError:
            stopwords = set()

        enforced = []
        for raw in raw_products:
            # 1. unidecode + uppercase
            text = unidecode.unidecode(raw).upper()

            # 2. Unit conversions: 5kg->5KG, 500g->500GRS, 75cl->75CL
            text = re.sub(r'\b(\d+)\s*(g|gr)\b', r'\1GRS', text)
            text = re.sub(r'\b(\d+)\s*(kg|KGS?)\b', r'\1KG', text)
            text = re.sub(r'\b(\d+)\s*(cl|CL)\b', r'\1CL', text)

            # 3. Remove non-alphanumeric characters (keep spaces and digits)
            text = re.sub(r'[^A-Z0-9\s]', ' ', text)

            # 4. Remove single-character words (except X which indicates quantity)
            tokens = text.split()
            tokens = [t for t in tokens if len(t) > 1 or t == 'X']
            text = ' '.join(tokens)

            # 6. Remove stopwords
            tokens = text.split()
            tokens = [t for t in tokens if t not in stopwords]
            text = ' '.join(tokens)

            # 5. Truncate to 40 characters
            text = text[:40]

            # 7. Clean up extra whitespace
            text = re.sub(r'\s+', ' ', text).strip()

            if len(text) >= 3:  # Keep only meaningful products
                enforced.append(text)

        return enforced

    # ── Prompt building ──────────────────────────────────────────────────

    def _build_generation_prompt(self, code_type: str = "standard") -> PromptTemplate:
        """Build the prompt template for synthetic data generation.

        Each template now includes:
        - Style constraints (uppercase, BDF units, no punctuation)
        - Few-shot section (injected dynamically from real annotations)
        """
        templates = {
            "standard": """Tu es un expert en classification des produits et services selon la nomenclature COICOP.

Genere {num_examples} exemples reels de produits ou services pour la categorie COICOP suivante:

Code COICOP: {code}
Libelle: {libelle}
{comprend_section}
{ne_comprend_section}

=== CONTRAINTES DE STYLE (OBLIGATOIRES) ===
- TOUT EN MAJUSCULES (comme sur un ticket de caisse)
- PAS de ponctuation (ni points, virgules, tirets)
- Poids/quantite abbrevies : GRS, KG, CL (ex: 500GRS, 2KG, 75CL)
- Format quantite produit : X12, X6, X1 (ex: OEUFS FRAIS X12)
- Longueur : 10 a 40 caracteres maximum
- Varier : parfois avec marque, parfois sans

=== EXEMPLES REELS DU MEME CODE (FEW-SHOT) ===
{fewshot_examples}

Genere maintenant des produits un par ligne :""",

            "technical_98": """Tu es un expert en classification des depenses des menages pour l'enquete Budget de Famille de l'INSEE.

Genere {num_examples} exemples reels de descriptifs de depenses pour la categorie technique suivante:

Code: {code}
Libelle: {libelle}
{comprend_section}
{ne_comprend_section}

=== CONTRAINTES DE STYLE (OBLIGATOIRES) ===
- TOUT EN MAJUSCULES
- PAS de ponctuation
- Formulations types releves bancaires : CARTE, VIR, PRELEVEMENT, RETRAIT
- Format quantite : X12, X6, X1
- Longueur : 10 a 40 caracteres maximum

=== EXEMPLES REELS DU MEME CODE (FEW-SHOT) ===
{fewshot_examples}

Genere maintenant des descriptifs un par ligne :""",

            "technical_99": """Tu es un expert en classification des depenses des menages pour l'enquete Budget de Famille de l'INSEE.

Genere {num_examples} exemples reels de descriptions de transactions pour la categorie hors champ COICOP suivante:

Code: {code}
Libelle: {libelle}
{comprend_section}
{ne_comprend_section}

=== CONTRAINTES DE STYLE (OBLIGATOIRES) ===
- TOUT EN MAJUSCULES
- PAS de ponctuation
- Formulations types operations bancaires/administratives
- Format quantite : X12, X6, X1
- Longueur : 10 a 40 caracteres maximum

=== EXEMPLES REELS DU MEME CODE (FEW-SHOT) ===
{fewshot_examples}

Genere maintenant des descriptions un par ligne :""",
        }

        template = templates.get(code_type, templates["standard"])
        return PromptTemplate(
            input_variables=[
                "num_examples", "code", "libelle",
                "comprend_section", "ne_comprend_section",
                "fewshot_examples",
            ],
            template=template,
        )

    # ── Core generation ──────────────────────────────────────────────────

    def _get_fewshot_examples(self, code_l4: str) -> str:
        """Get few-shot examples for a given level-4 code.

        Returns formatted string of real product labels for the prompt.
        """
        annotations = self._load_annotations()
        if code_l4 in annotations:
            candidates = annotations[code_l4]
            if len(candidates) > self.fewshot_per_category:
                import random
                import numpy as np
                rng = np.random.RandomState(42)
                sample = rng.choice(candidates, self.fewshot_per_category, replace=False)
            else:
                sample = candidates
            return "\n".join(f"- {ex}" for ex in sample)
        return "(pas d'exemples reels disponibles pour ce code)"

    def generate_for_category(
        self,
        code: str,
        libelle: str,
        comprend: str | None = None,
        ne_comprend_pas: str | None = None,
    ) -> list[dict[str, str]]:
        """Generate synthetic examples for a single COICOP category.

        Makes multiple LLM calls (self.calls_per_category) with varied prompts
        and accumulates/de-duplicates results.

        Args:
            code: COICOP code.
            libelle: COICOP category label.
            comprend: INSEE description of what the category includes (optional).
            ne_comprend_pas: INSEE description of what the category excludes (optional).

        Returns:
            List of dicts with 'product', 'code', 'libelle' keys.
        """
        code_type = self._get_code_type(code)
        prompt = self._build_generation_prompt(code_type)
        code_l4 = self._extract_level4(code)

        # Get few-shot real examples
        fewshot = self._get_fewshot_examples(code_l4)
        if not fewshot or "(pas d'exemples" in fewshot:
            fewshot = "(pas d'exemples reels disponibles pour ce code)"

        # Build optional context sections
        comprend_section = f"Cette categorie comprend: {comprend}" if comprend else ""
        ne_comprend_section = f"Cette categorie NE comprend PAS: {ne_comprend_pas}" if ne_comprend_pas else ""

        # Multiple calls with variations
        all_products: list[str] = []
        import random
        import numpy as np
        rng = np.random.RandomState(42)

        for call_idx in range(self.calls_per_category):
            # Vary the prompt slightly each call: shuffle includes/excludes order, sample few-shot
            if call_idx % 2 == 0:
                includes_section = comprend_section
                excludes_section = ne_comprend_section
            else:
                includes_section = ne_comprend_section
                excludes_section = comprend_section

            # Sample different few-shot examples for variety
            annotations = self._load_annotations()
            if code_l4 in annotations:
                candidates = annotations[code_l4]
                n_samples = min(self.fewshot_per_category, len(candidates))
                if n_samples > 0:
                    sample = list(rng.choice(candidates, n_samples, replace=False))
                    fewshot_current = "\n".join(f"- {ex}" for ex in sample)
                else:
                    fewshot_current = "(pas d'exemples reels pour ce code)"
            else:
                fewshot_current = "(pas d'exemples reels pour ce code)"

            formatted_prompt = prompt.format(
                num_examples=self.examples_per_call,
                code=code,
                libelle=libelle,
                comprend_section=includes_section,
                ne_comprend_section=excludes_section,
                fewshot_examples=fewshot_current,
            )

            try:
                response = self.llm.invoke(formatted_prompt)
                response_text = (
                    response.content if hasattr(response, "content") else str(response)
                )
                products = self._parse_response(response_text)
                all_products.extend(products)
                logger.info("  Call %d/%d: %d raw products extracted",
                            call_idx + 1, self.calls_per_category, len(products))
            except Exception as e:
                logger.warning("  Call %d/%d failed for %s: %s",
                               call_idx + 1, self.calls_per_category, code, e)

        # De-duplicate
        all_products = list(dict.fromkeys(all_products))  # preserve order

        # Apply style enforcement
        if self.style_enforce:
            all_products = self._style_enforce(all_products)

        # Final de-dup after style
        all_products = list(dict.fromkeys(all_products))

        return [
            {"product": product, "code": code, "libelle": libelle}
            for product in all_products
        ]

    def _parse_response(self, response: str) -> list[str]:
        """Parse LLM response into list of product names."""
        lines = response.strip().split("\n")
        products = []

        skip_patterns = [
            "voici", "voila", "exemples", "categorie",
            "coicop", "produits pour", "produits de",
            "ci-dessous", "suivants", "voici liste",
        ]

        for line in lines:
            line = line.strip()
            if not line:
                continue

            line_lower = line.lower()
            if any(pat in line_lower for pat in skip_patterns):
                continue

            if len(line) > 80:
                continue

            # Remove list markers
            for prefix in ["- ", "• ", "* ", "– ", "\u2022 "]:
                if line.startswith(prefix):
                    line = line[len(prefix):]
                    break

            # Remove numbered prefixes
            if len(line) > 2 and line[0].isdigit() and line[1] in [".", ")", ":"]:
                line = line[2:].strip()
            elif (len(line) > 3 and line[0].isdigit()
                  and line[1].isdigit()
                  and line[2] in [".", ")", ":"]):
                line = line[3:].strip()

            if line:
                products.append(line)

        return products

    # ── Style evaluation ─────────────────────────────────────────────────

    def _evaluate_style(self, products: list[str]) -> None:
        """Console report comparing generated style vs BDF tickets.

        Metrics: uppercase rate, length distribution, token diversity.
        """
        # Real BDF tickets baseline
        try:
            tickets = pd.read_csv("data/annotated/tickets_application.csv", sep=";")
            real_products = tickets["product"].dropna().tolist()
        except Exception:
            real_products = []

        print("\n" + "=" * 60)
        print("  BAZAR")
        print("  Style evaluation report")
        print("=" * 60)

        # Uppercase rate
        synth_upper = sum(1 for p in products if p.isupper()) / len(products) if products else 0
        if real_products:
            real_upper = sum(1 for p in real_products if p.isupper()) / len(real_products)
        else:
            real_upper = 0

        print(f"\n{'Metr.':<35} {'Reel':>10} {'Synth':>10} {'Ecarts':>10}")
        print("-" * 60)
        print(f"{'100% MAJUSCULES':.<35} {real_upper:>9.1%} {synth_upper:>9.1%} {abs(real_upper - synth_upper):>9.1%}")

        # Length distribution
        synth_lengths = [len(p) for p in products]
        real_lengths = [len(p) for p in real_products]

        if synth_lengths:
            synth_mean, synth_std = np.mean(synth_lengths), np.std(synth_lengths)
            print(f"{'Long. (moy ± std)':.<35} {len(real_lengths) if real_lengths else 0:>8} {synth_mean:>6.1f} ± {synth_std:>5.1f}")
            print(f"{'Long. (min/max)':.<35} {min(real_lengths) if real_lengths else 0:>8} {min(synth_lengths)} / {max(synth_lengths)}")

        if real_lengths:
            print(f"{'Long. reel (moy ± std)':.<35} {np.mean(real_lengths):>6.1f} ± {np.std(real_lengths):>5.1f} {min(real_lengths)} / {max(real_lengths)}")

        # Token diversity
        synth_tokens = set()
        for p in products:
            synth_tokens.update(p.split())
        real_tokens = set()
        for p in (real_products or []):
            real_tokens.update(p.split())

        synth_unique = len(synth_tokens)
        real_unique = len(real_tokens)

        print(f"{'Tokens uniques synth':.<35} {synth_unique:>8}")
        if real_products:
            print(f"{'Tokens uniques reel':.<35} {real_unique:>8}")

        print("=" * 60)

    # ── Dataset generation ───────────────────────────────────────────────

    def generate_dataset(
        self,
        level: int = 4,
        max_categories: int | None = None,
        exclude_technical: bool = False,
        output_path: str | Path | None = None,
    ) -> pd.DataFrame:
        """Generate a synthetic dataset with multi-call and style enforcement.

        Args:
            level: COICOP hierarchy level to generate for (1-5).
            max_categories: Maximum categories to process (for testing).
            exclude_technical: Whether to exclude 98.x and 99.x technical codes.
            output_path: If provided, save incrementally (CSV only).

        Returns:
            DataFrame with 'product', 'code', 'libelle' columns.
        """
        # Get standard COICOP codes at requested level
        categories = self._get_categories_by_level(level)
        categories = categories[~categories["code"].str.startswith(("98", "99"))]

        if not exclude_technical:
            tech_leaves = self._get_technical_leaf_nodes()
            categories = pd.concat([categories, tech_leaves], ignore_index=True)

        if max_categories is not None:
            categories = categories.head(max_categories)

        total = len(categories)
        all_examples: list[dict[str, str]] = []

        incremental_path = output_path if output_path and str(output_path).endswith(".csv") else None

        if incremental_path:
            output_path = Path(incremental_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("product;code;libelle\n")

        for i, (_, row) in enumerate(categories.iterrows(), 1):
            code = row["code"]
            libelle = row["libelle"]
            comprend = row.get("comprend") if pd.notna(row.get("comprend")) else None
            ne_comprend_pas = row.get("ne_comprend_pas") if pd.notna(row.get("ne_comprend_pas")) else None

            logger.info("[%d/%d] Generating examples for %s: %s",
                        i, total, code, libelle[:40])

            try:
                examples = self.generate_for_category(
                    code, libelle,
                    comprend=comprend, ne_comprend_pas=ne_comprend_pas,
                )
                all_examples.extend(examples)
                logger.info("  → %d examples for %s", len(examples), code)

                # Append to file incrementally
                if incremental_path is not None:
                    batch_df = pd.DataFrame(examples)
                    batch_df.to_csv(
                        incremental_path, mode="a", header=False,
                        index=False, sep=";", encoding="utf-8 ",
                    )
            except Exception as e:
                logger.warning("  Failed to generate for %s: %s", code, e)

        return pd.DataFrame(all_examples)


# ── CLI helpers ──────────────────────────────────────────────────────────────

def _parse_s3_paths(paths: list[str] | None) -> dict[str, str]:
    """Parse --annotations comma-separated paths into {source: path} mapping."""
    if not paths:
        return {}
    mapping = {}
    if len(paths) >= 1:
        mapping["tickets"] = paths[0]
    if len(paths) >= 2:
        mapping["carnets"] = paths[1]
    if len(paths) >= 3:
        mapping["ajouts"] = paths[2]
    return mapping


def generate_and_save(
    output_path: str | Path,
    coicop_path: str | Path = "data/coicop_et_codes_techniques.csv",
    rmes_path: str | Path | None = "data/coicop-2018_envoi_rmes_20251022.csv",
    annotations_paths: list[str] | None = None,
    examples_per_category: int = 10,
    level: int = 4,
    max_categories: int | None = None,
    exclude_technical: bool = False,
    calls_per_category: int = 5,
    fewshot_per_category: int = 6,
    style_enforce: bool = True,
    evaluate_style: bool = False,
) -> pd.DataFrame:
    """Generate synthetic data and save to file.

    Environment variables:
        LLMLAB_API_KEY: API key (required)
        LLMLAB_URL: Base URL for API (optional)
        OPENAI_MODEL: Model name (optional, defaults to gpt-oss:20b)

    Args:
        output_path: Path to save generated data (local or S3, parquet or csv).
        coicop_path: Path to COICOP definitions (with 98/99 codes).
        rmes_path: Path to RMES file for enriched descriptions (optional).
        annotations_paths: List of CSV paths (tickets, carnets, ajouts) — local or S3.
        examples_per_category: Number of examples per LLM call.
        level: COICOP hierarchy level.
        max_categories: Maximum categories to process (for testing).
        exclude_technical: Whether to exclude 98.x and 99.x technical codes.
        calls_per_category: LLM calls per category (5 = ~50 products/cat).
        fewshot_per_category: Real examples injected per category in prompt.
        style_enforce: Apply BDF ticket style post-processing.
        evaluate_style: Compare generated style vs real BDF tickets.

    Returns:
        Generated DataFrame.
    """
    annotations_map = _parse_s3_paths(annotations_paths)

    generator = COICOPSyntheticGenerator(
        coicop_path=coicop_path,
        rmes_path=rmes_path,
        annotations_s3_paths=annotations_map,
        examples_per_call=examples_per_category,
        calls_per_category=calls_per_category,
        fewshot_per_category=fewshot_per_category,
        style_enforce=style_enforce,
    )

    output_path = Path(output_path) if not str(output_path).startswith("s3://") else str(output_path)

    # For CSV, use incremental saving; for parquet, save at the end
    incremental_path = output_path if str(output_path).endswith(".csv") else None

    logger.info("Generating synthetic COICOP data:")
    logger.info("  Output: %s", output_path)
    logger.info("  Level: %d", level)
    logger.info("  Calls/category: %d × %d examples = ~%d products",
                calls_per_category, examples_per_category,
                calls_per_category * examples_per_category)
    logger.info("  Few-shot examples: %d per category", fewshot_per_category)
    logger.info("  Style enforce: %s", style_enforce)

    df = generator.generate_dataset(
        level=level,
        max_categories=max_categories,
        exclude_technical=exclude_technical,
        output_path=incremental_path,
    )

    if isinstance(output_path, str) and output_path.startswith("s3://"):
        _write_parquet_to_path(df, output_path)
        logger.info("Saved %d examples to %s", len(df), output_path)
    elif isinstance(output_path, Path) and output_path.suffix == ".parquet":
        df.to_parquet(output_path, index=False)
        logger.info("Saved %d examples to %s", len(df), output_path)
    else:
        logger.info("Done. %d examples total.", len(df))

    # Style evaluation
    if evaluate_style and "product" in df.columns:
        generator._evaluate_style(df["product"].tolist())

    return df


# ── Main CLI ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Generate synthetic COICOP training data (refactored per Code2Text report)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="data/synthetic_coicop_v2.parquet",
        help="Output path (local or s3://...). Detected format from extension.",
    )
    parser.add_argument(
        "--coicop",
        type=str,
        default="data/coicop_et_codes_techniques.csv",
        help="Path to COICOP definitions CSV (with 98/99 codes)",
    )
    parser.add_argument(
        "--rmes",
        type=str,
        default=None,
        help="Path to RMES file for enriched descriptions (set empty to skip)",
    )
    parser.add_argument(
        "--annotations",
        nargs=3,
        default=None,
        metavar="TICKETS",
        help="3 CSV paths (tickets, carnets, ajouts) — local or s3:// URIs",
    )
    parser.add_argument(
        "--examples", "-n",
        type=int,
        default=10,
        help="Number of examples per LLM call per category",
    )
    parser.add_argument(
        "--calls", "-c",
        type=int,
        default=5,
        help="LLM calls per category (default: 5, → ~50 products/cat)",
    )
    parser.add_argument(
        "--fewshot", "-f",
        type=int,
        default=6,
        help="Real examples injected per category in prompt (default: 6)",
    )
    parser.add_argument(
        "--level", "-l",
        type=int,
        default=4,
        help="COICOP hierarchy level (1-5, default: 4)",
    )
    parser.add_argument(
        "--max-categories", "-m",
        type=int,
        default=None,
        help="Maximum categories to process (for testing)",
    )
    parser.add_argument(
        "--exclude-technical",
        action="store_true",
        default=False,
        help="Exclude 98.x and 99.x technical codes",
    )
    parser.add_argument(
        "--no-style",
        action="store_true",
        default=False,
        help="Disable style post-processing (default: enabled)",
    )
    parser.add_argument(
        "--evaluate-style",
        action="store_true",
        default=False,
        help="Compare generated style vs real BDF tickets (console report)",
    )

    args = parser.parse_args()

    # Convert rmes to None if empty string
    rmes = args.rmes or None

    df = generate_and_save(
        output_path=args.output,
        coicop_path=args.coicop,
        rmes_path=rmes,
        annotations_paths=args.annotations,
        examples_per_category=args.examples,
        level=args.level,
        max_categories=args.max_categories,
        exclude_technical=args.exclude_technical,
        calls_per_category=args.calls,
        fewshot_per_category=args.fewshot,
        style_enforce=not args.no_style,
        evaluate_style=args.evaluate_style,
    )
