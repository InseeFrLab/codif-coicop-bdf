"""Synthetic data generation for COICOP classification using LangChain."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_experimental.synthetic_data import create_data_generation_chain
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)


class COICOPExample(BaseModel):
    """Schema for a synthetic COICOP product example."""

    product: str = Field(
        description="Product or service name/description in French"
    )
    code: str = Field(description="COICOP code (e.g., '01.1.1.1.1')")
    libelle: str = Field(description="COICOP category label in French")


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
    model_name = os.environ.get("OPENAI_MODEL", "gpt-oss:20b")

    logger.info(f"Configuring llm: {base_url} with model : {model_name}")


    return ChatOpenAI(
        api_key=api_key,
        base_url=base_url,
        model=model_name,
        temperature=0.8,
    )


class COICOPSyntheticGenerator:
    """Generator for synthetic COICOP classification training data.

    Uses LangChain's synthetic data generation capabilities to create
    realistic product descriptions for each COICOP category.
    """

    def __init__(
        self,
        llm: BaseChatModel | None = None,
        coicop_path: str | Path = "data/coicop_et_codes_techniques.csv",
        rmes_path: str | Path | None = "data/coicop-2018_envoi_rmes_20251022.csv",
        examples_per_category: int = 10,
    ) -> None:
        """Initialize the synthetic data generator.

        Args:
            llm: LangChain chat model (defaults to OpenAI from env vars)
            coicop_path: Path to COICOP definitions CSV (with 98/99 codes)
            rmes_path: Path to RMES COICOP file for enriched descriptions (optional)
            examples_per_category: Number of examples to generate per category
        """
        self.llm = llm if llm is not None else get_llm_from_env()
        self.coicop_path = Path(coicop_path)
        self.rmes_path = Path(rmes_path) if rmes_path else None
        self.examples_per_category = examples_per_category
        self._coicop_df: pd.DataFrame | None = None

    @property
    def coicop_df(self) -> pd.DataFrame:
        """Lazy load the COICOP hierarchy."""
        if self._coicop_df is None:
            self._coicop_df = self._load_coicop()
        return self._coicop_df

    def _load_coicop(self) -> pd.DataFrame:
        """Load COICOP hierarchy from CSV, optionally enriched with RMES descriptions.

        Supports multiple formats:
        - Old format: columns (libelle, code) or (Libelle, Code)
        - Enriched format: columns (libelle, code, url, description, comprend, ne_comprend_pas)
        - RMES format: columns (tri, type, parent, code, label_en, label_fr, note_generale_*,
          contenu_central_*, contenu_additionnel_*, note_exclusion_*)

        If rmes_path is set, descriptions from the RMES file are merged in.
        """
        df = pd.read_csv(self.coicop_path, sep=";", encoding="utf-8")

        # Detect and normalize different formats
        if "label_fr" in df.columns:
            # RMES format (coicop-2018_envoi_rmes_*.csv)
            df = df.rename(columns={
                "label_fr": "libelle",
                "contenu_central_fr": "comprend",
                "note_exclusion_fr": "ne_comprend_pas",
                "note_generale_fr": "description",
            })
        elif "comprend" not in df.columns:
            # Old simple format (libelle, code only)
            if "Libelle" in df.columns:
                df = df.rename(columns={"Libelle": "libelle", "Code": "code"})
            else:
                df.columns = ["libelle", "code"]
            df["comprend"] = None
            df["ne_comprend_pas"] = None
            df["description"] = None

        # Enrich with RMES descriptions if available
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

    def _get_leaf_categories(self) -> pd.DataFrame:
        """Get only leaf-level COICOP categories (most specific).

        Leaf categories have 5-level codes (e.g., '01.1.1.1.1').
        """
        df = self.coicop_df.copy()
        # Filter for codes with 5 parts (leaf level)
        mask = df["code"].str.count(r"\.") == 4
        return df[mask].copy()

    def _get_categories_by_level(self, level: int) -> pd.DataFrame:
        """Get COICOP categories at a specific hierarchy level.

        Args:
            level: Hierarchy level (1-5). Level 1 = '01', Level 5 = '01.1.1.1.1'

        Returns:
            DataFrame with categories at the specified level
        """
        df = self.coicop_df.copy()
        # Level 1 has 0 dots, level 5 has 4 dots
        mask = df["code"].str.count(r"\.") == (level - 1)
        return df[mask].copy()

    def _get_technical_leaf_nodes(self) -> pd.DataFrame:
        """Get leaf nodes among technical codes (98.x, 99.x).

        Technical codes have irregular hierarchy depths, so leaf detection
        is based on whether any other code starts with this code + '.'.
        """
        df = self.coicop_df.copy()
        technical = df[df["code"].str.startswith(("98", "99"))]
        tech_codes = set(technical["code"].values)
        is_leaf = technical["code"].apply(
            lambda c: not any(other.startswith(c + ".") for other in tech_codes)
        )
        return technical[is_leaf].copy()

    @staticmethod
    def _get_code_type(code: str) -> str:
        """Determine the type of COICOP code for prompt selection.

        Returns:
            'technical_98', 'technical_99', or 'standard'
        """
        if code.startswith("99"):
            return "technical_99"
        if code.startswith("98"):
            return "technical_98"
        return "standard"

    def _build_generation_prompt(self, code_type: str = "standard") -> PromptTemplate:
        """Build the prompt template for synthetic data generation.

        Args:
            code_type: One of 'standard', 'technical_98', 'technical_99'

        The prompt includes optional sections for "comprend" (what the category includes)
        and "ne_comprend_pas" (what it excludes) from INSEE COICOP descriptions.
        """
        templates = {
            "standard": """Tu es un expert en classification des produits et services selon la nomenclature COICOP.

Génère {num_examples} exemples réalistes de produits ou services pour la catégorie COICOP suivante:

Code COICOP: {code}
Libellé: {libelle}
{comprend_section}
{ne_comprend_section}

INSTRUCTIONS:
- Génère des noms de produits comme on les trouve sur un ticket de caisse ou relevé bancaire
- UTILISE DES MARQUES RÉELLES connues en France (exemples: Nestlé, Nutella, Danone, Carrefour, Président, Lu, Panzani, Evian, Coca-Cola, etc.)
- Varie les formulations: nom de marque seul, marque + produit, produit générique
- Courts (1 à 5 mots généralement)
- En français

Exemples de produits (un par ligne):""",
            "technical_98": """Tu es un expert en classification des dépenses des ménages pour l'enquête Budget de Famille de l'INSEE.

Génère {num_examples} exemples réalistes de descriptions de dépenses pour la catégorie technique suivante:

Code: {code}
Libellé: {libelle}
{comprend_section}
{ne_comprend_section}

INSTRUCTIONS:
- Génère des descriptions comme on les trouve sur un relevé bancaire, un ticket de caisse, ou un carnet de dépenses de ménage
- Formulations courtes et informelles (1 à 6 mots)
- En français
- Varie les formulations: abréviations bancaires, noms d'enseignes, descriptions informelles
- Exemples de style: "CB CARREFOUR", "COURSES LIDL", "PAIEMENT CB 15/03", "courses du samedi", "supermarché"

Exemples de descriptions (une par ligne):""",
            "technical_99": """Tu es un expert en classification des dépenses des ménages pour l'enquête Budget de Famille de l'INSEE.

Génère {num_examples} exemples réalistes de descriptions de transactions pour la catégorie hors champ COICOP suivante:

Code: {code}
Libellé: {libelle}
{comprend_section}
{ne_comprend_section}

INSTRUCTIONS:
- Génère des descriptions comme on les trouve sur un relevé bancaire, un avis d'imposition, ou un carnet de dépenses de ménage
- Formulations typiques des opérations bancaires et administratives (1 à 8 mots)
- En français
- Varie les formulations: libellés bancaires officiels, descriptions informelles du ménage
- Exemples de style: "VIR SEPA EMIS", "PRELEVEMENT IMPOTS", "RETRAIT DAB", "DON CROIX ROUGE", "cadeau anniversaire", "taxe foncière"

Exemples de descriptions (une par ligne):""",
        }

        template = templates.get(code_type, templates["standard"])
        return PromptTemplate(
            input_variables=["num_examples", "code", "libelle", "comprend_section", "ne_comprend_section"],
            template=template,
        )

    def _create_few_shot_prompt(
        self,
        examples: list[dict[str, str]],
    ) -> FewShotPromptTemplate:
        """Create a few-shot prompt with examples.

        Args:
            examples: List of example dictionaries with 'product', 'code', 'libelle'

        Returns:
            Configured FewShotPromptTemplate
        """
        example_template = PromptTemplate(
            input_variables=["product", "code", "libelle"],
            template="Produit: {product}\nCode: {code}\nCatégorie: {libelle}",
        )

        prefix = """Tu es un expert en classification COICOP. Voici quelques exemples de produits avec leurs codes COICOP:

"""
        suffix = """
Maintenant, génère {num_examples} nouveaux exemples de produits pour la catégorie suivante:
Code: {code}
Libellé: {libelle}

Produits (un par ligne):"""

        return FewShotPromptTemplate(
            examples=examples,
            example_prompt=example_template,
            prefix=prefix,
            suffix=suffix,
            input_variables=["num_examples", "code", "libelle"],
            example_separator="\n\n",
        )

    def generate_for_category(
        self,
        code: str,
        libelle: str,
        comprend: str | None = None,
        ne_comprend_pas: str | None = None,
        num_examples: int | None = None,
    ) -> list[dict[str, str]]:
        """Generate synthetic examples for a single COICOP category.

        Args:
            code: COICOP code
            libelle: COICOP category label
            comprend: INSEE description of what the category includes (optional)
            ne_comprend_pas: INSEE description of what the category excludes (optional)
            num_examples: Number of examples (defaults to examples_per_category)

        Returns:
            List of dictionaries with 'product', 'code', 'libelle' keys
        """
        if num_examples is None:
            num_examples = self.examples_per_category

        code_type = self._get_code_type(code)
        prompt = self._build_generation_prompt(code_type)

        # Build optional context sections from INSEE descriptions
        comprend_section = f"Cette catégorie comprend: {comprend}" if comprend else ""
        ne_comprend_section = f"Cette catégorie NE comprend PAS: {ne_comprend_pas}" if ne_comprend_pas else ""

        # Format the prompt
        formatted_prompt = prompt.format(
            num_examples=num_examples,
            code=code,
            libelle=libelle,
            comprend_section=comprend_section,
            ne_comprend_section=ne_comprend_section,
        )

        # Generate using LLM
        response = self.llm.invoke(formatted_prompt)
        response_text = (
            response.content if hasattr(response, "content") else str(response)
        )

        # Parse the response into individual products
        products = self._parse_response(response_text)

        return [
            {"product": product, "code": code, "libelle": libelle}
            for product in products[:num_examples]
        ]

    def _parse_response(self, response: str) -> list[str]:
        """Parse LLM response into list of product names.

        Args:
            response: Raw LLM response text

        Returns:
            List of product name strings
        """
        lines = response.strip().split("\n")
        products = []

        # Patterns that indicate introductory/explanatory lines to skip
        skip_patterns = [
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

        for line in lines:
            # Clean up the line
            line = line.strip()
            if not line:
                continue

            # Skip introductory lines (check before removing prefixes)
            line_lower = line.lower()
            if any(pattern in line_lower for pattern in skip_patterns):
                continue

            # Skip lines that are too long (likely explanatory text, not product names)
            if len(line) > 80:
                continue

            # Remove common prefixes like "1.", "- ", "• ", etc.
            for prefix in ["- ", "• ", "* ", "– "]:
                if line.startswith(prefix):
                    line = line[len(prefix) :]
                    break

            # Remove numbered prefixes like "1. ", "2. "
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

    def generate_dataset(
        self,
        level: int = 4,
        max_categories: int | None = None,
        exclude_technical: bool = False,
        output_path: str | Path | None = None,
    ) -> pd.DataFrame:
        """Generate a complete synthetic dataset.

        Args:
            level: COICOP hierarchy level to generate for (1-5)
            max_categories: Maximum number of categories to process (for testing)
            exclude_technical: Whether to exclude 98.x and 99.x technical codes
            output_path: If provided, save incrementally after each category (CSV only)

        Returns:
            DataFrame with 'product', 'code', 'libelle' columns
        """
        # Get standard COICOP codes at the requested level
        categories = self._get_categories_by_level(level)
        # Exclude technical codes from the level-based selection (they are added separately)
        categories = categories[~categories["code"].str.startswith(("98", "99"))]

        if not exclude_technical:
            # Add 98/99 leaf nodes (irregular depths, not filtered by level)
            tech_leaves = self._get_technical_leaf_nodes()
            categories = pd.concat([categories, tech_leaves], ignore_index=True)

        if max_categories is not None:
            categories = categories.head(max_categories)

        total = len(categories)
        all_examples: list[dict[str, str]] = []

        # Write CSV header if incremental saving is enabled
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("product;code;libelle\n")

        for i, (_, row) in enumerate(categories.iterrows(), 1):
            code = row["code"]
            libelle = row["libelle"]
            # Extract enriched INSEE descriptions if available
            comprend = row.get("comprend") if pd.notna(row.get("comprend")) else None
            ne_comprend_pas = row.get("ne_comprend_pas") if pd.notna(row.get("ne_comprend_pas")) else None

            logger.info(f"[{i}/{total}] Generating examples for {code}: {libelle}")

            try:
                examples = self.generate_for_category(
                    code, libelle, comprend=comprend, ne_comprend_pas=ne_comprend_pas
                )
                all_examples.extend(examples)
                logger.info(f"  Generated {len(examples)} examples")

                # Append to file incrementally
                if output_path is not None:
                    batch_df = pd.DataFrame(examples)
                    batch_df.to_csv(
                        output_path, mode="a", header=False,
                        index=False, sep=";", encoding="utf-8",
                    )
                    logger.info(f"  Saved to {output_path} (total: {len(all_examples)} examples)")
            except Exception as e:
                logger.warning(f"  Failed to generate for {code}: {e}")

        return pd.DataFrame(all_examples)

    def generate_with_dataset_generator(
        self,
        level: int = 4,
        max_categories: int | None = None,
        exclude_technical: bool = False,
    ) -> pd.DataFrame:
        """Generate dataset using LangChain's DatasetGenerator.

        This is an alternative method using the experimental DatasetGenerator
        for more structured output.

        Args:
            level: COICOP hierarchy level to generate for (1-5)
            max_categories: Maximum number of categories to process
            exclude_technical: Whether to exclude 98.x and 99.x technical codes

        Returns:
            DataFrame with synthetic examples
        """
        categories = self._get_categories_by_level(level)
        categories = categories[~categories["code"].str.startswith(("98", "99"))]

        if not exclude_technical:
            tech_leaves = self._get_technical_leaf_nodes()
            categories = pd.concat([categories, tech_leaves], ignore_index=True)

        if max_categories is not None:
            categories = categories.head(max_categories)

        # Create the data generation chain
        chain = create_data_generation_chain(self.llm, COICOPExample)

        all_examples: list[dict[str, str]] = []

        for _, row in categories.iterrows():
            code = row["code"]
            libelle = row["libelle"]

            logger.info(f"Generating structured examples for {code}: {libelle}")

            # Build subject description for the generator
            subject = f"produit ou service de la catégorie COICOP '{libelle}' (code {code})"

            try:
                for _ in range(self.examples_per_category):
                    result = chain.invoke({"subject": subject})
                    if isinstance(result, dict) and "text" in result:
                        # Extract the generated text
                        all_examples.append(
                            {
                                "product": result.get("text", ""),
                                "code": code,
                                "libelle": libelle,
                            }
                        )
            except Exception as e:
                logger.warning(f"  Failed to generate for {code}: {e}")

        return pd.DataFrame(all_examples)


def generate_and_save(
    output_path: str | Path,
    coicop_path: str | Path = "data/coicop_et_codes_techniques.csv",
    rmes_path: str | Path | None = "data/coicop-2018_envoi_rmes_20251022.csv",
    examples_per_category: int = 10,
    level: int = 4,
    max_categories: int | None = None,
    exclude_technical: bool = False,
) -> pd.DataFrame:
    """Generate synthetic data and save to file.

    Uses environment variables for LLM configuration:
        LLMLAB_API_KEY: API key (required)
        LLMLAB_URL: Base URL for API (optional)
        OPENAI_MODEL: Model name (optional, defaults to gpt-oss:20b)

    Args:
        output_path: Path to save the generated data (parquet or csv)
        coicop_path: Path to COICOP definitions (with 98/99 codes)
        rmes_path: Path to RMES file for enriched descriptions (optional)
        examples_per_category: Number of examples per category
        level: COICOP hierarchy level
        max_categories: Maximum categories to process (for testing)
        exclude_technical: Whether to exclude 98.x and 99.x technical codes

    Returns:
        Generated DataFrame
    """
    generator = COICOPSyntheticGenerator(
        coicop_path=coicop_path,
        rmes_path=rmes_path,
        examples_per_category=examples_per_category,
    )

    output_path = Path(output_path)

    # For CSV, use incremental saving (writes after each category)
    # For parquet, save at the end (parquet doesn't support appending)
    incremental_path = output_path if output_path.suffix != ".parquet" else None

    df = generator.generate_dataset(
        level=level,
        max_categories=max_categories,
        exclude_technical=exclude_technical,
        output_path=incremental_path,
    )

    if output_path.suffix == ".parquet":
        df.to_parquet(output_path, index=False)
        logger.info(f"Saved {len(df)} examples to {output_path}")
    else:
        logger.info(f"Done. {len(df)} examples saved to {output_path}")

    return df


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Generate synthetic COICOP training data"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="data/synthetic_coicop.parquet",
        help="Output file path",
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
        default="data/coicop-2018_envoi_rmes_20251022.csv",
        help="Path to RMES file for enriched descriptions (set to empty to skip)",
    )
    parser.add_argument(
        "--examples",
        "-n",
        type=int,
        default=10,
        help="Number of examples per category",
    )
    parser.add_argument(
        "--level",
        "-l",
        type=int,
        default=4,
        help="COICOP hierarchy level (1-5)",
    )
    parser.add_argument(
        "--max-categories",
        "-m",
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

    args = parser.parse_args()

    generate_and_save(
        output_path=args.output,
        coicop_path=args.coicop,
        rmes_path=args.rmes or None,
        examples_per_category=args.examples,
        level=args.level,
        max_categories=args.max_categories,
        exclude_technical=args.exclude_technical,
    )
