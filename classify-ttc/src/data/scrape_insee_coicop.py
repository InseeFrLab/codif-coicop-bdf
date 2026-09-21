"""Scrape COICOP descriptions from INSEE website."""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# INSEE URL patterns for COICOP 2018
BASE_URL = "https://www.insee.fr/fr/metadonnees/coicop2018"

# Level name mapping based on number of dots in code
LEVEL_NAMES = {
    0: "division",      # 01
    1: "groupe",        # 01.1
    2: "classe",        # 01.1.1
    3: "sousClasse",    # 01.1.1.1
    4: "poste",         # 01.1.1.1.1
}


def get_level(code: str) -> int:
    """Get the hierarchy level based on number of dots in the code."""
    return code.count(".")


def get_url_for_code(code: str) -> str:
    """Build the INSEE URL for a given COICOP code."""
    level = get_level(code)
    level_name = LEVEL_NAMES.get(level)
    if level_name is None:
        raise ValueError(f"Unknown level for code: {code}")
    return f"{BASE_URL}/{level_name}/{code}"


def clean_text(text: str) -> str:
    """Clean extracted text by removing extra whitespace."""
    # Replace multiple whitespace with single space
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def fetch_description(code: str, session: requests.Session) -> dict[str, str | None]:
    """Fetch the description for a COICOP code from INSEE.

    Returns a dict with:
        - code: the COICOP code
        - description: the main description text
        - comprend: what the category includes
        - ne_comprend_pas: what the category excludes
    """
    url = get_url_for_code(code)
    result = {
        "code": code,
        "url": url,
        "description": None,
        "comprend": None,
        "ne_comprend_pas": None,
    }

    try:
        response = session.get(url, timeout=30)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Get all text content from the main content area
        # Remove script and style elements
        for element in soup(["script", "style", "nav", "header", "footer"]):
            element.decompose()

        # Find the main content - INSEE uses specific content divs
        main_content = (
            soup.find("div", class_="contenu-onglet") or
            soup.find("div", class_="contenu-central") or
            soup.find("main") or
            soup.find("article") or
            soup.body
        )

        if main_content:
            # Get all text content
            full_text = main_content.get_text(separator="\n", strip=True)

            # Extract description - text before "comprend" sections
            # Look for patterns like "La division X couvre..." or main descriptive text
            lines = full_text.split("\n")
            description_lines = []
            comprend_lines = []
            ne_comprend_lines = []

            mode = "description"  # Track which section we're in

            for line in lines:
                line = clean_text(line)
                if not line or len(line) < 3:
                    continue

                line_lower = line.lower()

                # Detect section changes
                if "ne comprend pas" in line_lower or "ne couvre pas" in line_lower:
                    mode = "ne_comprend_pas"
                    # Keep the line content after the marker
                    if ":" in line:
                        content = line.split(":", 1)[1].strip()
                        if content:
                            ne_comprend_lines.append(content)
                    continue
                elif "comprend" in line_lower and mode != "ne_comprend_pas":
                    # Check if it's "Ce poste comprend", "Ce groupe comprend", etc.
                    if re.search(r"(ce |cette |comprend\s*:)", line_lower):
                        mode = "comprend"
                        # Keep the line content after the marker
                        if ":" in line:
                            content = line.split(":", 1)[1].strip()
                            if content:
                                comprend_lines.append(content)
                        continue

                # Skip navigation/breadcrumb/metadata lines
                if any(skip in line_lower for skip in [
                    "accueil", "nomenclature", "ecoicop", "télécharger",
                    "navigation", "rechercher", "aller au contenu",
                    "métadonnées", "arborescence", "fil d'ariane",
                    "classification européenne", "date de publication",
                    "dernière mise à jour", "tous les niveaux",
                    "sous-classe", "groupe", "classe", "poste", "division"
                ]):
                    continue

                # Skip lines that look like timestamps or IDs (numbers only)
                if re.match(r'^\d+$', line):
                    continue

                # Skip ISO timestamps and date patterns
                if re.match(r'^\d{4}-\d{2}-\d{2}', line):
                    continue
                # Remove inline timestamps from lines
                line = re.sub(r'\d{4}-\d{2}-\d{2}T[\d:\.+]+', '', line).strip()
                # Remove date patterns like ": 19/12/2025"
                line = re.sub(r':\s*\d{2}/\d{2}/\d{4}', '', line).strip()
                if not line:
                    continue

                # Add content to appropriate section
                if mode == "description":
                    # Only add substantive description lines
                    if len(line) > 10 and not line.startswith(code):
                        description_lines.append(line)
                elif mode == "comprend":
                    comprend_lines.append(line)
                elif mode == "ne_comprend_pas":
                    ne_comprend_lines.append(line)

            # Compile results
            if description_lines:
                # Take the first few meaningful lines as description
                result["description"] = " ".join(description_lines[:5])

            if comprend_lines:
                result["comprend"] = " ".join(comprend_lines)

            if ne_comprend_lines:
                result["ne_comprend_pas"] = " ".join(ne_comprend_lines)

        logger.debug(f"Extracted for {code}: desc={bool(result['description'])}, "
                    f"comprend={bool(result['comprend'])}")

    except requests.RequestException as e:
        logger.warning(f"Failed to fetch {url}: {e}")
    except Exception as e:
        logger.warning(f"Error parsing {url}: {e}")

    return result


def scrape_all_codes(
    input_csv: str | Path,
    output_csv: str | Path,
    delay: float = 0.5,
    max_codes: int | None = None,
    skip_existing: bool = False,
) -> pd.DataFrame:
    """Scrape descriptions for all COICOP codes from the input CSV.

    Args:
        input_csv: Path to CSV with existing COICOP codes (columns: Libelle, Code)
        output_csv: Path to save the enriched data
        delay: Delay between requests in seconds (be nice to the server)
        max_codes: Maximum number of codes to process (for testing)
        skip_existing: Skip codes that already have descriptions in output file

    Returns:
        DataFrame with codes and their descriptions
    """
    # Load existing codes
    df = pd.read_csv(input_csv, sep=";", encoding="utf-8")
    df.columns = ["libelle", "code"]

    # Filter out technical codes (98.x, 99.x) that won't be on INSEE
    mask = ~df["code"].astype(str).str.startswith(("98", "99"))
    codes_df = df[mask].copy()

    codes_to_fetch = codes_df["code"].tolist()

    if max_codes:
        codes_to_fetch = codes_to_fetch[:max_codes]

    logger.info(f"Will fetch descriptions for {len(codes_to_fetch)} codes")

    # Create session for connection reuse
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "fr-FR,fr;q=0.9",
    })

    results = []
    for i, code in enumerate(codes_to_fetch):
        logger.info(f"[{i+1}/{len(codes_to_fetch)}] Fetching {code}...")
        result = fetch_description(code, session)
        results.append(result)

        # Progress logging
        if result["description"]:
            logger.info(f"  -> Got description ({len(result['description'])} chars)")
        else:
            logger.warning(f"  -> No description found")

        # Be nice to the server
        if delay > 0 and i < len(codes_to_fetch) - 1:
            time.sleep(delay)

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Merge with original data
    final_df = df.merge(results_df, on="code", how="left")

    # Save to CSV
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_path, sep=";", index=False, encoding="utf-8")

    # Summary statistics
    has_desc = final_df["description"].notna().sum()
    has_comprend = final_df["comprend"].notna().sum()
    logger.info(f"Saved {len(final_df)} entries to {output_path}")
    logger.info(f"  - {has_desc} with descriptions")
    logger.info(f"  - {has_comprend} with 'comprend' content")

    return final_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Scrape COICOP descriptions from INSEE")
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="data/20260130-coicop_et_codes_techniques.csv",
        help="Input CSV with COICOP codes",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="data/coicop_with_descriptions.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--delay", "-d",
        type=float,
        default=0.5,
        help="Delay between requests in seconds",
    )
    parser.add_argument(
        "--max", "-m",
        type=int,
        default=None,
        help="Maximum codes to fetch (for testing)",
    )

    args = parser.parse_args()

    scrape_all_codes(
        input_csv=args.input,
        output_csv=args.output,
        delay=args.delay,
        max_codes=args.max,
    )
