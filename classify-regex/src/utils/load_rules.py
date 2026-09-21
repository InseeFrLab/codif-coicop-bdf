import yaml
import pandas as pd
import re


def load_regex_rules(yaml_path: str) -> list[dict]:
    with open(yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config["rules"]


def apply_regex_rules(series: pd.Series, rules: list[dict]) -> pd.Series:
    """Applique une liste de règles {pattern, code} sur une Series."""
    # Pré-compilation des patterns
    compiled_rules = [
        (re.compile(rule["pattern"], re.IGNORECASE), rule["code"])
        for rule in rules
    ]

    def classify(text):
        if not isinstance(text, str):
            return None
        for pattern, code in compiled_rules:
            if pattern.search(text):
                return code
        return None

    return series.map(classify)
