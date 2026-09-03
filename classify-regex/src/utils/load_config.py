import yaml

# `expand_paths` vit dans `codif_common` (elle existait en 4 exemplaires).
from codif_common.paths import expand_paths  # noqa: F401  (ré-export)


def load_config(config_path="config/config.yaml"):
    """
    Load configuration from YAML file

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        dict: Configuration dictionary
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config
