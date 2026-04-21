import yaml


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
