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


def expand_paths(obj, **kwargs):
    """Recursively apply str.format(**kwargs) to every string in a nested
    dict/list structure. Used to substitute {run_id} / {run_date} in config
    path templates."""
    if isinstance(obj, dict):
        return {k: expand_paths(v, **kwargs) for k, v in obj.items()}
    if isinstance(obj, list):
        return [expand_paths(v, **kwargs) for v in obj]
    if isinstance(obj, str):
        return obj.format(**kwargs)
    return obj
