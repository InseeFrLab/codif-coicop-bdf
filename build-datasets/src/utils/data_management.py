import json


def concat_path_from_key(config, *keys):
    """
    Concatenate path with config file and keys values

    Args:
        config: YAML configuration file with name of S3 bucket
        *keys: one or few keys which can find relative path
    """
    value = config
    if keys is not None:
        for k in keys:
            value = value[k]
    return f"s3://{config['S3']['BUCKET']}{value}"


def load_stopwords(path="data/stopwords.json"):
    with open(path, "r", encoding="utf-8") as json_file:
        stopwords = json.load(json_file)
    return stopwords
