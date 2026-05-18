import os
import pandas as pd


def export_parquet_s3(df: pd.DataFrame, path: str) -> None:
    df.to_parquet(
        f"s3://{path}",
        storage_options={
            "key": os.environ["AWS_ACCESS_KEY_ID"],
            "secret": os.environ["AWS_SECRET_ACCESS_KEY"],
            "token": os.environ.get("AWS_SESSION_TOKEN"),
            "client_kwargs": {
                "endpoint_url": f"https://{os.environ['AWS_S3_ENDPOINT']}"
            },
        },
        index=False
    )