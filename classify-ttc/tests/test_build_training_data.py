"""Tests for src.data.build_training_data (local files only)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data.build_training_data import _read_synthetic, build_training_data

MODULE_DIR = Path(__file__).resolve().parent.parent

SYNTHETIC = pd.DataFrame(
    {
        "product": ["RIZ BASMATI TILDA", "SUCRE GARDEL", "PAIN COMPLET"],
        "code": ["01.1.1.1", "01.1.8.1", "01.1.1.2"],
        "libelle": ["Céréales", "Sucre", "Pain"],
    }
)


@pytest.fixture
def ddc_path(tmp_path, monkeypatch):
    monkeypatch.chdir(MODULE_DIR)  # stopwords path is CWD-relative
    path = tmp_path / "ddc.parquet"
    pd.DataFrame(
        {
            "description_ean": ["COQUILLETTES PANZANI", "RIZ LONG UNCLE BENS"],
            "coicop_code": ["01.1.1.3.1", "01.1.1.1.1"],
        }
    ).to_parquet(path)
    return str(path)


def _build(ddc_path, synthetic_path, tmp_path) -> pd.DataFrame:
    out = tmp_path / "train.parquet"
    build_training_data(ddc_path, str(out), synthetic_path=str(synthetic_path))
    return pd.read_parquet(out).sort_values(["code", "product"]).reset_index(drop=True)


def test_synthetic_csv_and_parquet_give_same_output(ddc_path, tmp_path):
    csv = tmp_path / "syn.csv"
    SYNTHETIC.to_csv(csv, sep=";", index=False)
    parquet = tmp_path / "syn.parquet"
    SYNTHETIC.to_parquet(parquet)

    from_csv = _build(ddc_path, csv, tmp_path)
    from_parquet = _build(ddc_path, parquet, tmp_path)

    pd.testing.assert_frame_equal(from_csv, from_parquet)
    assert (from_csv["source"] == "synthetic").sum() == 3
    assert (from_csv["source"] == "ddc").sum() == 2


def test_synthetic_parquet_code_cast_to_str(tmp_path):
    path = tmp_path / "syn.parquet"
    pd.DataFrame({"product": ["X"], "code": [1]}).to_parquet(path)
    assert _read_synthetic(str(path))["code"].tolist() == ["1"]


def test_synthetic_parquet_missing_column(tmp_path):
    path = tmp_path / "syn.parquet"
    SYNTHETIC.drop(columns="code").to_parquet(path)
    with pytest.raises(ValueError, match="code"):
        _read_synthetic(str(path))
