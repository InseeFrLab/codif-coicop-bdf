"""Tests for src.data.synthetic_generator (no network: fake LLM)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from src.data import synthetic_generator as sg
from src.preprocessing.data_preparation import preprocess_text

MODULE_DIR = Path(__file__).resolve().parent.parent
STOPWORDS_PATH = MODULE_DIR / "data" / "text" / "stopwords.json"


@pytest.fixture(scope="module")
def stopwords() -> list[str]:
    with open(STOPWORDS_PATH, encoding="utf-8") as f:
        return json.load(f)


TRICKY = [
    "TILDA RIZ BASMATI 500GRS",
    "Crème fraîche épaisse 30% M.G.",
    "RIZ riz RIZ au lait",
    "C++ 3X20CL a b c",
    "rien du tout",
    "rien",
    "le la les de",
    "   ",
    "",
    "!!! ???",
    "Pâté de campagne, 2 x 125 g",
    "OEUFS X12 ST JACQUES",
]


@pytest.mark.parametrize("text", TRICKY)
def test_clean_product_matches_preprocess_text(text, stopwords):
    df = preprocess_text(pd.DataFrame({"product": [text]}), "product", stopwords)
    expected = df["product"].iloc[0] if len(df) else None
    assert sg.clean_product(text, stopwords) == expected


def test_parse_response_strips_bullets_and_intro():
    text = "Voici les exemples:\n- RIZ 1KG\n2. SUCRE 750G\n12) PAIN\n\n• KIRI X6"
    assert sg.parse_response(text) == ["RIZ 1KG", "SUCRE 750G", "PAIN", "KIRI X6"]


def test_parse_json_response():
    assert sg._parse_json_response('bla ["A B", "C D"] bla') == ["A B", "C D"]
    assert sg._parse_json_response('{"a": 1}') is None
    assert sg._parse_json_response('["ok", ""]') is None
    assert sg._parse_json_response("") is None


def test_allocate_bands():
    counts = {"a": 0, "b": 1, "c": 99, "d": 100, "e": 500, "f": 1000}
    assert sg.allocate(counts) == {
        "a": 400, "b": 300, "c": 300, "d": 200, "e": 100, "f": 0,
    }
    assert sg.allocate(None) == {}


def test_curate_cross_code_conflict_keeps_first_sorted_code(stopwords):
    per_code = {"02.1": ["Riz basmati 1g"], "01.1": ["RIZ BASMATI 500G", "!!"]}
    cleaned, frag = sg.curate(per_code, stopwords=stopwords)
    assert cleaned == {"01.1": ["RIZ BASMATI 500G"], "02.1": []}
    [conflict] = frag["cross_code_conflicts"]
    assert conflict["first_code"] == "01.1"
    assert conflict["dropped_code"] == "02.1"


class FakeLLM:
    """No structured output (tier 1 fails); tier 2 JSON; verify keeps all."""

    def __init__(self):
        self.calls = 0

    def with_structured_output(self, schema):
        raise NotImplementedError

    def invoke(self, prompt: str):
        self.calls += 1
        if prompt.startswith("Categorie:"):  # verify pass
            return SimpleNamespace(content="\n".join(["1"] * 10))
        code = next(ln for ln in prompt.splitlines() if "Code COICOP:" in ln)
        code = code.split(":", 1)[1].strip()
        # Digits are cleaned away: spell the code so items stay distinct per code.
        tag = "".join("ABCDEFGHIJ"[int(c)] if c.isdigit() else "X" for c in code)
        items = [f"PRODUIT {tag} VARIANTE {w}" for w in ("ALPHA", "BETA", "GAMMA")]
        return SimpleNamespace(content=json.dumps(items))


def _run(tmp_path, llm):
    return sg.generate_and_save(
        llm=llm,
        coicop_path=MODULE_DIR / sg.DEFAULT_COICOP_PATH,
        examples_per_category=3,
        stopwords_path=STOPWORDS_PATH,
        output_csv=tmp_path / "out.csv",
        raw_dir=tmp_path / "raw",
        manifest_path=tmp_path / "manifest.json",
        max_categories=2,
        reference_path=MODULE_DIR / sg.DEFAULT_REFERENCE_PATH,
    )


def test_generate_and_save_end_to_end_and_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(sg.time, "sleep", lambda _: None)
    llm = FakeLLM()
    out = _run(tmp_path, llm)

    # Read exactly as build_training_data does.
    df = pd.read_csv(
        out, sep=";", skiprows=1, header=None, usecols=[0, 1],
        names=["product", "code"],
    )
    assert len(df) == 6
    assert df["code"].nunique() == 2
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["per_code"]) == set(df["code"])
    assert all(v["accepted"] == 3 for v in manifest["per_code"].values())

    # Re-run resumes from the manifest: no new LLM call, same output.
    calls = llm.calls
    _run(tmp_path, llm)
    assert llm.calls == calls
    assert pd.read_csv(out, sep=";").shape[0] == 6


def test_dry_run_unknown_code_raises():
    with pytest.raises(ValueError):
        sg.dry_run(
            coicop_path=MODULE_DIR / sg.DEFAULT_COICOP_PATH,
            only_codes=["00.0.0.0"],
            reference_path=MODULE_DIR / sg.DEFAULT_REFERENCE_PATH,
        )
