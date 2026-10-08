"""Colonnes de traçabilité du livrable : votes, scores bruts, type de conciliation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.traceability import (  # noqa: E402
    AUCUN_CODE,
    CANDIDAT_UNIQUE,
    CHOIX_LLM,
    CHOIX_SIRUS,
    CONSENSUS,
    REGEX,
    add_classifier_traceability,
    decision_columns,
    reconciliation_type,
)


def _result(**overrides):
    """Cinq produits, codes déjà tronqués/élagués comme en sortie de conciliation.

    p1 : unanimité ; p2 : choix entre deux codes, le retenu porté par 3 ;
    p3 : abstentions et valeur non-code ; p4 : code regex ; p5 : aucun code.
    """
    base = pd.DataFrame(
        {
            "id": ["p1", "p2", "p3", "p4", "p5"],
            "lcs_code": ["01.1.1", "01.1.2", "NON_CODABLE", None, None],
            "lcs_distance": [0.1, 0.3, 0.9, np.nan, np.nan],
            "rag_code": ["01.1.1", "01.1.2", np.nan, None, None],
            "rag_confidence": [0.9, 0.8, np.nan, np.nan, np.nan],
            "ragann_code": ["01.1.1", "01.1.2", "Pain de mie", None, None],
            "ragann_confidence": [0.95, 0.7, 0.5, np.nan, np.nan],
            "ttc_code_1": ["01.1.1", "07.2.1", " 02.1.1 ", None, None],
            "ttc_conf_1": [0.99, 0.6, 0.4, np.nan, np.nan],
            "_decision_code": ["01.1.1", "01.1.2", "02.1.1", None, None],
            "predict_code": [None, None, None, "98.4", None],
            "predicted_code": ["01.1.1", "01.1.2", "02.1.1", "98.4", None],
            "sirus_route": ["candidat_unique", "modele", "candidat_unique", None, None],
            "sirus_n_candidats": [1, 2, 1, None, None],
        }
    )
    return base.assign(**overrides)


def _trace(result):
    return add_classifier_traceability(result, result["_decision_code"].notna()).set_index("id")


def test_unanimity_counts_four_votes():
    out = _trace(_result())
    assert out.loc["p1", "n_classifiers_agreeing"] == 4
    assert all(out.loc["p1", f"proposed_by_{c}"] for c in ("lcs", "rag", "ragann", "ttc"))


def test_real_choice_counts_only_classifiers_carrying_the_code():
    out = _trace(_result())
    assert out.loc["p2", "n_classifiers_agreeing"] == 3
    assert not out.loc["p2", "proposed_by_ttc"]


def test_abstentions_and_non_codes_are_not_votes():
    """`NON_CODABLE`, NaN et un libellé ne votent pas ; un code entouré
    d'espaces, si."""
    out = _trace(_result())
    assert out.loc["p3", "n_classifiers_agreeing"] == 1
    assert out.loc["p3", "proposed_by_ttc"]
    assert not out.loc["p3", "proposed_by_lcs"]
    assert not out.loc["p3", "proposed_by_ragann"]


def test_llm_code_proposed_by_nobody_counts_zero():
    res = _result(_decision_code=["05.1.1", *[None] * 4], predicted_code=["05.1.1", None, None, "98.4", None])
    out = _trace(res)
    assert out.loc["p1", "n_classifiers_agreeing"] == 0


def test_rows_outside_reconciliation_are_na():
    out = _trace(_result())
    for pid in ("p4", "p5"):
        assert pd.isna(out.loc[pid, "n_classifiers_agreeing"])
        assert pd.isna(out.loc[pid, "proposed_by_lcs"])


def test_raw_scores_are_delivered_even_for_disagreeing_classifier():
    out = _trace(_result())
    assert out.loc["p2", "ttc_confidence"] == 0.6
    assert out.loc["p2", "lcs_distance"] == 0.3
    assert str(out["ttc_confidence"].dtype) == "Float64"


def test_missing_classifier_columns_give_na():
    """Run antérieur à RAG-annotations : colonnes absentes, pas d'erreur."""
    res = _result().drop(columns=["ragann_code", "ragann_confidence"])
    out = _trace(res)
    assert out["ragann_confidence"].isna().all()
    assert out.loc["p1", "n_classifiers_agreeing"] == 3
    assert not out.loc["p1", "proposed_by_ragann"]


def test_reconciliation_type_sirus_uses_route():
    kinds = reconciliation_type(_result(), "sirus")
    assert list(kinds) == [CANDIDAT_UNIQUE, CHOIX_SIRUS, CANDIDAT_UNIQUE, REGEX, AUCUN_CODE]


def test_reconciliation_type_sirus_falls_back_on_candidate_count():
    """Run antérieur au routage : pas de `sirus_route`, même résultat."""
    kinds = reconciliation_type(_result().drop(columns=["sirus_route"]), "sirus")
    assert list(kinds) == [CANDIDAT_UNIQUE, CHOIX_SIRUS, CANDIDAT_UNIQUE, REGEX, AUCUN_CODE]


def test_reconciliation_type_llm():
    res = _result(llm_model=["consensus", "gpt-x", "gpt-x", None, None]).drop(
        columns=["sirus_route", "sirus_n_candidats"]
    )
    kinds = reconciliation_type(res, "llm")
    assert list(kinds) == [CONSENSUS, CHOIX_LLM, CHOIX_LLM, REGEX, AUCUN_CODE]


def test_decision_columns_keeps_only_available():
    cols = decision_columns(["id", "lcs_code", "lcs_distance", "ttc_code_1", "sirus_n_candidats"], "sirus")
    assert cols == ["lcs_code", "lcs_distance", "ttc_code_1", "sirus_n_candidats"]
