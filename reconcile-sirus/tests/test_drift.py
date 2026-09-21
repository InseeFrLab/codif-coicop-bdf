"""Contrôle de dérive amont : détecte-t-il ce qu'il doit, et se tait-il sinon ?

Un contrôle qui alerte sur un run sain est pire que pas de contrôle : on apprend
à ignorer ses avertissements, et il ne sert plus quand la dérive est réelle.
Ces tests verrouillent les deux moitiés — la détection ET le silence.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.train import check_drift, feature_distribution  # noqa: E402


def _table(n=100, *, division="01", conf_ragann=1.0, nb_votants=2):
    return pd.DataFrame(
        {
            "conf_ragann": [conf_ragann] * n,
            "conf_ttc": [0.8] * n,
            "nb_votants": [nb_votants] * n,
            "code_candidat_n1": [division] * n,
        }
    )


def _meta(table, *, share_without_candidate=0.0):
    return {
        "training": {
            "feature_distributions": feature_distribution(table),
            "share_without_candidate": share_without_candidate,
        }
    }


def test_confidences_are_binned_not_counted_raw():
    """Les confiances continues doivent être résumées en tranches.

    Compter leurs valeurs brutes produirait des milliers de modalités uniques :
    la référence, tronquée, ne couvrirait qu'une poignée de lignes et la
    comparaison alerterait sur un run parfaitement sain — c'est le défaut qui a
    motivé ce découpage.
    """
    table = pd.DataFrame(
        {
            "conf_ragann": [-1.0, 0.3, 0.7, 0.95, 1.0],
            "conf_ttc": [-1.0, -1.0, -1.0, -1.0, -1.0],
            "nb_votants": [1, 1, 2, 3, 4],
            "code_candidat_n1": ["01", "01", "02", "02", "05"],
        }
    )
    dist = feature_distribution(table)
    assert set(dist["conf_ragann"]) <= {"absent", "faible", "moyenne", "haute", "certaine"}
    assert dist["conf_ragann"]["absent"] == 1
    assert dist["conf_ttc"]["absent"] == 5
    # Les features discrètes gardent leurs modalités telles quelles.
    assert dist["nb_votants"] == {"1": 2, "2": 1, "3": 1, "4": 1}


def test_identical_run_is_silent(caplog):
    """Un run identique à l'entraînement ne doit produire aucune alerte."""
    table = _table()
    with caplog.at_level(logging.WARNING, logger="sirus.train"):
        check_drift(_meta(table), table, {"n_products_in": 100, "ids_without_candidate": []})
    assert not caplog.records, [r.message for r in caplog.records]


def test_shifted_division_distribution_warns(caplog):
    """Toutes les divisions changent ⇒ avertissement sur `code_candidat_n1`."""
    ref = _table(division="01")
    run = _table(division="07")
    with caplog.at_level(logging.WARNING, logger="sirus.train"):
        check_drift(_meta(ref), run, {"n_products_in": 100, "ids_without_candidate": []})
    assert any("code_candidat_n1" in r.message for r in caplog.records)


def test_confidence_shift_warns(caplog):
    """RAG-ANN passant de « certaine » à « moyenne » doit alerter.

    C'est le scénario documenté : un ajustement de prompt en amont déplace
    `conf_ragann`, et les règles apprises contre l'ancienne distribution
    changent de sens sans que rien ne le signale autrement.
    """
    ref = _table(conf_ragann=1.0)
    run = _table(conf_ragann=0.7)
    with caplog.at_level(logging.WARNING, logger="sirus.train"):
        check_drift(_meta(ref), run, {"n_products_in": 100, "ids_without_candidate": []})
    assert any("conf_ragann" in r.message for r in caplog.records)


def test_no_candidate_surge_is_an_error(caplog):
    """Un bond de la part de produits sans candidat est le signal le plus fort.

    C'est ce qui arrive quand le format des codes émis en amont change : ils ne
    sont plus reconnus, et le volume s'effondre en silence.
    """
    table = _table()
    with caplog.at_level(logging.ERROR, logger="sirus.train"):
        check_drift(
            _meta(table, share_without_candidate=0.0),
            table,
            {"n_products_in": 100, "ids_without_candidate": [f"p{i}" for i in range(60)]},
        )
    erreurs = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert erreurs and "sans aucun candidat" in erreurs[0].message


def test_model_without_reference_says_so(caplog):
    """Un modèle antérieur à ce contrôle ne doit pas faire échouer, mais le dire."""
    with caplog.at_level(logging.INFO, logger="sirus.train"):
        check_drift({}, _table(), {"n_products_in": 100, "ids_without_candidate": []})
    assert any("contrôle de dérive impossible" in r.message for r in caplog.records)
