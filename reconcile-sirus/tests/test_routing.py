"""Routage amont des produits à candidat unique, et filtre d'entraînement associé.

Un produit à candidat unique n'offre rien à choisir : il reçoit ce code sans
passer par le modèle. Le filtre d'entraînement (`keep_multi_candidates`) reste
disponible en option, non appliqué par défaut.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.candidates import FEATURES  # noqa: E402
from src.scorer import (  # noqa: E402
    ROUTE_CANDIDAT_UNIQUE,
    ROUTE_MODELE,
    load_rules,
    route_and_score,
    score,
    split_single_candidates,
)
from src.train import keep_multi_candidates, split_by_product  # noqa: E402

GOLDEN = Path(__file__).parent / "golden"


@pytest.fixture(scope="module")
def rules():
    return load_rules(GOLDEN / "rules.json")


def _ligne(id_, code, nb_votants, division="01"):
    """Une ligne candidat aux features plausibles (valeurs sans importance ici)."""
    return {
        "id": id_,
        "code_candidat": code,
        "vote_lcs": 1,
        "vote_rag": int(nb_votants >= 2),
        "vote_ragann": int(nb_votants >= 3),
        "vote_ttc": int(nb_votants >= 4),
        "nb_votants": nb_votants,
        "conf_rag": 0.8,
        "conf_ragann": 0.9,
        "conf_ttc": 0.85,
        "dist_lcs": 1.0,
        "code_candidat_n1": division,
    }


def _table():
    """p1 : unanimité ; p2 : candidat unique après abstentions ; p3 : 3 candidats."""
    return pd.DataFrame(
        [
            _ligne("p1", "01.1.1.1", 4),
            _ligne("p2", "02.1.1.1", 2, "02"),
            _ligne("p3", "01.1.1.1", 2),
            _ligne("p3", "01.1.2.1", 1),
            _ligne("p3", "05.1.1.1", 1, "05"),
        ]
    )[["id", "code_candidat", *FEATURES]]


def test_split_routes_every_single_candidate_product():
    """Unanimité ET candidat unique par abstention sont routés, sans score."""
    routed, multi = split_single_candidates(_table())

    assert sorted(routed["id"]) == ["p1", "p2"]
    assert dict(zip(routed["id"], routed["sirus_code"])) == {
        "p1": "01.1.1.1",
        "p2": "02.1.1.1",
    }
    assert routed["sirus_proba"].isna().all()
    assert (routed["sirus_n_candidats"] == 1).all()
    assert (routed["sirus_route"] == ROUTE_CANDIDAT_UNIQUE).all()
    assert set(multi["id"]) == {"p3"} and len(multi) == 3


def test_route_and_score_scores_only_multi_candidates(rules):
    table = _table()
    decided, proba = route_and_score(rules, table)

    assert sorted(decided["id"]) == ["p1", "p2", "p3"]
    # Le modèle n'a vu que les 3 candidats de p3.
    assert proba.shape == (3,)
    p3 = decided[decided["id"] == "p3"].iloc[0]
    assert p3["sirus_route"] == ROUTE_MODELE
    assert p3["sirus_n_candidats"] == 3
    attendu = score(rules, table[table["id"] == "p3"][FEATURES])
    assert p3["sirus_proba"] == attendu.max()


def test_route_and_score_all_single_candidates_does_not_score(rules):
    """Un run sans aucun désaccord est légitime : tout est routé, rien n'est scoré."""
    table = _table()
    decided, proba = route_and_score(rules, table[table["id"] != "p3"])

    assert proba.size == 0
    assert sorted(decided["id"]) == ["p1", "p2"]
    assert (decided["sirus_route"] == ROUTE_CANDIDAT_UNIQUE).all()


def test_route_and_score_drops_unscorable_multi_product(rules):
    """Un produit multi-candidats dont aucun candidat n'est scorable manque à la
    sortie : c'est à l'appelant de le rattraper en `aucun_candidat`."""
    table = _table()
    table.loc[table["id"] == "p3", "code_candidat_n1"] = "ZZ"
    decided, _ = route_and_score(rules, table)
    assert "p3" not in set(decided["id"])


def test_keep_multi_candidates_after_split_preserves_test_products():
    """Filtrer après le split garde, pour les produits multi, le même côté
    train/test qu'un entraînement sans filtre : les deux restent comparables."""
    rng = np.random.default_rng(0)
    lignes = []
    for i in range(200):
        for k in range(int(rng.integers(1, 4))):
            lignes.append(_ligne(f"p{i:03d}", f"01.1.1.{k}", 1))
    table = split_by_product(pd.DataFrame(lignes), frac=0.8, seed=42)

    filtre = keep_multi_candidates(table)

    assert (filtre.groupby("id").size() >= 2).all()
    multi_ids = set(table.groupby("id").size().loc[lambda s: s >= 2].index)
    assert set(filtre["id"]) == multi_ids
    avant = table.drop_duplicates("id").set_index("id")["split"]
    apres = filtre.drop_duplicates("id").set_index("id")["split"]
    assert (apres == avant.loc[apres.index]).all()
