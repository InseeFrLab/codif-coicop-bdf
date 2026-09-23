"""La métrique d'entraînement mesure-t-elle le modèle, ou ses sources ?

`accuracy_product` est gonflée, et c'est structurel : les candidats sont les
codes distincts proposés par les 4 classifieurs, donc un produit sur lequel ils
s'accordent n'en laisse qu'un, et l'argmax ne peut que le retenir. Ces produits
comptent comme des succès du modèle alors qu'il n'a rien choisi — et ce sont les
cas faciles, donc majoritairement corrects.

`accuracy_product_multi` est la même mesure restreinte aux produits où il y avait
effectivement quelque chose à trancher. Elle est publiée à côté de l'autre et non
à sa place, pour que la série historique reste comparable.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.train import build_calibration  # noqa: E402


def _test_set():
    """Quatre produits. Deux à candidat unique, tous deux corrects : ce sont eux
    qui gonflent. Deux à deux candidats, dont un seul est bien tranché."""
    return pd.DataFrame(
        {
            "id": ["a", "b", "c", "c", "d", "d"],
            "correcte": [1, 1, 1, 0, 0, 1],
        }
    )


# Probabilités : sur `c` le modèle choisit le bon candidat, sur `d` le mauvais.
PROBA = np.array([0.9, 0.8, 0.7, 0.2, 0.6, 0.3])


def test_the_pooled_metric_counts_products_with_nothing_to_choose():
    """3 produits sur 4 « réussis », dont 2 sans aucun choix à faire."""
    assert build_calibration(_test_set(), PROBA)["metrics"]["accuracy_product"] == 0.75


def test_the_multi_candidate_metric_is_the_one_that_measures_the_model():
    """Sur les seuls produits à plusieurs candidats : 1 sur 2."""
    m = build_calibration(_test_set(), PROBA)["metrics"]
    assert m["accuracy_product_multi"] == 0.5
    assert m["n_test_products_multi"] == 2


def test_the_gap_between_the_two_is_the_inflation():
    m = build_calibration(_test_set(), PROBA)["metrics"]
    assert m["accuracy_product"] > m["accuracy_product_multi"]


def test_a_test_set_without_any_real_choice_yields_nan_not_zero():
    """Zéro serait un résultat — « le modèle se trompe toujours » — là où il n'y
    a simplement rien à mesurer."""
    single = pd.DataFrame({"id": ["a", "b"], "correcte": [1, 0]})
    m = build_calibration(single, np.array([0.9, 0.8]))["metrics"]
    assert m["n_test_products_multi"] == 0
    assert np.isnan(m["accuracy_product_multi"])


def test_the_upper_bound_still_covers_every_product():
    """La borne haute n'est pas restreinte : elle dit ce qui était atteignable
    sur tout le test, candidats uniques compris."""
    assert build_calibration(_test_set(), PROBA)["metrics"]["upper_bound"] == 1.0
