"""Le scorer Python reproduit-il ``sirus.predict`` ?

C'est le test qui rend sûre la décision « entraîner en R, prédire en Python ».
Les fixtures de ``tests/golden/`` sont produites par ``R/make_golden.R`` et
committées : R n'est pas exécuté ici (le dépôt n'a pas de CI qui installe R).

Le jeu à scorer est **construit**, pas échantillonné : il contient pour chaque
seuil numérique du modèle une ligne exactement sur le seuil et ses deux voisins
immédiats — c'est là que ``<`` se distingue de ``>=``, et donc là qu'un seuil
arrondi à l'export basculerait dans la mauvaise branche.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.candidates import FEATURES  # noqa: E402
from src.scorer import (  # noqa: E402
    SCHEMA_VERSION,
    load_rules,
    pick_best,
    score,
    scorable_mask,
)

GOLDEN = Path(__file__).parent / "golden"


@pytest.fixture(scope="module")
def rules():
    return load_rules(GOLDEN / "rules.json")


NUMERIC_FEATURES = [f for f in FEATURES if f != "code_candidat_n1"]


@pytest.fixture(scope="module")
def expected():
    """Fixture golden, avec les nombres reconvertis par ``float()`` de Python.

    Tout est lu en chaînes puis converti explicitement pour deux raisons :

    - ``float()`` de Python est correctement arrondi, alors que le parseur
      rapide de pandas peut perdre 1 ULP — ce qui suffirait à faire basculer une
      ligne posée exactement sur un seuil dans l'autre branche, et donc à faire
      échouer le test pour une raison étrangère au scorer ;
    - aucun forçage en entier : les lignes bordant le seuil sur ``nb_votants``
      portent 2 - eps et 2 + eps. Un cast en ``int64`` les tronquerait et le
      test ne testerait plus les frontières.

    Les codes restent en chaînes, sinon "01" deviendrait l'entier 1.
    """
    df = pd.read_csv(GOLDEN / "expected_proba.csv", dtype=str)
    for col in NUMERIC_FEATURES:
        df[col] = [float(s) for s in df[col]]
    return df


def test_scorer_bit_identical_to_r(rules, expected):
    got = score(rules, expected[FEATURES])
    want = np.array([float(s) for s in expected["proba_R"]], dtype="float64")

    # Égalité bit-à-bit, pas `allclose`. `src/scorer.py::_r_mean` reproduit
    # l'algorithme de `mean()` de R (accumulation en long double + passe de
    # correction) précisément pour que cette égalité soit atteignable. Si elle
    # échoue, c'est un vrai signal : soit l'export perd de la précision, soit le
    # scorer ne reproduit plus la sémantique du paquet.
    assert np.array_equal(got, want), (
        f"écart max {np.abs(got - want).max():.3e} sur "
        f"{int((got != want).sum())}/{len(want)} lignes"
    )


def test_threshold_values_are_on_boundaries(rules, expected):
    """Le jeu golden borde-t-il réellement chaque seuil ?

    Sans cette garantie, le test précédent pourrait passer avec des seuils
    arrondis : il faut au moins une ligne dont la valeur vaut exactement le
    seuil pour que l'arrondi change la branche empruntée.
    """
    seuils = {
        (c.var, c.value)
        for r in rules.rules
        for c in r.conditions
        if c.op != "in"
    }
    assert seuils, "modèle golden sans condition numérique : fixture inutilisable"
    for var, seuil in seuils:
        assert (expected[var].to_numpy() == seuil).any(), (
            f"aucune ligne du golden ne vaut exactement {seuil!r} sur `{var}` : "
            "le test ne détecterait pas un arrondi de ce seuil"
        )


def test_threshold_strings_roundtrip():
    """Chaque seuil exporté se relit-il à l'identique ?

    Garde-fou contre un futur « on simplifie » qui remplacerait le formatage
    17 chiffres par un arrondi : `jsonlite` par défaut donne 4 chiffres et
    `digits = NA` seulement 15, ni l'un ni l'autre ne round-trippent.
    """
    rt = pd.read_csv(GOLDEN / "threshold_roundtrip.csv", dtype=str)
    assert len(rt) > 0
    for row in rt.itertuples():
        assert row.threshold_string == row.reformatted
        assert f"{float(row.threshold_string):.17g}" == row.threshold_string


def test_rules_json_has_no_rounded_floats(rules):
    """Les nombres de rules.json sont-ils bien des chaînes ?

    Un nombre JSON aurait été écrit par `jsonlite` avec sa précision par défaut.
    Les sérialiser en chaînes rend l'exactitude vérifiable plutôt qu'espérée.
    """
    payload = json.loads((GOLDEN / "rules.json").read_text(encoding="utf-8"))
    assert isinstance(payload["mean"], str)
    for rule in payload["rules"]:
        assert all(isinstance(o, str) for o in rule["outputs"])
        for cond in rule["conditions"]:
            if cond["op"] != "in":
                assert isinstance(cond["value"], str)


def test_aggregation_is_declared_unweighted(rules):
    """rules.json déclare-t-il explicitement l'agrégation ?

    En classification `sirus.predict` fait une moyenne NON pondérée : le champ
    `rule.weights` de l'objet R existe mais n'est jamais lu. La confusion est
    facile (la ridge à coefficients positifs est l'agrégation de la variante
    régression), d'où ce champ explicite.
    """
    assert rules.meta["aggregation"] == "mean"
    assert "weights" not in rules.meta
    assert "intercept" not in rules.meta


def test_zero_rules_falls_back_to_mean(expected):
    """Modèle sans aucune règle : R renvoie le taux de base (sirus.R:405)."""
    rules_vides = load_rules(GOLDEN / "rules_empty.json")
    got = score(rules_vides, expected[FEATURES])
    assert np.array_equal(got, np.full(len(expected), rules_vides.mean))


def test_unseen_factor_level_is_dropped_not_scored(rules, expected):
    """Une modalité inédite doit être ÉCARTÉE, pas scorée.

    ``isin()`` renverrait tranquillement ``False`` et produirait une probabilité
    *plausible* que R n'aurait jamais produite (il abandonne sur NA via
    ``data.check``). Écarter est le seul comportement qui reproduit la sémantique
    de R tout en restant exploitable en production.
    """
    table = expected[FEATURES].copy()
    table.loc[table.index[0], "code_candidat_n1"] = "77"  # division inexistante

    mask = scorable_mask(rules, table)
    assert not mask.iloc[0], "la modalité inédite aurait dû être écartée"
    assert mask.iloc[1:].all(), "les autres lignes doivent rester scorables"

    # Et les lignes conservées gardent exactement le score qu'elles avaient.
    got = score(rules, table[mask])
    want = np.array([float(s) for s in expected["proba_R"]], dtype="float64")[1:]
    assert np.array_equal(got, want)


def test_nan_in_numeric_feature_raises(rules, expected):
    """NaN vaut False pour `<` ET `>=` : une branche que R n'emprunte jamais.

    ``build_candidate_table`` rend le cas structurellement impossible ; ce test
    verrouille le fait que le scorer refuse plutôt que d'inventer un score.
    """
    table = expected[FEATURES].copy()
    table.loc[table.index[0], "conf_ttc"] = np.nan
    with pytest.raises(ValueError, match="manquantes"):
        score(rules, table)


def test_features_order_matches_model(rules):
    """FEATURES et le modèle doivent lister les mêmes colonnes."""
    assert tuple(FEATURES) == rules.features


def test_unknown_schema_version_raises(tmp_path):
    """Une version de schéma inconnue doit refuser, pas relire au hasard.

    Ce scorer est la seule chose qui sait interpréter rules.json, et il doit
    rester correct longtemps après que le paquet R qui a produit le fichier soit
    devenu difficile à réinstaller. Mieux vaut échouer que produire des
    probabilités plausibles et fausses.
    """
    payload = json.loads((GOLDEN / "rules.json").read_text(encoding="utf-8"))
    payload["schema_version"] = 99
    cible = tmp_path / "rules.json"
    cible.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version"):
        load_rules(cible)


def test_current_golden_declares_supported_schema():
    """La fixture doit déclarer la version que le scorer implémente."""
    payload = json.loads((GOLDEN / "rules.json").read_text(encoding="utf-8"))
    assert payload["schema_version"] == SCHEMA_VERSION


# --- Contrat de sortie de pick_best -----------------------------------------
# L'étape livre un code et un score par produit, sans verdict : décider d'un
# seuil d'exploitation est une question métier, instruite par le rapport
# d'évaluation. Ces tests verrouillent cette absence de verdict.


def _table_deux_produits():
    """Deux produits : p1 avec 3 candidats, p2 avec 1 seul."""
    return pd.DataFrame(
        {
            "id": ["p1", "p1", "p1", "p2"],
            "code_candidat": ["01.1", "02.3", "05.2", "07.4"],
        }
    )


def test_pick_best_keeps_argmax_per_product():
    table = _table_deux_produits()
    out = pick_best(table, np.array([0.20, 0.75, 0.31, 0.05]))

    assert list(out["id"]) == ["p1", "p2"]
    assert dict(zip(out["id"], out["sirus_code"])) == {"p1": "02.3", "p2": "07.4"}
    assert out.loc[out["id"] == "p1", "sirus_proba"].iloc[0] == 0.75
    # Le score du gagnant est conservé tel quel, même très bas : c'est
    # l'information que l'aval exploitera pour arbitrer.
    assert out.loc[out["id"] == "p2", "sirus_proba"].iloc[0] == 0.05


def test_pick_best_emits_no_verdict():
    """Aucune colonne de décision : ni `sirus_decision`, ni équivalent."""
    out = pick_best(_table_deux_produits(), np.array([0.2, 0.75, 0.31, 0.05]))
    assert set(out.columns) == {"id", "sirus_code", "sirus_proba", "sirus_n_candidats"}
    assert "sirus_decision" not in out.columns


def test_pick_best_counts_scored_candidates():
    """`sirus_n_candidats` explique pourquoi un produit a le code qu'il a."""
    out = pick_best(_table_deux_produits(), np.array([0.2, 0.75, 0.31, 0.05]))
    assert dict(zip(out["id"], out["sirus_n_candidats"])) == {"p1": 3, "p2": 1}


def test_pick_best_ties_keep_first():
    """Ex æquo : le premier, comme `slice_max(..., with_ties = FALSE)` en R."""
    table = pd.DataFrame({"id": ["p1", "p1"], "code_candidat": ["01.1", "02.3"]})
    out = pick_best(table, np.array([0.6, 0.6]))
    assert len(out) == 1
    assert out["sirus_code"].iloc[0] == "01.1"
