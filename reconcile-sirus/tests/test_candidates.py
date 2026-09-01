"""Invariants de la table candidat-level.

Ce fichier ne compare pas à une sortie de R, contrairement à
``test_scorer_golden.py``, et c'est délibéré : le ``construire_table_long()`` de
l'expérimentation R appliquait sa propre troncature (comptage de chiffres +
cascade de retrait des ``.0`` terminaux), là où le pipeline s'appuie sur
``prune.trunc_and_prune_lvl4`` (découpage par segments + table de mapping). Les
deux diffèrent sur tout code à ``.0`` terminal, et le filtre d'abstention n'est
pas le même non plus. Un golden R-vs-Python échouerait donc pour des raisons
connues et sans rapport avec le portage de la logique de votes.

Ce qui compte ici est vérifié directement : les sentinelles, le filtrage des
abstentions, la visibilité des produits sans candidat, et l'absence de NaN — les
quatre propriétés dont dépend la validité des seuils appris.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.candidates import (  # noqa: E402
    CONF_MISSING,
    DIST_MISSING,
    FEATURES,
    build_candidate_table,
)


def _merged(**overrides) -> pd.DataFrame:
    base = {
        "id": ["p1"],
        "lcs_code": ["01.1.2"],
        "lcs_distance": [0.2],
        "rag_code": ["01.1.2"],
        "rag_confidence": [0.8],
        "ragann_code": ["02.3"],
        "ragann_confidence": [0.9],
        "ttc_code_1": ["01.1.2"],
        "ttc_conf_1": [0.7],
        "code_lvl4": ["01.1.2"],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def test_one_row_per_distinct_candidate():
    table, diag = build_candidate_table(_merged())
    assert list(table["code_candidat"]) == ["01.1.2", "02.3"]
    assert diag["n_candidates"] == 2


def test_votes_and_nb_votants():
    table, _ = build_candidate_table(_merged())
    gagnant = table[table["code_candidat"] == "01.1.2"].iloc[0]
    assert (gagnant["vote_lcs"], gagnant["vote_rag"], gagnant["vote_ttc"]) == (1, 1, 1)
    assert gagnant["vote_ragann"] == 0
    assert gagnant["nb_votants"] == 3

    autre = table[table["code_candidat"] == "02.3"].iloc[0]
    assert autre["nb_votants"] == 1


def test_sentinels_when_classifier_did_not_vote():
    """Un classifieur qui n'a pas voté pour CE candidat reçoit la sentinelle.

    C'est la propriété la plus load-bearing du module : les seuils des règles ont
    été appris contre ces valeurs exactes. Un `0` à la place de `-1` entrerait en
    collision avec une vraie confiance faible.
    """
    table, _ = build_candidate_table(_merged())
    autre = table[table["code_candidat"] == "02.3"].iloc[0]
    assert autre["conf_rag"] == CONF_MISSING
    assert autre["conf_ttc"] == CONF_MISSING
    assert autre["dist_lcs"] == DIST_MISSING
    assert autre["conf_ragann"] == 0.9  # celui-là a voté


def test_voted_but_missing_score_gets_sentinel():
    """`coalesce(score, sentinelle)` : voté sans score ⇒ sentinelle, pas NaN.

    Le `fillna` est à l'intérieur de la branche « a voté ». Le mettre après
    donnerait un NaN, que le scorer refuse et que R n'accepte pas non plus.
    """
    table, _ = build_candidate_table(_merged(ttc_conf_1=[None]))
    gagnant = table[table["code_candidat"] == "01.1.2"].iloc[0]
    assert gagnant["vote_ttc"] == 1
    assert gagnant["conf_ttc"] == CONF_MISSING


@pytest.mark.parametrize("abstention", [None, "N/A", "n/a", "NONE", "", "null"])
def test_abstentions_are_not_candidates(abstention):
    """Une abstention n'est pas un code candidat.

    Les sentinelles historiques ``AUCUNE_SUGGESTION``/``NON_CODABLE`` n'existent
    pas dans ce pipeline : l'abstention arrive en NULL ou sous une des formes
    textuelles de ``prune_codes.utils.ABSTENTION_SENTINELS``. Ne filtrer que les deux
    chaînes historiques laisserait ``"N/A"`` devenir un code.
    """
    table, _ = build_candidate_table(_merged(ragann_code=[abstention]))
    assert list(table["code_candidat"]) == ["01.1.2"]
    assert (table["vote_ragann"] == 0).all()


def test_malformed_code_is_not_a_candidate():
    """Un libellé renvoyé par erreur à la place d'un code est écarté."""
    table, diag = build_candidate_table(_merged(ragann_code=["pain de mie"]))
    assert list(table["code_candidat"]) == ["01.1.2"]
    assert diag["n_rejected_not_a_proposal"]["ragann_code"] == 1


def test_products_without_candidate_are_surfaced_not_dropped():
    """Un produit sans aucun candidat doit être signalé, jamais disparaître."""
    merged = _merged(
        id=["p1", "p2"],
        lcs_code=["01.1.2", None],
        lcs_distance=[0.2, None],
        rag_code=["01.1.2", "N/A"],
        rag_confidence=[0.8, None],
        ragann_code=["01.1.2", None],
        ragann_confidence=[0.9, None],
        ttc_code_1=["01.1.2", None],
        ttc_conf_1=[0.7, None],
        code_lvl4=["01.1.2", "05.1"],
    )
    table, diag = build_candidate_table(merged)
    assert diag["ids_without_candidate"] == ["p2"]
    assert "p2" not in set(table["id"])
    assert diag["n_products_in"] == 2
    assert diag["n_products_with_candidates"] == 1


def test_no_nan_in_features():
    """Le scorer et R refusent tous deux les NaN : ils doivent être impossibles."""
    table, _ = build_candidate_table(_merged())
    numeriques = [f for f in FEATURES if f != "code_candidat_n1"]
    assert not table[numeriques].isna().any().any()
    assert table["code_candidat_n1"].notna().all()


def test_division_is_first_segment():
    table, _ = build_candidate_table(_merged(ragann_code=["12.4.5.6"]))
    assert set(table["code_candidat_n1"]) == {"01", "12"}


def test_production_mode_has_no_target_column():
    """Sans vérité terrain, `correcte` est ABSENTE, pas remplie de NaN.

    Une colonne float toute-NaN passerait jusqu'à R et déclencherait son
    ``data.check`` — mieux vaut l'absence, qui est détectable.
    """
    table, diag = build_candidate_table(_merged(), truth_col=None)
    assert "correcte" not in table.columns
    assert diag["mode"] == "production"


def test_truth_never_taken_from_raw_code():
    """`code` (annotation brute niveau 5) ne doit jamais servir de cible.

    Comparer des candidats prunés à une vérité non prunée donnerait une accuracy
    proche de zéro. Seul `code_lvl4` est comparable.
    """
    merged = _merged()
    merged["code"] = ["01.1.2.9"]  # brut, plus profond
    table, _ = build_candidate_table(merged)
    gagnant = table[table["code_candidat"] == "01.1.2"].iloc[0]
    assert gagnant["correcte"] == 1  # comparé à code_lvl4, pas à code


def test_duplicate_ids_raise():
    """Un `id` dupliqué démultiplierait les lignes en silence."""
    merged = pd.concat([_merged(), _merged()], ignore_index=True)
    with pytest.raises(ValueError, match="dupliqué"):
        build_candidate_table(merged)


def test_missing_classifier_column_is_tolerated():
    """Un classifieur absent du run ⇒ vote 0 + sentinelle, pas un échec.

    Cas réel : un run antérieur à l'intégration de RAG-ANN.
    """
    merged = _merged().drop(columns=["ragann_code", "ragann_confidence"])
    table, _ = build_candidate_table(merged)
    assert (table["vote_ragann"] == 0).all()
    assert (table["conf_ragann"] == CONF_MISSING).all()


def test_dtypes_are_explicit():
    """Types figés : le CSV transmis à R doit porter les mêmes doubles."""
    table, _ = build_candidate_table(_merged())
    for col in ("vote_lcs", "vote_rag", "vote_ragann", "vote_ttc", "nb_votants"):
        assert table[col].dtype == np.int64
    for col in ("conf_rag", "conf_ragann", "conf_ttc", "dist_lcs"):
        assert table[col].dtype == np.float64


def test_output_is_deterministically_ordered():
    """Deux appels sur la même entrée donnent le même ordre de lignes."""
    a, _ = build_candidate_table(_merged())
    b, _ = build_candidate_table(_merged())
    pd.testing.assert_frame_equal(a, b)
    assert list(a["code_candidat"]) == sorted(a["code_candidat"])
