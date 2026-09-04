"""L'évaluation refuse de mesurer contre une vérité brute.

`code_lvl4` naît en un seul endroit du pipeline — `reconcile-llm`, et seulement
si `--mapping-file` lui est passé. L'oublier ne cassait rien : `truth_column`
se rabattait sur `code`, avec un avertissement, et tout le rapport sortait avec
une accuracy sous-estimée sur près d'un quart des postes.

Comparer des prédictions canoniques (tronquées niveau 4, hiérarchies linéaires
élaguées) à une vérité brute compte comme fausses des prédictions justes. Ce
repli est acceptable dans un rapport de production, qui ne mesure rien ; il ne
l'est pas quand quelqu'un a explicitement demandé une évaluation.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from main import require_canonical_truth  # noqa: E402


def test_accepts_the_canonical_truth():
    df = pd.DataFrame({"code_lvl4": ["0111"], "llm_code": ["0111"]})
    assert require_canonical_truth(df) == "code_lvl4"


def test_refuses_the_raw_truth():
    """Le cas qui passait en silence : `code` seul, sans `code_lvl4`."""
    df = pd.DataFrame({"code": ["01111"], "llm_code": ["0111"]})
    with pytest.raises(SystemExit) as e:
        require_canonical_truth(df)
    assert "code_lvl4" in str(e.value)


def test_the_message_names_the_probable_cause():
    """Un échec en fin de pipeline doit dire quoi corriger en amont, pas
    seulement ce qui manque."""
    df = pd.DataFrame({"code": ["01111"]})
    with pytest.raises(SystemExit) as e:
        require_canonical_truth(df)
    message = str(e.value)
    assert "mapping-file" in message
    assert "code" in message  # les colonnes présentes sont listées


def test_refuses_when_no_truth_at_all():
    """Run non étiqueté : l'étape ne devrait pas avoir été déclenchée."""
    df = pd.DataFrame({"llm_code": ["0111"]})
    with pytest.raises(SystemExit):
        require_canonical_truth(df)
