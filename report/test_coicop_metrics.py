"""Tests des deux conventions d'accuracy et de la résolution de la vérité terrain.

Lancer depuis `report/` : `uv run --with pytest pytest test_coicop_metrics.py`
"""

import pandas as pd

from coicop_metrics import (
    CANONICAL_LEVELS,
    TRUTH_COL_CANONICAL,
    TRUTH_COL_RAW,
    accuracy,
    accuracy_table,
    level_result,
    truth_column,
    truth_depth_distribution,
)


class TestTruthColumn:
    def test_prefers_canonical_when_present(self):
        df = pd.DataFrame({TRUTH_COL_RAW: ["01.1.1.1.2"], TRUTH_COL_CANONICAL: ["01.1.1.1"]})
        assert truth_column(df) == TRUTH_COL_CANONICAL

    def test_falls_back_to_raw_for_legacy_runs(self):
        df = pd.DataFrame({TRUTH_COL_RAW: ["01.1.1.1.2"]})
        assert truth_column(df) == TRUTH_COL_RAW


class TestStrictConvention:
    def test_shallow_truth_is_excluded(self):
        """Vérité de profondeur 2 : non évaluable aux niveaux 3 et 4."""
        assert level_result("01.3", "01.3", 2) is True
        assert level_result("01.3", "01.3", 3) is None
        assert level_result("01.3", "01.3", 4) is None

    def test_shorter_prediction_is_an_error(self):
        assert level_result("01.1.1.3", "01.1.1", 4) is False

    def test_deeper_prediction_matching_prefix_is_correct(self):
        assert level_result("01.1.1", "01.1.1.3", 3) is True


class TestInclusiveConvention:
    def test_shallow_truth_is_scored_at_every_level(self):
        """C'est la différence de fond : la ligne compte à tous les niveaux."""
        for k in CANONICAL_LEVELS:
            assert level_result("01.3", "01.3", k, inclusive=True) is True

    def test_prediction_deeper_than_canonical_truth_is_wrong(self):
        """Dans l'espace pruné, `01.3.0.1` n'existe pas si la vérité canonique
        est `01.3` : la prédiction désigne un code inexistant."""
        assert level_result("01.3", "01.3.0.1", 4, inclusive=True) is False
        # ... mais elle reste juste aux niveaux où les préfixes coïncident
        assert level_result("01.3", "01.3.0.1", 2, inclusive=True) is True

    def test_missing_prediction_is_an_error_not_an_exclusion(self):
        assert level_result("01.3", None, 4, inclusive=True) is False

    def test_missing_truth_is_unscorable(self):
        assert level_result(None, "01.3", 4, inclusive=True) is None

    def test_saturates_at_level_4(self):
        """Les codes canoniques ont au plus 4 segments : k=4 compare les codes
        en entier, donc k=5 donnerait le même résultat."""
        for truth, pred in [("01.3", "01.3"), ("01.1.1.3", "01.1.1"), ("01.1.1.3", "02.1")]:
            assert level_result(truth, pred, 4, inclusive=True) == level_result(
                truth, pred, 5, inclusive=True
            )


class TestDenominators:
    @staticmethod
    def _frame():
        return pd.DataFrame(
            {
                TRUTH_COL_CANONICAL: ["01.3", "01.3", "01.1.1.3", "01.1.1.3", None],
                "llm_code": ["01.3", "01.4", "01.1.1.3", "01.1.1", "01.2"],
            }
        )

    def test_strict_denominator_shrinks_with_depth(self):
        df = self._frame()
        _, n2, _ = accuracy(df[TRUTH_COL_CANONICAL], df["llm_code"], 2)
        _, n4, _ = accuracy(df[TRUTH_COL_CANONICAL], df["llm_code"], 4)
        assert n2 == 4 and n4 == 2

    def test_inclusive_denominator_is_constant(self):
        df = self._frame()
        counts = {
            k: accuracy(df[TRUTH_COL_CANONICAL], df["llm_code"], k, inclusive=True)[1]
            for k in CANONICAL_LEVELS
        }
        assert set(counts.values()) == {4}, counts

    def test_inclusive_accuracy_counts_shallow_rows(self):
        df = self._frame()
        # niveau 4 : 01.3→01.3 juste, 01.3→01.4 faux, 01.1.1.3→01.1.1.3 juste,
        # 01.1.1.3→01.1.1 faux  =>  2/4
        n_ok, n_all, acc = accuracy(
            df[TRUTH_COL_CANONICAL], df["llm_code"], 4, inclusive=True
        )
        assert (n_ok, n_all) == (2, 4)
        assert acc == 0.5


class TestTables:
    def test_accuracy_table_uses_canonical_truth(self):
        """Une prédiction canonique correcte face à une annotation brute de
        niveau 5 ne doit pas être comptée fausse."""
        df = pd.DataFrame(
            {
                TRUTH_COL_RAW: ["01.3.0.0.1"],
                TRUTH_COL_CANONICAL: ["01.3"],
                "llm_code": ["01.3"],
            }
        )
        tbl = accuracy_table(df, inclusive=True)
        assert tbl.loc["LLM", "niv4"] == 1.0

    def test_inclusive_table_stops_at_level_4(self):
        df = pd.DataFrame({TRUTH_COL_CANONICAL: ["01.3"], "llm_code": ["01.3"]})
        assert list(accuracy_table(df, inclusive=True).columns) == [
            f"niv{k}" for k in CANONICAL_LEVELS
        ]

    def test_strict_table_reports_per_level_counts(self):
        df = pd.DataFrame({TRUTH_COL_CANONICAL: ["01.3"], "llm_code": ["01.3"]})
        cols = list(accuracy_table(df).columns)
        assert cols[0] == "niv1 (n=1)"
        assert cols[3] == "niv4 (n=0)"


class TestTruthDepth:
    def test_counts_rows_shallower_than_each_level(self):
        truth = pd.Series(["01", "01.3", "01.1.1", "01.1.1.3", None])
        d = truth_depth_distribution(truth)
        assert d["total"] == 4
        assert d["per_depth"][1]["count"] == 1
        assert {k: v["count"] for k, v in d["shallower_than"].items()} == {
            1: 0,
            2: 1,
            3: 2,
            4: 3,
        }
