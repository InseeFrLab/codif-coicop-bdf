"""Tests des deux conventions d'accuracy et de la résolution de la vérité terrain.

Lancer depuis `report/` : `uv run --with pytest pytest test_coicop_metrics.py`
"""

import pandas as pd

from coicop_metrics import (
    CANONICAL_LEVELS,
    CONSENSUS_LABEL,
    REGIME_COL,
    TRUTH_COL_CANONICAL,
    TRUTH_COL_RAW,
    accuracy,
    accuracy_table,
    level_result,
    regime_accuracy_table,
    regime_masks,
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


class TestRegimes:
    @staticmethod
    def _frame():
        """Deux consensus (le juge reprend TTC top-1, juste dans les deux cas) et
        trois arbitrages : le juge casse deux bons codes TTC et en répare un.
        TTC arbitré = 2/3, LLM arbitré = 1/3, LLM d'ensemble = 3/5."""
        return pd.DataFrame(
            {
                TRUTH_COL_CANONICAL: ["01.3", "01.1.1.3", "02.1.1.1", "03.2", "04.1.1.1"],
                "ttc_code_1": ["01.3", "01.1.1.3", "02.1.1.1", "03.2", "04.1.1.9"],
                "llm_code": ["01.3", "01.1.1.3", "02.1.1.9", "03.9", "04.1.1.1"],
                REGIME_COL: [
                    CONSENSUS_LABEL,
                    CONSENSUS_LABEL,
                    "gemma4-26b-moe",
                    "gemma4-26b-moe",
                    "gemma4-26b-moe",
                ],
            }
        )

    def test_masks_split_consensus_from_arbitration(self):
        masks = regime_masks(self._frame())
        labels = [label for label, _suffix, _mask in masks]
        suffixes = [suffix for _label, suffix, _mask in masks]
        assert labels == ["Consensus", "Arbitré"]
        assert suffixes == ["consensus", "arbitrated"]
        assert [int(mask.sum()) for _l, _s, mask in masks] == [2, 3]

    def test_absent_regime_column_yields_none(self):
        """Runs antérieurs au tag `llm_model` : pas de découpage possible."""
        df = self._frame().drop(columns=[REGIME_COL])
        assert regime_masks(df) is None
        assert regime_accuracy_table(df) is None

    def test_table_reports_pooled_and_per_regime_counts(self):
        tbl = regime_accuracy_table(self._frame(), 4)
        assert list(tbl.columns) == [
            "Ensemble (n=5)",
            "Consensus (n=2)",
            "Arbitré (n=3)",
        ]

    def test_llm_equals_ttc_on_consensus_rows(self):
        """Le raccourci consensus retient TTC top-1 : sur ce sous-ensemble les deux
        colonnes sont tautologiquement égales, d'où l'intérêt du découpage."""
        tbl = regime_accuracy_table(self._frame(), 4)
        assert tbl.loc["LLM", "Consensus (n=2)"] == tbl.loc["TTC", "Consensus (n=2)"]
        assert tbl.loc["LLM", "Arbitré (n=3)"] != tbl.loc["TTC", "Arbitré (n=3)"]

    def test_pooled_figure_hides_the_arbitration_result(self):
        """Le chiffre d'ensemble est une moyenne des deux régimes : il surestime
        ce que le juge fait là où il décide vraiment."""
        tbl = regime_accuracy_table(self._frame(), 4)
        pooled = tbl.loc["LLM", "Ensemble (n=5)"]
        assert tbl.loc["LLM", "Arbitré (n=3)"] < pooled < tbl.loc["LLM", "Consensus (n=2)"]


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
