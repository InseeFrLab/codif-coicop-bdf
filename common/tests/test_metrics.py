"""Tests de la convention d'accuracy et de la résolution de la vérité terrain.

Lancer depuis `common/` : `uv run pytest tests/test_metrics.py`
"""

import json

import pandas as pd
import pytest

from codif_common.metrics import (
    CANONICAL_LEVELS,
    CONSENSUS_LABEL,
    REGIME_COL,
    TRUTH_COL_CANONICAL,
    TRUTH_COL_RAW,
    accuracy,
    accuracy_series,
    accuracy_table,
    answer_mask,
    coverage_table,
    declared_refusal_table,
    is_answer,
    level_result,
    parse_step_timings,
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


class TestTruncateAndCompare:
    """La règle unique : tronquer les deux codes à `k`, comparer. C'est tout."""

    def test_shallow_truth_is_still_scored(self):
        """Une vérité de profondeur 2 n'est plus écartée : tronquée à elle-même,
        elle est mesurée — et juste si la prédiction lui est égale. C'est le
        gain du changement, un Poste peu profond bien codé compte."""
        for k in [2, 3, 4]:
            assert level_result("01.3", "01.3", k) is True

    def test_a_prediction_finer_than_the_truth_is_wrong_below_its_depth(self):
        """Le cas qui a motivé la règle, et ce qu'elle coûte : `01.4.3.1` face à
        une vérité `01.4` est juste tant qu'on ne descend pas sous la profondeur
        de la vérité, faux ensuite."""
        assert level_result("01.4", "01.4.3.1", 1) is True
        assert level_result("01.4", "01.4.3.1", 2) is True
        assert level_result("01.4", "01.4.3.1", 3) is False
        assert level_result("01.4", "01.4.3.1", 4) is False

    def test_shorter_prediction_is_an_error(self):
        assert level_result("01.1.1.3", "01.1.1", 4) is False

    def test_deeper_prediction_matching_prefix_is_correct(self):
        """Tant que la vérité atteint `k`, une prédiction plus fine reste juste :
        la troncature des DEUX côtés les égalise. Ce n'est pas une survivance de
        la règle du préfixe — au-delà de la profondeur de la vérité, le test
        ci-dessus montre qu'elle devient fausse."""
        assert level_result("01.1.1", "01.1.1.3", 3) is True

    def test_missing_truth_is_unscorable(self):
        """Le SEUL cas de None désormais : pas de vérité, donc rien à juger."""
        for absent in [None, "", "   ", float("nan")]:
            assert level_result(absent, "01.3", 4) is None

    def test_empty_truth_and_empty_prediction_are_not_a_match(self):
        """La garde `if not tp` prise de face : sans elle, `[] == []` vaudrait
        True et « pas de vérité + pas de prédiction » compterait comme un
        succès. C'est la seule subtilité du nouveau code."""
        assert level_result(None, None, 4) is None
        assert level_result("", "", 4) is None

    def test_level_5_duplicates_level_4_on_canonical_codes(self):
        """Les codes canoniques ont au plus 4 segments : tronquer à 5 ne tronque
        rien, niv5 reproduirait niv4. C'est ce qui justifie
        STRICT_LEVELS = CANONICAL_LEVELS dans le rapport."""
        for truth in ["01.3", "01.1.1.3"]:
            assert level_result(truth, truth, 5) is level_result(truth, truth, 4)
            assert level_result(truth, truth, 5) is True

    def test_level_5_would_measure_prediction_depth_not_correctness(self):
        """La nuance, et la seconde raison de ne pas publier le niveau 5 : face à
        une prédiction à 5 segments, niv5 la compte fausse alors que niv4 la
        compte juste. Il mesurerait la profondeur, pas l'exactitude."""
        assert level_result("01.1.1.3", "01.1.1.3.2", 4) is True
        assert level_result("01.1.1.3", "01.1.1.3.2", 5) is False

    def test_a_correct_row_is_always_an_answered_row(self):
        """Le lemme sur lequel repose l'identité de `coverage_table` : toute
        forme d'abstention compte moins de k segments, donc est une erreur —
        jamais une exclusion."""
        for sentinel in [None, "", "   ", "N/A", "none", "-"]:
            for k in CANONICAL_LEVELS:
                assert level_result("01.1.1.3", sentinel, k) is not True


class TestDenominators:
    @staticmethod
    def _frame():
        return pd.DataFrame(
            {
                TRUTH_COL_CANONICAL: ["01.3", "01.3", "01.1.1.3", "01.1.1.3", None],
                "llm_code": ["01.3", "01.4", "01.1.1.3", "01.1.1", "01.2"],
            }
        )

    def test_denominator_does_not_move_with_depth(self):
        """Le `n` ne dépend plus de `k` : c'est le nombre de lignes portant une
        vérité (4 sur 5 ici, la dernière n'en a pas). Les colonnes du rapport
        deviennent donc comparables entre elles."""
        df = self._frame()
        _, n2, _ = accuracy(df[TRUTH_COL_CANONICAL], df["llm_code"], 2)
        _, n4, _ = accuracy(df[TRUTH_COL_CANONICAL], df["llm_code"], 4)
        assert n2 == n4 == 4

    def test_shallow_truth_stays_in_the_denominator(self):
        df = self._frame()
        # Niveau 4, les quatre lignes étiquetées : `01.3`/`01.3` juste (les deux
        # tronquées valent elles-mêmes), `01.3`/`01.4` faux, `01.1.1.3` juste,
        # `01.1.1.3`/`01.1.1` faux (prédiction trop courte) => 2/4.
        # Sous l'ancienne convention les deux lignes `01.3` sortaient et le
        # résultat était (1, 2, 0.5) : même accuracy, population deux fois plus
        # petite.
        assert accuracy(df[TRUTH_COL_CANONICAL], df["llm_code"], 4) == (2, 4, 0.5)


class TestTables:
    def test_accuracy_table_uses_canonical_truth(self):
        """Scorer contre l'annotation brute compte fausse une prédiction
        canonique correcte : au niveau 3 la vérité canonique `01.3` égale la
        prédiction `01.3`, tandis que la brute `01.3.0.0.1` tronquée donne
        `01.3.0` et la juge fausse."""
        df = pd.DataFrame(
            {
                TRUTH_COL_RAW: ["01.3.0.0.1"],
                TRUTH_COL_CANONICAL: ["01.3"],
                "llm_code": ["01.3"],
            }
        )
        assert accuracy_table(df, levels=[3]).loc["LLM", "niv3"] == 1.0
        legacy = accuracy_table(df.drop(columns=[TRUTH_COL_CANONICAL]), levels=[3])
        assert legacy.loc["LLM", "niv3"] == 0.0

    def test_table_columns_are_one_per_level(self):
        """Plus de `(n=…)` dans les en-têtes : le `n` est le même partout, le
        répéter inviterait à lire des populations différentes."""
        df = pd.DataFrame({TRUTH_COL_CANONICAL: ["01.3"], "llm_code": ["01.3"]})
        assert list(accuracy_table(df).columns) == ["niv1", "niv2", "niv3", "niv4", "niv5"]


class TestAbstention:
    def test_recognises_the_forms_a_refusal_takes(self):
        for refusal in [None, float("nan"), "", "   ", "N/A", "n/a", "None", "null", "-"]:
            assert is_answer(refusal) is False, refusal
        for code in ["01.3", "01.1.1.3", " 02.1 "]:
            assert is_answer(code) is True, code

    def test_answer_mask_follows_the_series(self):
        pred = pd.Series(["01.3", None, "N/A", "02.1"])
        assert list(answer_mask(pred)) == [True, False, False, True]

    @staticmethod
    def _frame():
        """Quatre observations évaluables au niveau 4 : deux codes justes, un code
        faux, une abstention.

        Les quatre vérités atteignent la profondeur 4 délibérément : sur une
        vérité plus courte, une prédiction à 4 segments serait comptée fausse et
        le tableau mesurerait la profondeur des annotations autant que la
        couverture. Ici il ne mesure que la couverture.
        """
        return pd.DataFrame(
            {
                TRUTH_COL_CANONICAL: ["01.3.1.1", "01.1.1.3", "02.1.1.1", "03.2.1.1"],
                "llm_code": ["01.3.1.1", "01.1.1.3", "02.1.1.9", None],
            }
        )

    def test_global_accuracy_factorises_into_coverage_times_answered(self):
        """L'identité qui justifie le tableau : une abstention est une erreur
        (aucun code ne fait 4 segments), donc globale = couverture × sur
        réponses — à condition que les trois grandeurs partagent le dénominateur
        des lignes portant une vérité."""
        tbl = coverage_table(self._frame(), 4)
        row = tbl.loc["LLM"]
        assert row["n"] == 4
        assert row["couverture"] == 0.75
        assert row["abstentions"] == 1
        assert row["accuracy niv4 sur réponses"] == 2 / 3
        assert row["accuracy niv4 globale"] == 0.5
        assert row["accuracy niv4 globale"] == (
            row["couverture"] * row["accuracy niv4 sur réponses"]
        )

    def test_sentinel_string_is_an_abstention_not_a_wrong_code(self):
        """Sans le traitement des sentinelles, "N/A" gonflerait le dénominateur
        des réponses et écraserait l'accuracy sur réponses."""
        df = self._frame()
        df.loc[3, "llm_code"] = "N/A"
        tbl = coverage_table(df, 4)
        assert tbl.loc["LLM", "couverture"] == 0.75
        assert tbl.loc["LLM", "accuracy niv4 sur réponses"] == 2 / 3
        # La sentinelle reste une ERREUR dans le chiffre global, et non une
        # exclusion : c'est ce qui rend l'identité exacte.
        assert tbl.loc["LLM", "accuracy niv4 globale"] == 0.5

    def test_declared_flag_is_crossed_with_the_emitted_code(self):
        # Vérité de profondeur 4 : avec une vérité plus courte, les codes émis à
        # 4 segments seraient tous comptés faux et le test mesurerait la
        # profondeur des annotations au lieu du croisement drapeau × sortie.
        df = pd.DataFrame(
            {
                TRUTH_COL_CANONICAL: ["01.3.1.1"] * 4,
                "ragann_code": ["01.3.1.1", None, "01.3.1.1", None],
                "ragann_codable": [True, True, False, False],
            }
        )
        tbl = declared_refusal_table(df, 4)
        assert set(tbl["sortie"]) == {"code émis", "abstention"}
        assert tbl["n"].sum() == 4
        # Le cas contradictoire (déclaré non codable mais code émis) est isolé.
        contradiction = tbl[
            (tbl["drapeau"] == "ragann_codable = non codable")
            & (tbl["sortie"] == "code émis")
        ]
        assert int(contradiction["n"].iloc[0]) == 1
        # `n` compte la cellule, `n avec vérité` le dénominateur de l'accuracy.
        emitted = tbl[
            (tbl["drapeau"] == "ragann_codable = codable") & (tbl["sortie"] == "code émis")
        ]
        assert int(emitted["n avec vérité"].iloc[0]) == 1
        assert float(emitted["accuracy niv4"].iloc[0]) == 1.0

    def test_no_flag_column_yields_none(self):
        df = pd.DataFrame({TRUTH_COL_CANONICAL: ["01.3"], "llm_code": ["01.3"]})
        assert declared_refusal_table(df) is None


class TestCoverageDecomposition:
    """Le dénominateur partagé de `coverage_table`.

    Ces tests existent parce que l'erreur est silencieuse : rapporter la
    couverture à `len(data)` plutôt qu'aux lignes portant une vérité laisse le
    tableau se rendre normalement, avec un produit qui ne retombe plus sur
    l'accuracy globale.
    """

    def test_identity_holds_at_every_level(self):
        # Une ligne SANS vérité en plus des quatre étiquetées : sans elle le
        # dénominateur vaudrait trivialement `len(data)` et le test ne mordrait
        # plus sur ce qu'il prétend vérifier.
        df = pd.concat(
            [
                TestAbstention._frame(),
                pd.DataFrame({TRUTH_COL_CANONICAL: [None], "llm_code": ["09.9.9.9"]}),
            ],
            ignore_index=True,
        )
        for k in CANONICAL_LEVELS:
            row = coverage_table(df, k).loc["LLM"]
            if not row["couverture"]:
                continue
            assert row[f"accuracy niv{k} globale"] == pytest.approx(
                row["couverture"] * row[f"accuracy niv{k} sur réponses"]
            ), k

    def test_shallow_truth_keeps_its_abstention(self):
        """Le piège s'est inversé. Une abstention sur une ligne dont la vérité
        est peu profonde était retirée du calcul avec sa ligne, ce qui gonflait
        la couverture affichée ; elle y reste désormais. La couverture est celle
        de toutes les lignes étiquetées."""
        df = pd.DataFrame(
            {
                TRUTH_COL_CANONICAL: ["01.1.1.3", "03.2"],
                "llm_code": ["01.1.1.3", None],
            }
        )
        row = coverage_table(df, 4).loc["LLM"]
        assert row["n"] == 2
        assert row["abstentions"] == 1
        assert row["couverture"] == 0.5

    def test_accuracy_series_keeps_the_unscorable_rows_apart(self):
        """`accuracy_series` renvoie None, pas False, sur une ligne SANS VÉRITÉ.
        Un appelant qui écrit `serie == True` écrase ces None en False et compte
        comme fausses des lignes qu'on ne peut pas juger : c'est exactement ce
        que `coverage_table` évite en filtrant d'abord."""
        res = accuracy_series(
            pd.Series(["01.1.1.3", None]), pd.Series(["01.1.1.3", "03.2"]), 4
        )
        assert list(res) == [True, None]
        assert list(res == True) == [True, False]  # noqa: E712
        assert list(res[res.notna()] == True) == [True]  # noqa: E712


class TestRegimes:
    @staticmethod
    def _frame():
        """Deux consensus (le juge reprend TTC top-1, juste dans les deux cas) et
        trois arbitrages : le juge casse deux bons codes TTC et en répare un.
        TTC arbitré = 2/3, LLM arbitré = 1/3, LLM d'ensemble = 3/5.

        Les cinq vérités atteignent la profondeur 4 : sur une vérité plus courte,
        les codes TTC à 4 segments seraient comptés faux au niveau 4 et les
        effectifs annoncés ci-dessus ne tiendraient plus. Le test mesurerait la
        profondeur des annotations au lieu du partage consensus / arbitrage.
        """
        return pd.DataFrame(
            {
                TRUTH_COL_CANONICAL: [
                    "01.3.1.1", "01.1.1.3", "02.1.1.1", "03.2.1.1", "04.1.1.1",
                ],
                "ttc_code_1": [
                    "01.3.1.1", "01.1.1.3", "02.1.1.1", "03.2.1.1", "04.1.1.9",
                ],
                "llm_code": [
                    "01.3.1.1", "01.1.1.3", "02.1.1.9", "03.9.1.1", "04.1.1.1",
                ],
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


# --- Horodatages d'une tâche skippée ----------------------------------------
# Depuis l'introduction du paramètre `reconciliation`, reconcile-llm et
# reconcile-sirus sont exclusifs : l'un des deux est toujours Skipped, et Argo
# résout ses horodatages en zéro temps.


def test_skipped_task_timestamps_are_ignored():
    raw = json.dumps(
        {
            "build-datasets": ["2026-08-28T10:00:00Z", "2026-08-28T10:01:00Z"],
            "reconcile-llm": ["0001-01-01T00:00:00Z", "0001-01-01T00:00:00Z"],
            "reconcile-sirus": ["2026-08-28T10:02:00Z", "2026-08-28T10:03:00Z"],
        }
    )
    out = parse_step_timings(raw)
    assert "duration_reconcile_llm_seconds" not in out
    assert out["duration_reconcile_sirus_seconds"] == 60.0
    # Le total retombe sur la conciliation qui a réellement tourné.
    assert out["codification_total_seconds"] == 180.0


def test_total_uses_llm_conciliation_when_it_ran():
    raw = json.dumps(
        {
            "build-datasets": ["2026-08-28T10:00:00Z", "2026-08-28T10:01:00Z"],
            "reconcile-llm": ["2026-08-28T10:02:00Z", "2026-08-28T10:05:00Z"],
            "reconcile-sirus": ["0001-01-01T00:00:00Z", "0001-01-01T00:00:00Z"],
        }
    )
    out = parse_step_timings(raw)
    assert out["codification_total_seconds"] == 300.0
    assert "duration_reconcile_sirus_seconds" not in out
