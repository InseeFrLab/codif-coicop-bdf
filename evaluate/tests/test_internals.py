"""Tests de la décomposition interne des classifieurs.

`test_duckdb_list_column` couvre un défaut qui n'aurait rien cassé : DuckDB rend
une colonne de listes en `numpy.ndarray`, et le test `isinstance(..., list)` la
rejetait — un classifieur perdait tout son retrieval, sans erreur, et son recall
tombait à zéro.

`test_accuracy_may_exceed_recall` fixe une limite d'interprétation : la consigne
faite au modèle de reprendre un candidat à l'identique est incitative, pas
structurelle. Le recall n'est donc pas un plafond, et ni le module ni le rapport
ne doivent le présenter comme tel.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import internals as I  # noqa: E402


def frame(truth, pred, conf=None):
    d = {"id": range(len(truth)), "code_lvl4": truth, "rag_code": pred}
    if conf is not None:
        d["rag_confidence"] = conf
    return pd.DataFrame(d)


class TestWidenRetrieved:
    def test_gathers_rank_columns_into_one_list(self):
        wide = pd.DataFrame({"id": [1, 2], "0": ["01.1", "02.1"], "1": ["01.2", "02.2"]})
        out = I.widen_retrieved(wide)
        assert out.loc[0, "list_retrieved_codes"] == ["01.1", "01.2"]

    def test_rank_order_is_numeric_not_lexicographic(self):
        """Avec 10 voisins ou plus, un tri alphabétique mettrait "10" avant "2"."""
        wide = pd.DataFrame({"id": [1], **{str(i): [f"c{i}"] for i in range(12)}})
        out = I.widen_retrieved(wide)
        assert out.loc[0, "list_retrieved_codes"] == [f"c{i}" for i in range(12)]

    def test_survives_a_frame_without_ranks(self):
        out = I.widen_retrieved(pd.DataFrame({"id": [1]}))
        assert list(out.columns) == ["id", "list_retrieved_codes"]
        assert len(out) == 0


class TestBuildRecords:
    def test_duckdb_list_column(self):
        """DuckDB rend une colonne de listes en `numpy.ndarray`. Un test
        `isinstance(..., list)` la rejetait : le classifieur perdait tout son
        retrieval, sans erreur, et le recall tombait à zéro."""
        scorable = frame(["01.1"], ["01.1"])
        retrieved = pd.DataFrame({
            "id": [0],
            "list_retrieved_codes": [np.array(["01.1", "02.3"], dtype=object)],
        })
        rec = I.build_records(scorable, "code_lvl4", "rag_code", retrieved=retrieved)[0]
        assert rec["list_retrieved_codes"] == ["01.1", "02.3"]

    def test_missing_retrieval_is_an_empty_list_not_nan(self):
        rec = I.build_records(frame(["01.1"], ["01.1"]), "code_lvl4", "rag_code")[0]
        assert rec["list_retrieved_codes"] == []

    def test_parsed_defaults_to_true(self):
        """TTC et LCS n'ont pas de notion de parsing : sans ce défaut,
        `confidence_reliability` les écarterait tous."""
        rec = I.build_records(frame(["01.1"], ["01.1"]), "code_lvl4", "rag_code")[0]
        assert rec["parsed"] is True or rec["parsed"] == True  # noqa: E712

    def test_flags_are_joined_on_id(self):
        scorable = frame(["01.1", "02.1"], ["01.1", "09.9"])
        flags = pd.DataFrame({"id": [1, 0], "parsed": [False, True], "codable": [True, False]})
        recs = I.build_records(scorable, "code_lvl4", "rag_code", flags=flags)
        assert [r["parsed"] for r in recs] == [True, False]
        assert [r["codable"] for r in recs] == [False, True]


class TestRetrieval:
    def test_decomposes_accuracy_into_recall_times_generation(self):
        """L'identité qui rend le tableau lisible : quand le modèle ne répond
        juste que sur les lignes où la vérité a été récupérée,
        accuracy = recall × accuracy-si-récupéré."""
        truth = [f"0{i % 4 + 1}.1.1" for i in range(40)]
        # juste uniquement quand récupéré (i pair), et une fois sur deux alors
        got = [i % 2 == 0 for i in range(40)]
        pred = [t if (got[i] and i % 4 == 0) else "09.9.9" for i, t in enumerate(truth)]
        retrieved = pd.DataFrame({
            "id": range(40),
            "list_retrieved_codes": [[t] if got[i] else ["09.9.9"]
                                     for i, t in enumerate(truth)],
        })
        recs = I.build_records(frame(truth, pred), "code_lvl4", "rag_code",
                               retrieved=retrieved)
        tbl = I.retrieval_table(I.hierarchical(recs))
        for k in I.LEVELS:
            recall = tbl.loc[k, "recall du retriever"]
            cond = tbl.loc[k, "accuracy si récupéré"]
            assert recall == pytest.approx(0.5), f"niveau {k}"
            assert tbl.loc[k, "accuracy"] == pytest.approx(recall * cond), f"niveau {k}"

    def test_accuracy_may_exceed_recall(self):
        """La consigne « reprendre un candidat à l'identique » est incitative,
        pas structurelle : un modèle peut sortir un code juste absent de la
        liste. Le recall n'est donc pas un plafond, et le tableau ne doit pas
        le présenter comme tel."""
        truth = ["01.1.1"] * 20
        retrieved = pd.DataFrame({"id": range(20),
                                  "list_retrieved_codes": [["09.9.9"]] * 20})
        recs = I.build_records(frame(truth, truth), "code_lvl4", "rag_code",
                               retrieved=retrieved)
        tbl = I.retrieval_table(I.hierarchical(recs))
        assert tbl.loc[4, "recall du retriever"] == 0.0
        assert tbl.loc[4, "accuracy"] == pytest.approx(1.0)

    def test_recall_never_increases_with_level(self):
        """Un préfixe court est plus facile à faire correspondre qu'un long."""
        truth = ["01.2.3.4"] * 10
        retrieved = pd.DataFrame({"id": range(10),
                                  "list_retrieved_codes": [["01.2.9.9"]] * 10})
        recs = I.build_records(frame(truth, truth), "code_lvl4", "rag_code",
                               retrieved=retrieved)
        tbl = I.retrieval_table(I.hierarchical(recs))
        recalls = [tbl.loc[k, "recall du retriever"] for k in I.LEVELS]
        assert recalls == sorted(recalls, reverse=True)
        assert recalls[1] == 1.0 and recalls[3] == 0.0


class TestRegimes:
    def test_population_shrinks_as_filters_tighten(self):
        n = 50
        scorable = frame([f"0{i % 3 + 1}.1" for i in range(n)],
                         [f"0{i % 3 + 1}.1" for i in range(n)],
                         conf=[0.9 if i < 10 else 0.1 for i in range(n)])
        flags = pd.DataFrame({"id": range(n),
                              "parsed": [i % 10 != 0 for i in range(n)],
                              "codable": [i % 7 != 0 for i in range(n)]})
        recs = I.build_records(scorable, "code_lvl4", "rag_code", "rag_confidence",
                               flags=flags)
        tbl = I.regime_table(I.hierarchical(recs))
        counts = list(tbl["n"])
        assert counts[0] == n
        assert counts == sorted(counts, reverse=True)
        assert tbl.loc["Toutes les lignes", "part"] == 1.0


class TestConfidence:
    def test_auroc_is_one_when_confidence_ranks_perfectly(self):
        truth = ["01.1"] * 20
        pred = ["01.1"] * 10 + ["09.9"] * 10
        conf = [0.9] * 10 + [0.1] * 10
        recs = I.build_records(frame(truth, pred, conf), "code_lvl4",
                               "rag_code", "rag_confidence")
        tbl = I.confidence_table({"RAG": recs})
        assert tbl.loc["RAG", "AUROC"] == pytest.approx(1.0)
        assert tbl.loc["RAG", "écart"] == pytest.approx(0.8)

    def test_classifier_without_confidence_is_absent(self):
        recs = I.build_records(frame(["01.1"], ["01.1"]), "code_lvl4", "rag_code")
        assert len(I.confidence_table({"LCS": recs})) == 0

    def test_threshold_sweep_coverage_decreases(self):
        conf = [i / 20 for i in range(20)]
        recs = I.build_records(frame(["01.1"] * 20, ["01.1"] * 20, conf),
                               "code_lvl4", "rag_code", "rag_confidence")
        sweep = I.threshold_sweep_table(recs)
        cov = list(sweep["couverture"])
        assert cov == sorted(cov, reverse=True)


class TestCodable:
    def test_lift_is_positive_when_the_flag_helps(self):
        truth = ["01.1"] * 20
        pred = ["01.1"] * 10 + ["09.9"] * 10
        scorable = frame(truth, pred)
        flags = pd.DataFrame({"id": range(20), "parsed": [True] * 20,
                              "codable": [True] * 10 + [False] * 10})
        recs = I.build_records(scorable, "code_lvl4", "rag_code", flags=flags)
        tbl = I.codable_table({"RAG": recs})
        assert tbl.loc["RAG", "lift"] > 0
        assert tbl.loc["RAG", "accuracy si codable"] == pytest.approx(1.0)

    def test_classifier_without_the_flag_is_absent(self):
        recs = I.build_records(frame(["01.1"], ["01.1"]), "code_lvl4", "rag_code")
        assert len(I.codable_table({"TTC": recs})) == 0


class TestFlattenInternal:
    def test_every_value_is_a_float_for_mlflow(self):
        """`mlflow.log_metrics` refuse un None ou un booléen : les écarter ici
        évite de faire échouer tout le log pour un indicateur manquant."""
        recs = I.build_records(frame(["01.1"] * 5, ["01.1"] * 5, [0.5] * 5),
                               "code_lvl4", "rag_code", "rag_confidence")
        flat = I.flatten_internal({"RAG": recs})
        assert flat
        assert all(isinstance(v, float) for v in flat.values())

    def test_names_are_namespaced_by_classifier(self):
        recs = I.build_records(frame(["01.1"] * 5, ["01.1"] * 5), "code_lvl4", "rag_code")
        flat = I.flatten_internal({"RAG-annot": recs})
        assert any(k.startswith("regime/rag_annot/") for k in flat)


class TestResponseFlags:
    def test_a_brick_without_a_codable_flag_is_recognised(self):
        """TTC et LCS n'ont pas de notion de codabilité : `build_records` leur
        pose `codable = None`. Le rapport s'en servait via une heuristique
        fausse (« tous les régimes ont le même effectif »), qui ne se
        déclenchait jamais — les régimes sont vides, pas identiques."""
        recs = I.build_records(frame(["01.1.1.1"], ["01.1.1.1"]), "code_lvl4", "rag_code")
        assert I.has_response_flags(recs) is False

    def test_a_brick_carrying_the_flag_is_recognised(self):
        flags = pd.DataFrame({"id": [0], "parsed": [True], "codable": [True]})
        recs = I.build_records(
            frame(["01.1.1.1"], ["01.1.1.1"]), "code_lvl4", "rag_code", flags=flags
        )
        assert I.has_response_flags(recs) is True

    def test_the_empty_regimes_are_what_the_old_guard_missed(self):
        """La preuve du défaut : sans drapeau, la colonne `n` vaut [N, N, 0, 0, 0]
        — deux valeurs distinctes, donc `nunique() == 1` était toujours faux."""
        recs = I.build_records(frame(["01.1.1.1"] * 5, ["01.1.1.1"] * 5),
                               "code_lvl4", "rag_code")
        counts = list(I.regime_table(I.hierarchical(recs))["n"])
        assert counts == [5, 5, 0, 0, 0]
        assert len(set(counts)) == 2

    def test_recall_is_empty_not_zero_without_a_retriever(self):
        """0 % se lirait « le retriever ne ramène rien » pour une brique qui n'a
        pas de retriever."""
        recs = I.build_records(frame(["01.1.1.1"] * 3, ["01.1.1.1"] * 3),
                               "code_lvl4", "rag_code")
        tbl = I.regime_table(I.hierarchical(recs), has_retrieval=False)
        assert tbl[f"recall niv{I.TARGET_LEVEL}"].isna().all()


class TestRetrievalSize:
    def test_reports_the_observed_range(self):
        """Mesurée sur les données du run, jamais lue dans `retrieval.size` des
        configs : c'est la valeur qui a servi à ce run-là qui compte."""
        retrieved = pd.DataFrame({"id": [0, 1], "0": ["01.1", "02.1"], "1": ["01.2", None]})
        recs = I.build_records(
            frame(["01.1.1.1", "02.1.1.1"], ["01.1.1.1", "02.1.1.1"]),
            "code_lvl4", "rag_code", retrieved=I.widen_retrieved(retrieved),
        )
        assert I.retrieval_size(recs) == (1, 2)

    def test_none_without_any_retrieval(self):
        recs = I.build_records(frame(["01.1.1.1"], ["01.1.1.1"]), "code_lvl4", "rag_code")
        assert I.retrieval_size(recs) is None


class TestEndToEnd:
    """Périmètre du chiffre métier.

    Le défaut corrigé ici : `export-results` construit le livrable sur toutes les
    observations, alors que l'échantillonnage a lieu en aval à `classify-regex`.
    Sur un run de 100 produits le tableau annonçait ~16 000 observations, avec
    une accuracy diluée par les lignes jamais codées — et se présentait comme
    « le chiffre métier ».
    """

    @staticmethod
    def _paths(monkeypatch, deliverable, observations):
        mapping = pd.DataFrame({"code": [], "code_parent_equivalent": []})
        frames = {"d": deliverable, "o": observations, "m": mapping}
        monkeypatch.setattr(I, "_read", lambda con, path: frames[path])
        return I.end_to_end(None, "d", "o", "m")

    @staticmethod
    def _deliverable():
        """Quatre lignes livrées, deux seulement décidées par ce run — la forme
        exacte d'un run échantillonné."""
        return pd.DataFrame(
            {
                "id": [1, 2, 3, 4],
                "predicted_code": ["01.1.1.1", "02.1.1.1", None, None],
                "prediction_source": ["regex", "llm", None, None],
            }
        )

    @staticmethod
    def _observations():
        return pd.DataFrame(
            {
                "id": [1, 2, 3, 4],
                "code": ["01.1.1.1", "02.1.1.9", "03.1.1.1", "04.1.1.1"],
            }
        )

    def test_uncodified_rows_leave_the_denominator(self, monkeypatch):
        out = self._paths(monkeypatch, self._deliverable(), self._observations())
        assert out["n_deliverable"] == 4
        assert out["n_decided"] == 2
        assert out["levels"].loc[4, "n évaluable"] == 2

    def test_regex_contribution_is_isolated(self, monkeypatch):
        """La demande « impact de l'étape regex » : la regex code juste,
        la conciliation se trompe, donc retirer la regex fait chuter le chiffre."""
        out = self._paths(monkeypatch, self._deliverable(), self._observations())
        row = out["levels"].loc[4]
        assert row["accuracy livrée"] == 0.5
        assert row["n hors regex"] == 1
        assert row["accuracy hors regex"] == 0.0
        assert row["apport regex"] == 0.5

    def test_breakdown_by_source(self, monkeypatch):
        out = self._paths(monkeypatch, self._deliverable(), self._observations())
        by_source = out["by_source"]
        assert set(by_source.index) == {"regex", "llm"}
        assert by_source.loc["regex", f"accuracy niv{I.TARGET_LEVEL}"] == 1.0
        assert by_source.loc["llm", f"accuracy niv{I.TARGET_LEVEL}"] == 0.0
        assert by_source["n livré"].sum() == out["n_scored"]

    def test_legacy_deliverable_without_the_source_column(self, monkeypatch):
        """Runs antérieurs à `prediction_source` : on retombe sur « décidée =
        porte un code », sans ventilation."""
        deliverable = self._deliverable().drop(columns=["prediction_source"])
        out = self._paths(monkeypatch, deliverable, self._observations())
        assert out["has_source"] is False
        assert out["by_source"] is None
        assert out["n_decided"] == 2
        assert "apport regex" not in out["levels"].columns
