"""Volumétrie du run : ce que le tableau de métadonnées doit dire.

Ces lignes vivent dans `internals.py` et non dans le gabarit Quarto justement
pour être testées ici : le `.qmd` fait plus de 1 500 lignes et n'en porte aucun
test, alors que toute la logique intéressante de cette section est
conditionnelle — artefact absent, écart inexpliqué, livrable qui ne coïncide pas
avec les observations.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import internals as I  # noqa: E402

COUNTS = {
    "run_id": "codif-abc12",
    "run_date": "2026-09-03",
    "input_file": "s3://bucket/data/workflow_inputs/tickets.csv",
    "n_input_rows": 1000,
    "n_dropped_empty_label": 30,
    "n_dropped_uncodable": 70,
    "n_observations": 900,
    "n_labelled": 800,
}


def _labels(rows):
    return [label for label, _value in rows]


def _value(rows, needle):
    return next(v for label, v in rows if needle in label)


class TestInputCounts:
    def test_reads_the_single_row_as_a_dict(self, monkeypatch):
        monkeypatch.setattr(I, "_read", lambda con, path: pd.DataFrame([COUNTS]))
        assert I.input_counts(None, "peu importe")["n_input_rows"] == 1000

    def test_a_missing_artefact_is_not_an_error(self, monkeypatch):
        """Un run antérieur à cet artefact, ou lancé sans `--input-file`, doit
        rendre un rapport lisible — pas faire échouer l'étape."""
        monkeypatch.setattr(I, "_read", lambda con, path: None)
        assert I.input_counts(None, "absent") is None

    def test_an_empty_frame_is_treated_as_absent(self, monkeypatch):
        monkeypatch.setattr(I, "_read", lambda con, path: pd.DataFrame(columns=["n_input_rows"]))
        assert I.input_counts(None, "vide") is None

    def test_a_null_label_count_comes_back_as_none(self, monkeypatch):
        """`n_labelled` est nullable : pandas le relit en NaN, et l'appelant ne
        doit avoir qu'une seule chose à tester."""
        monkeypatch.setattr(
            I, "_read", lambda con, path: pd.DataFrame([{**COUNTS, "n_labelled": None}])
        )
        assert I.input_counts(None, "sans étiquettes")["n_labelled"] is None


class TestRunMetadataRows:
    def test_shows_the_input_output_gap_and_its_causes(self):
        rows = I.run_metadata_rows(COUNTS, n_deliverable=900)
        assert "1 000" in _value(rows, "Lignes du fichier d'entrée")
        assert _value(rows, "libellé vide") == "30"
        assert _value(rows, "produit non codable") == "70"
        assert "900" in _value(rows, "Lignes retenues")
        assert _value(rows, "étiquette") == "800"

    def test_stays_silent_when_the_identity_closes(self):
        """La ligne « cause non identifiée » ne doit apparaître que si elle a
        quelque chose à dire — sinon elle inquiète pour rien."""
        assert not any("non identifiée" in lbl for lbl in _labels(
            I.run_metadata_rows(COUNTS, n_deliverable=900)
        ))

    def test_surfaces_an_unexplained_gap(self):
        """Le jour où quelqu'un ajoute un filtre dans `build-datasets` sans
        toucher au compteur, le rapport le dit au lieu d'afficher des chiffres
        qui ne s'additionnent pas."""
        counts = {**COUNTS, "n_observations": 850}
        rows = I.run_metadata_rows(counts, n_deliverable=850)
        assert "50" in _value(rows, "non identifiée")

    def test_flags_a_deliverable_that_does_not_match(self):
        """`export-results` part de toutes les observations et fusionne en
        `left` : les deux nombres sont égaux par construction. S'ils divergent,
        une jointure duplique — c'est une régression, pas une information de
        volumétrie."""
        rows = I.run_metadata_rows(COUNTS, n_deliverable=912)
        assert "912" in _value(rows, "réellement dans le livrable")

    def test_omits_the_label_line_when_the_run_has_no_ground_truth(self):
        rows = I.run_metadata_rows({**COUNTS, "n_labelled": None}, n_deliverable=900)
        assert not any("étiquette" in lbl for lbl in _labels(rows))

    def test_degrades_to_a_single_explicit_line_without_the_artefact(self):
        rows = I.run_metadata_rows(None)
        assert len(rows) == 1
        assert "décompte absent" in rows[0][1]
