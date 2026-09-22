"""Volumétrie du run : l'identité du fichier d'entrée, et l'entonnoir des volumes.

Ces lignes vivent dans `internals.py` et non dans le gabarit Quarto justement
pour être testées ici : le `.qmd` fait plus de 1 500 lignes et n'en porte aucun
test, alors que toute la logique intéressante de ces deux tableaux est
conditionnelle — artefact absent, run échantillonné, écart inexpliqué, livrable
qui ne coïncide pas avec les observations.
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
    return [r[0] for r in rows]


def _value(rows, needle):
    return next(r[1] for r in rows if needle in r[0])


def _part(rows, needle):
    return next(r[2] for r in rows if needle in r[0])


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
    def test_carries_the_input_file_path(self):
        rows = I.run_metadata_rows(COUNTS)
        assert COUNTS["input_file"] in rows[0][1]

    def test_degrades_to_a_single_explicit_line_without_the_artefact(self):
        rows = I.run_metadata_rows(None)
        assert len(rows) == 1
        assert "décompte absent" in rows[0][1]


class TestFunnelRows:
    """L'entonnoir : écartées d'emblée / captées par la regex / entrées dans la
    chaîne. Ces trois volumes vivaient dans trois endroits du pipeline et nulle
    part ensemble — c'est toute la raison d'être de ce tableau."""

    def _rows(self, **kw):
        base = dict(n_regex=150, n_chain=750, n_conciliation=750, n_scorable=740,
                    n_deliverable=900)
        return I.funnel_rows(COUNTS, **{**base, **kw})

    def test_separates_the_three_populations(self):
        rows = self._rows()
        assert _value(rows, "produit non codable") == "70"
        assert _value(rows, "codées par la regex") == "150"
        assert _value(rows, "entrées dans la chaîne") == "750"

    def test_shares_are_relative_to_the_input_file(self):
        """150 lignes de regex sur 1 000 en entrée, pas sur 900 retenues : la
        part doit se lire par rapport à ce qu'on a donné au pipeline."""
        assert _part(self._rows(), "codées par la regex") == "15.0%"

    def test_surfaces_what_sampling_removed(self):
        """Ce qui n'est ni capté par la regex ni entré dans la chaîne a été
        retiré par --sample-observations. Sans cette ligne, l'entonnoir ne
        s'additionne pas et le lecteur cherche l'erreur."""
        rows = self._rows(n_chain=200)
        assert _value(rows, "échantillonnage") == "550"

    def test_stays_silent_when_nothing_was_sampled(self):
        assert not any("échantillonnage" in lbl for lbl in _labels(self._rows()))

    def test_still_works_on_a_run_without_the_input_count(self):
        """Run antérieur à input_counts : l'entonnoir démarre aux observations
        retenues et les parts s'y rapportent, au lieu de disparaître."""
        rows = I.funnel_rows(None, n_regex=150, n_chain=750, n_deliverable=900)
        assert not any("Fichier d'entrée" in lbl for lbl in _labels(rows))
        assert _value(rows, "Observations retenues") == "**900**"
        assert _part(rows, "codées par la regex") == "16.7%"

    def test_returns_none_when_no_volume_is_known_at_all(self):
        assert I.funnel_rows(None) is None

    def test_omits_the_regex_split_when_its_artefacts_are_missing(self):
        rows = I.funnel_rows(COUNTS, n_regex=None, n_chain=None)
        assert not any("regex" in lbl for lbl in _labels(rows))
        assert _value(rows, "Observations retenues") == "**900**"

    def test_flags_a_deliverable_that_does_not_match(self):
        """`export-results` part de toutes les observations et fusionne en
        `left` : les deux nombres sont égaux par construction. S'ils divergent,
        une jointure duplique — c'est une régression, pas de la volumétrie."""
        rows = self._rows(n_deliverable=912)
        assert "912" in _value(rows, "réellement dans le livrable")

    def test_surfaces_an_unexplained_gap(self):
        """Le jour où quelqu'un ajoute un filtre dans `build-datasets` sans
        toucher au compteur, l'entonnoir le dit au lieu d'afficher des chiffres
        qui ne s'additionnent pas."""
        rows = I.funnel_rows({**COUNTS, "n_observations": 850}, n_deliverable=850)
        assert "50" in _value(rows, "non identifiée")
