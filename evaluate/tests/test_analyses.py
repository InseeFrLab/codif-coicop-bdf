"""Ventilation par provenance, et distorsion de distribution au niveau 1.

Ces deux analyses répondent à des questions que l'accuracy ne pose pas :
« quel mode de saisie coûte de la qualité ? » et « la répartition par division
est-elle juste, même quand les codifications individuelles ne le sont pas ? ».

Le point le plus important est testé en bas : distorsion et accuracy sont
INDÉPENDANTES. Un jeu entièrement faux peut avoir une distribution parfaite.
C'est toute la raison d'être du chapitre, et c'est ce qu'un lecteur pressé
refusera de croire tant qu'un test ne l'aura pas écrit.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import internals as I  # noqa: E402

TRUTH, FINAL = "code_lvl4", "llm_code"


class TestGroupSummary:
    @staticmethod
    def _frame():
        """Deux provenances : le ticket est bon partout ; le carnet trouve la
        bonne division trois fois sur quatre mais le bon poste une seule, et
        s'abstient une fois. C'est la forme d'une saisie pauvre — le libellé
        suffit à situer le rayon, pas à choisir dans le rayon."""
        return pd.DataFrame(
            {
                "source": ["ticket"] * 4 + ["carnet"] * 4,
                TRUTH: ["01.1.1.1"] * 8,
                FINAL: ["01.1.1.1"] * 4 + ["01.1.1.1", "01.2.1.1", "01.3.1.1", None],
            }
        )

    def _row(self, source):
        out = I.group_summary(self._frame(), "source", truth_col=TRUTH, final_col=FINAL)
        return out.set_index("source").loc[source]

    def test_separates_the_provenances(self):
        assert self._row("ticket")["accuracy niv4"] == 1.0
        assert self._row("carnet")["accuracy niv4"] == 0.25

    def test_coverage_is_reported_next_to_accuracy(self):
        """Une source peut faire chuter l'accuracy en produisant des libellés que
        la chaîne refuse de coder : ce n'est pas le même défaut que de les coder
        faux, et les deux colonnes doivent permettre de les distinguer."""
        assert self._row("ticket")["couverture"] == 1.0
        assert self._row("carnet")["couverture"] == 0.75

    def test_the_gap_is_measured_against_the_whole(self):
        """Ensemble = 5/8 = 62,5 %. Le ticket est donc à +37,5 points."""
        assert self._row("ticket")["écart / ensemble"] == 0.375
        assert self._row("carnet")["écart / ensemble"] == -0.375

    def test_shallower_levels_are_reported_too(self):
        """Une saisie pauvre dégrade souvent le niveau 4 en laissant le niveau 1
        intact : on sait que c'est de l'alimentaire, pas quel poste."""
        assert self._row("carnet")["accuracy niv1"] == 0.75
        assert self._row("carnet")["accuracy niv4"] == 0.25

    def test_a_missing_column_yields_none(self):
        assert I.group_summary(
            self._frame(), "provenance", truth_col=TRUTH, final_col=FINAL
        ) is None

    def test_an_empty_column_yields_none(self):
        frame = self._frame().assign(source=None)
        assert I.group_summary(frame, "source", truth_col=TRUTH, final_col=FINAL) is None


class TestDistortionLevel1:
    def test_a_perfect_prediction_has_no_distortion(self):
        frame = pd.DataFrame({TRUTH: ["01.1", "02.1"], FINAL: ["01.1", "02.1"]})
        out = I.distortion_level1(frame, truth_col=TRUTH, final_col=FINAL)
        assert out["tv_distance"] == 0.0
        assert out["erreurs_brutes"] == 0

    def test_errors_that_cancel_out_leave_the_distribution_intact(self):
        """LE cas qui justifie le chapitre : deux produits échangent leur
        division. Tout est faux, la répartition est parfaite."""
        frame = pd.DataFrame({TRUTH: ["01.1", "02.1"], FINAL: ["02.1", "01.1"]})
        out = I.distortion_level1(frame, truth_col=TRUTH, final_col=FINAL)
        assert out["erreurs_brutes"] == 2
        assert out["tv_distance"] == 0.0
        assert out["compensation"] == 1.0

    def test_errors_all_in_the_same_direction_do_not_compensate(self):
        """Le cas opposé : les deux erreurs vont vers 01, l'agrégat est biaisé
        d'autant et rien ne s'annule."""
        frame = pd.DataFrame(
            {TRUTH: ["01.1", "02.1", "03.1"], FINAL: ["01.1", "01.1", "01.1"]}
        )
        out = I.distortion_level1(frame, truth_col=TRUTH, final_col=FINAL)
        assert out["erreurs_brutes"] == 2
        assert out["compensation"] == 0.0
        # 2 produits sur 3 à déplacer pour retrouver la vraie répartition.
        assert out["deplacements"] == 2
        assert round(out["tv_distance"], 6) == round(2 / 3, 6)

    def test_an_abstention_is_its_own_category(self):
        """Une conciliation muette ne disparaît pas du tableau : elle sous-code
        une division réelle, et c'est visible comme telle."""
        frame = pd.DataFrame({TRUTH: ["01.1", "01.1"], FINAL: ["01.1", None]})
        out = I.distortion_level1(frame, truth_col=TRUTH, final_col=FINAL)
        assert "<none>" in [c["category"] for c in out["per_category"]]
        assert out["erreurs_brutes"] == 1

    def test_over_and_under_coding_carry_opposite_signs(self):
        frame = pd.DataFrame(
            {TRUTH: ["01.1", "02.1", "02.1"], FINAL: ["01.1", "01.1", "02.1"]}
        )
        out = I.distortion_level1(frame, truth_col=TRUTH, final_col=FINAL)
        par_div = {c["category"]: c for c in out["per_category"]}
        assert par_div["01"]["diff"] > 0   # sur-codée
        assert par_div["02"]["diff"] < 0   # sous-codée

    def test_no_labelled_row_yields_none(self):
        frame = pd.DataFrame({TRUTH: [None, None], FINAL: ["01.1", "02.1"]})
        assert I.distortion_level1(frame, truth_col=TRUTH, final_col=FINAL) is None


class TestDivisionLabels:
    @staticmethod
    def _nomenclature():
        """Une nomenclature contient TOUS les niveaux : on ne veut que les
        divisions, c'est-à-dire les codes à un seul segment."""
        return pd.DataFrame({
            "code": ["01", "02", "01.1", "01.1.1", "11"],
            "label_fr": [
                "Produits alimentaires et boissons non alcoolisées",
                "Boissons alcoolisées et tabac",
                "Produits alimentaires",
                "Pain et céréales",
                "Restaurants et hôtels",
            ],
        })

    def test_keeps_only_the_divisions(self, monkeypatch):
        monkeypatch.setattr(I, "_read", lambda con, path: self._nomenclature())
        labels = I.division_labels(None, "peu importe")
        assert set(labels) == {"01", "02", "11"}
        assert labels["11"] == "Restaurants et hôtels"

    def test_a_missing_nomenclature_is_not_an_error(self, monkeypatch):
        """Un libellé absent ne doit jamais faire échouer un rapport de mesure."""
        monkeypatch.setattr(I, "_read", lambda con, path: None)
        assert I.division_labels(None, "absente") == {}

    def test_a_nomenclature_without_the_columns_is_ignored(self, monkeypatch):
        monkeypatch.setattr(I, "_read", lambda con, path: pd.DataFrame({"code": ["01"]}))
        assert I.division_labels(None, "sans libellés") == {}

    def test_formats_code_and_label_together(self):
        labels = {"01": "Produits alimentaires"}
        assert I.label_division("01", labels) == "01 — Produits alimentaires"

    def test_falls_back_to_the_bare_code(self):
        assert I.label_division("98", {}) == "98"

    def test_truncates_a_very_long_label(self):
        labels = {"01": "Produits alimentaires et boissons non alcoolisées"}
        out = I.label_division("01", labels, width=20)
        assert out.startswith("01 — Produits") and out.endswith("…")


class TestAccuracyByPredictedDivision:
    @staticmethod
    def _frame():
        """Cinq produits annoncés en `01`, dont un qui est en réalité du `11`.
        Parmi les quatre vraiment alimentaires, deux ont le bon code complet."""
        return pd.DataFrame({
            TRUTH: ["01.1.1.1", "01.1.1.1", "01.2.2.2", "01.3.3.3", "11.1.1.1"],
            FINAL: ["01.1.1.1", "01.1.1.1", "01.9.9.9", "01.9.9.9", "01.1.1.1"],
        })

    def _row(self, label="01"):
        out = I.accuracy_by_predicted_division(
            self._frame(), truth_col=TRUTH, final_col=FINAL,
            labels={"01": "Produits alimentaires"},
        )
        return out[out["division prédite"].str.startswith(label)].iloc[0]

    def test_answers_the_operational_question(self):
        """« Quand la chaîne annonce de l'alimentaire, a-t-elle le bon code ? »
        4 des 5 annonces sont bien de l'alimentaire, 2 ont le code complet."""
        row = self._row()
        assert row["n prédits"] == 5
        assert row["bonne division"] == 0.8
        assert row["bon code niv4"] == 0.4

    def test_the_full_code_can_never_beat_the_division(self):
        """Avoir le bon code de niveau 4 suppose la bonne division : le second
        taux ne peut pas dépasser le premier, à aucune ligne."""
        out = I.accuracy_by_predicted_division(
            self._frame(), truth_col=TRUTH, final_col=FINAL
        )
        assert (out["bon code niv4"] <= out["bonne division"]).all()

    def test_it_groups_by_the_prediction_not_the_truth(self):
        """La ligne `11` n'existe pas : aucune prédiction n'annonce `11`, même
        si un produit en relève réellement. C'est toute la différence avec la
        ventilation par division vraie."""
        out = I.accuracy_by_predicted_division(
            self._frame(), truth_col=TRUTH, final_col=FINAL
        )
        assert list(out["division prédite"]) == ["01"]

    def test_abstentions_form_their_own_group(self):
        frame = pd.DataFrame({TRUTH: ["01.1.1.1", "01.1.1.1"], FINAL: ["01.1.1.1", None]})
        out = I.accuracy_by_predicted_division(frame, truth_col=TRUTH, final_col=FINAL)
        assert "— (aucun code émis)" in list(out["division prédite"])

    def test_labels_are_used_when_available(self):
        assert "Produits alimentaires" in self._row()["division prédite"]
