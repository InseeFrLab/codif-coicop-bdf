"""Tests de l'élagage des hiérarchies linéaires.

Deux niveaux de test :
- cas synthétiques ciblant la règle d'équivalence (un parent à enfant unique
  est équivalent à son enfant, même si celui-ci a plusieurs enfants) ;
- tests d'intégration sur la nomenclature COICOP 2018 réelle (copie locale
  vendorisée dans classify-ttc/data/).
"""

from pathlib import Path

import pandas as pd
import pytest

from prune_codes.pruning import prune_linear_hierarchies, trunc_and_prune_lvl4

NOMENCLATURE_CSV = (
    Path(__file__).resolve().parents[2]
    / "classify-ttc"
    / "data"
    / "coicop-2018_envoi_rmes_20251022.csv"
)

# Les 7 codes redondants historiquement oubliés par l'élagage : chacun est
# l'enfant unique de son parent, mais a lui-même plusieurs enfants, ce que
# l'ancienne implémentation excluait à tort de la relation d'équivalence.
HISTORICALLY_MISSED = {
    "02.3.0": "02.3",
    "05.4.0": "05.4",
    "10.1.0": "10.1",
    "10.5.0": "10.5",
    "11.2.0": "11.2",
    "13.3.0": "13.3",
    "13.9.0": "13.9",
}


def _synthetic_nomenclature() -> pd.DataFrame:
    """Mini-nomenclature : 11.2 -> 11.2.0 (enfant unique) -> 4 sous-classes,
    plus une chaîne strictement linéaire 01.3 -> 01.3.0 -> 01.3.0.0 et une
    branche multi-enfants 11.1 qui ne doit pas être touchée."""
    rows = [
        ("Division", None, "01"),
        ("Groupe", "01", "01.1"),
        ("Classe", "01.1", "01.1.1"),
        ("Classe", "01.1", "01.1.2"),
        ("Groupe", "01", "01.3"),
        ("Classe", "01.3", "01.3.0"),
        ("Sous-classe", "01.3.0", "01.3.0.0"),
        ("Division", None, "11"),
        ("Groupe", "11", "11.1"),
        ("Classe", "11.1", "11.1.1"),
        ("Classe", "11.1", "11.1.2"),
        ("Groupe", "11", "11.2"),
        ("Classe", "11.2", "11.2.0"),
        ("Sous-classe", "11.2.0", "11.2.0.1"),
        ("Sous-classe", "11.2.0", "11.2.0.2"),
        ("Sous-classe", "11.2.0", "11.2.0.3"),
        ("Sous-classe", "11.2.0", "11.2.0.4"),
    ]
    df = pd.DataFrame(rows, columns=["type", "parent", "code"])
    df["label_fr"] = "libellé " + df["code"]
    return df


class TestSyntheticChains:
    def test_single_child_parent_collapses_even_if_child_branches(self):
        """11.2 n'a qu'un enfant (11.2.0) : ils sont équivalents, même si
        11.2.0 a 4 enfants. 11.2.0 doit être replié sur 11.2."""
        pruned, mapping = prune_linear_hierarchies(_synthetic_nomenclature())
        mapping_dict = dict(zip(mapping["code"], mapping["code_parent_equivalent"]))

        assert mapping_dict.get("11.2.0") == "11.2"
        assert "11.2.0" not in set(pruned["code"])
        assert "11.2" in set(pruned["code"])

    def test_strictly_linear_chain_collapses_to_root(self):
        pruned, mapping = prune_linear_hierarchies(_synthetic_nomenclature())
        mapping_dict = dict(zip(mapping["code"], mapping["code_parent_equivalent"]))

        assert mapping_dict.get("01.3.0") == "01.3"
        assert mapping_dict.get("01.3.0.0") == "01.3"
        assert set(pruned["code"]) & {"01.3.0", "01.3.0.0"} == set()

    def test_branching_nodes_untouched(self):
        pruned, _ = prune_linear_hierarchies(_synthetic_nomenclature())
        codes = set(pruned["code"])
        assert {"11", "11.1", "11.1.1", "11.1.2"} <= codes

    def test_children_of_collapsed_code_survive_with_remapped_parent(self):
        """Les enfants de 11.2.0 survivent et leur parent est remappé sur 11.2."""
        pruned, _ = prune_linear_hierarchies(_synthetic_nomenclature())
        children = pruned[pruned["code"].str.startswith("11.2.0.")]
        assert len(children) == 4
        assert (children["parent"] == "11.2").all()

    def test_no_residual_synonyms(self):
        """Après élagage, aucun code de la nomenclature prunée ne doit avoir
        exactement un enfant (sinon il reste une paire de synonymes)."""
        pruned, _ = prune_linear_hierarchies(_synthetic_nomenclature())
        nb_children = pruned.groupby("parent")["code"].nunique()
        assert (nb_children != 1).all(), (
            f"parents à enfant unique restants : {nb_children[nb_children == 1].index.tolist()}"
        )


@pytest.fixture(scope="module")
def real_nomenclature() -> pd.DataFrame:
    if not NOMENCLATURE_CSV.exists():
        pytest.skip(f"nomenclature locale absente : {NOMENCLATURE_CSV}")
    df = pd.read_csv(NOMENCLATURE_CSV, sep=";", dtype=str)
    return df.loc[df["type"] != "Poste"]


@pytest.fixture(scope="module")
def real_pruned(real_nomenclature):
    return prune_linear_hierarchies(real_nomenclature.copy())


class TestRealNomenclature:
    def test_historically_missed_codes_are_collapsed(self, real_pruned):
        pruned, mapping = real_pruned
        mapping_dict = dict(zip(mapping["code"], mapping["code_parent_equivalent"]))
        pruned_codes = set(pruned["code"])

        for code, parent in HISTORICALLY_MISSED.items():
            assert mapping_dict.get(code) == parent, f"{code} devrait mapper vers {parent}"
            assert code not in pruned_codes, f"{code} devrait être retiré de la nomenclature prunée"
            assert parent in pruned_codes

    def test_previous_collapses_are_preserved(self, real_pruned):
        """Les replis déjà effectués par l'ancienne implémentation restent
        corrects (échantillon représentatif)."""
        _, mapping = real_pruned
        mapping_dict = dict(zip(mapping["code"], mapping["code_parent_equivalent"]))
        expected = {
            "01.2.1.0": "01.2.1",
            "01.3.0": "01.3",
            "01.3.0.0": "01.3",
            "02.2.0": "02.2",
            "02.2.0.0": "02.2",
            "10.4.0.0": "10.4",
        }
        for code, parent in expected.items():
            assert mapping_dict.get(code) == parent

    def test_no_residual_synonyms(self, real_pruned):
        pruned, _ = real_pruned
        nb_children = pruned.groupby("parent")["code"].nunique()
        offenders = nb_children[nb_children == 1].index.tolist()
        assert not offenders, f"parents à enfant unique restants : {offenders}"

    def test_no_collapse_below_level_2(self, real_pruned):
        """Aucun code ne doit être replié sur une division entière (niveau 1)."""
        _, mapping = real_pruned
        levels = mapping["code_parent_equivalent"].str.count(r"\.") + 1
        assert (levels >= 2).all()

    def test_all_parents_exist_in_pruned_nomenclature(self, real_pruned):
        """La colonne parent de la nomenclature prunée ne pointe jamais vers
        un code supprimé."""
        pruned, _ = real_pruned
        codes = set(pruned["code"])
        parents = set(pruned["parent"].dropna())
        assert parents <= codes, f"parents orphelins : {sorted(parents - codes)}"

    def test_trunc_and_prune_is_idempotent(self, real_pruned):
        _, mapping = real_pruned
        df = pd.DataFrame(
            {"code": ["11.2.0", "01.3.0.0.1", "11.2", "98.1.1", None, "05.4.0.2"]}
        )
        once = trunc_and_prune_lvl4(df, mapping)["code_tpruned"]
        twice = trunc_and_prune_lvl4(
            once.to_frame(name="code"), mapping
        )["code_tpruned"]
        assert once.tolist() == twice.tolist()
        # troncature + repli attendus
        assert once.tolist()[:3] == ["11.2", "01.3", "11.2"]
        # les codes techniques BdF hors nomenclature passent inchangés
        assert once.tolist()[3] == "98.1.1"
