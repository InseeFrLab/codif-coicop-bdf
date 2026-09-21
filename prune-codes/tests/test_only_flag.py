"""Tests du drapeau `--only` de scripts/main.py.

Ce que ces tests verrouillent : en mode `--only nomenclature`, l'étape 1 seule
tourne et le script **ne lit aucune clé de config liée aux étapes 2 et 3**
(`features`, `annotations_train`, `annotations_test`, `suggester`,
`product_col`). C'est exactement le contrat dont dépend le pipeline Argo
d'indexation des notices, qui n'a pas de sortie `classify-regex` : si quelqu'un
remonte un jour `features = cfg["features"]` au-dessus de la sortie anticipée,
ce pipeline casse, et c'est ce test qui doit le dire.

La preuve tient en deux temps : la config minimale suffit en mode
`nomenclature`, et la **même** config échoue en mode `all`. Sans le second test,
le premier serait vacant.
"""

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

from test_pruning import _synthetic_nomenclature

MAIN_PY = Path(__file__).resolve().parents[1] / "scripts" / "main.py"


def _load_main_module():
    """Charge scripts/main.py comme module (ce n'est pas un paquet importable)."""
    spec = importlib.util.spec_from_file_location("prune_codes_main", MAIN_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeConnection:
    """Connexion DuckDB minimale : sert la nomenclature synthétique en lecture
    et enregistre les écritures parquet au lieu de les envoyer sur S3."""

    def __init__(self, nomenclature: pd.DataFrame):
        self._nomenclature = nomenclature
        self.registered: dict[str, pd.DataFrame] = {}
        self.copies: list[str] = []

    def sql(self, query: str):
        if query.strip().upper().startswith("COPY"):
            self.copies.append(query)
            return None
        return self

    def to_df(self) -> pd.DataFrame:
        return self._nomenclature.copy()

    def register(self, name: str, df: pd.DataFrame) -> None:
        self.registered[name] = df


@pytest.fixture
def minimal_config(tmp_path: Path) -> Path:
    """Config ne portant QUE ce dont l'étape 1 a besoin.

    Les clés des étapes 2 et 3 sont délibérément absentes : leur lecture
    lèverait un KeyError, ce qui est le mécanisme de détection du test.
    """
    cfg = {
        "nomenclature_raw": "s3://bucket/nomenclature.csv",
        "outputs": {
            "nomenclature_pruned": "s3://bucket/{run_date}/{run_id}/prune-codes/nomenclature_pruned.parquet",
            "mapping_lvl4": "s3://bucket/{run_date}/{run_id}/prune-codes/mapping_lvl4.parquet",
        },
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


@pytest.fixture
def main_module(monkeypatch):
    module = _load_main_module()
    fake_con = _FakeConnection(_synthetic_nomenclature())
    monkeypatch.setattr(module, "create_duckdb_connection", lambda: fake_con)
    return module, fake_con


def _run(module, config_path: Path, *extra: str):
    argv = [
        "main.py",
        "--config", str(config_path),
        "--run-id", "index-notices-test",
        "--run-date", "2026-09-02",
        *extra,
    ]
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(sys, "argv", argv)
        module.main()


class TestOnlyNomenclature:
    def test_runs_with_a_config_that_lacks_every_downstream_key(
        self, main_module, minimal_config
    ):
        """Le cas d'usage du pipeline d'indexation des notices."""
        module, con = main_module
        _run(module, minimal_config, "--only", "nomenclature")

        # Étape 1 faite : les deux artefacts sont écrits, et eux seuls.
        assert set(con.registered) == {"nomenclature_pruned", "mapping_table"}
        assert len(con.copies) == 2

    def test_writes_the_two_stage_one_artifacts_to_the_run_folder(
        self, main_module, minimal_config
    ):
        module, con = main_module
        _run(module, minimal_config, "--only", "nomenclature")

        written = " ".join(con.copies)
        assert "2026-09-02/index-notices-test/prune-codes/nomenclature_pruned.parquet" in written
        assert "2026-09-02/index-notices-test/prune-codes/mapping_lvl4.parquet" in written

    def test_pruning_actually_happened(self, main_module, minimal_config):
        """Le mode restreint ne court-circuite pas l'élagage lui-même."""
        module, con = main_module
        _run(module, minimal_config, "--only", "nomenclature")

        mapping = con.registered["mapping_table"]
        assert dict(zip(mapping["code"], mapping["code_parent_equivalent"]))["11.2.0"] == "11.2"


class TestOnlyKb:
    """`--only kb` : le pipeline d'indexation des annotations. Il produit la KB
    et le suggester, mais PAS le jeu à coder — dont la source est une sortie
    `classify-regex` que ce pipeline ne génère pas."""

    @pytest.fixture
    def kb_module(self, main_module, monkeypatch):
        """Neutralise les deux chargements lourds pour n'observer que le flux
        de contrôle : qui est appelé, avec quelle source."""
        module, con = main_module
        calls = {"annotations": [], "suggester": []}

        def fake_prune_annotations_file(con_, in_path, features, mapping, notices, log):
            calls["annotations"].append(in_path)
            return pd.DataFrame({"code": ["01.1"], "l_pr_product": ["café"]})

        def fake_load_and_prune_suggester(con_, sug_cfg, product_col, mapping, notices, log):
            calls["suggester"].append(sug_cfg["s3_path"])
            return pd.DataFrame({"code": ["01.1"], "l_pr_product": ["thé"]})

        monkeypatch.setattr(module, "prune_annotations_file", fake_prune_annotations_file)
        monkeypatch.setattr(module, "load_and_prune_suggester", fake_load_and_prune_suggester)
        return module, con, calls

    @pytest.fixture
    def kb_config(self, tmp_path: Path) -> Path:
        """Config SANS `annotations_test` : sa lecture lèverait un KeyError,
        ce qui est le mécanisme de détection du test."""
        cfg = {
            "nomenclature_raw": "s3://bucket/nomenclature.csv",
            "annotations_train": "s3://bucket/defaut-classify-regex.parquet",
            "features": ["code"],
            "product_col": "l_pr_product",
            "suggester": {"s3_path": "s3://bucket/defaut.csv", "source_product_col": "product"},
            "outputs": {
                "nomenclature_pruned": "s3://bucket/{run_date}/{run_id}/prune-codes/nomenclature_pruned.parquet",
                "mapping_lvl4": "s3://bucket/{run_date}/{run_id}/prune-codes/mapping_lvl4.parquet",
                "annotations_train_pruned": "s3://bucket/{run_date}/{run_id}/prune-codes/annotations_train_pruned.parquet",
                "suggester_pruned": "s3://bucket/{run_date}/{run_id}/prune-codes/suggester_pruned.parquet",
            },
        }
        path = tmp_path / "config.yaml"
        path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        return path

    def test_never_reads_the_to_codify_set(self, kb_module, kb_config):
        module, _, calls = kb_module
        _run(module, kb_config, "--only", "kb")
        # Une seule source d'annotations lue : la KB. Jamais annotations_test.
        assert len(calls["annotations"]) == 1
        assert len(calls["suggester"]) == 1

    def test_kb_source_can_be_overridden(self, kb_module, kb_config):
        """C'est ainsi que le pipeline pointe annotations_full au lieu de la
        sortie de classify-regex."""
        module, _, calls = kb_module
        _run(module, kb_config, "--only", "kb",
             "--annotations-train", "s3://b/build-datasets/annotations_full.parquet")
        assert calls["annotations"] == ["s3://b/build-datasets/annotations_full.parquet"]

    def test_suggester_source_can_be_overridden(self, kb_module, kb_config):
        """Le suggester préprocessé de build-datasets, pas le CSV brut."""
        module, _, calls = kb_module
        _run(module, kb_config, "--only", "kb",
             "--suggester-path", "s3://b/build-datasets/suggester.parquet",
             "--suggester-product-col", "l_pr_product")
        assert calls["suggester"] == ["s3://b/build-datasets/suggester.parquet"]

    def test_falls_back_to_config_without_overrides(self, kb_module, kb_config):
        module, _, calls = kb_module
        _run(module, kb_config, "--only", "kb")
        assert calls["annotations"] == ["s3://bucket/defaut-classify-regex.parquet"]
        assert calls["suggester"] == ["s3://bucket/defaut.csv"]

    def test_full_mode_needs_the_key_kb_mode_does_not(self, kb_module, kb_config):
        """Contre-épreuve : la même config échoue en mode complet, faute
        d'`annotations_test`. Sans ça, les tests ci-dessus seraient vacants."""
        module, _, _ = kb_module
        with pytest.raises(KeyError):
            _run(module, kb_config, "--only", "all")


class TestDefaultIsUnchanged:
    def test_same_config_fails_in_full_mode(self, main_module, minimal_config):
        """Contre-épreuve : la config minimale est bien insuffisante en mode
        complet. Sans ça, le test ci-dessus ne prouverait rien."""
        module, _ = main_module
        with pytest.raises(KeyError):
            _run(module, minimal_config, "--only", "all")

    def test_default_is_full_mode(self, main_module, minimal_config):
        """Aucun appel existant ne change : sans --only, on est en mode complet.
        Le template prune-codes de codif-pipeline.yaml n'en passe pas."""
        module, _ = main_module
        with pytest.raises(KeyError):
            _run(module, minimal_config)
