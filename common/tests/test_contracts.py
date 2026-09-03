"""Tests du registre des artefacts.

Le test qui compte est `test_no_dangling_inputs` : c'est celui qui aurait rendu
visible le décalage entre `build-datasets`, qui écrit le suggester sous
`build-datasets/suggester.parquet`, et `classify-lcs`, qui le cherchait à
`{run_root}/suggester.parquet`. Un `tryCatch` avalait l'erreur, la branche
n'était jamais prise, et rien ne le signalait.
"""

import pytest

from codif_common.contracts import (
    artifact,
    bucket,
    consumers,
    dangling_inputs,
    external,
    load_registry,
    run_root,
)

RUN = {"run_date": "2026-09-03", "run_id": "codif-abc12"}


class TestRegistryIsCoherent:
    def test_no_dangling_inputs(self):
        """Toute entrée déclarée doit désigner une sortie déclarée."""
        assert dangling_inputs() == []

    def test_every_step_declares_something(self):
        for name, spec in load_registry()["steps"].items():
            assert "outputs" in spec, f"{name} ne déclare pas d'outputs"

    def test_no_step_consumes_itself(self):
        for name, spec in load_registry()["steps"].items():
            for ref in spec.get("inputs") or []:
                assert not ref.startswith(f"{name}."), f"{name} se consomme lui-même"


class TestArtifactPaths:
    def test_builds_a_full_uri(self):
        assert artifact("prune-codes", "mapping_lvl4", **RUN) == (
            "s3://projet-budget-famille/data/workflow_runs/2026-09-03/codif-abc12"
            "/prune-codes/mapping_lvl4.parquet"
        )

    def test_run_root_is_shared_by_all_steps(self):
        root = run_root(**RUN)
        for step, key in [("build-datasets", "raw_test"), ("classify-lcs", "predictions")]:
            assert artifact(step, key, **RUN).startswith(root)

    def test_extra_placeholders_are_supported(self):
        """Le livrable porte le nom du fichier d'entrée : seul gabarit à
        variable supplémentaire."""
        path = artifact("export-results", "deliverable", **RUN, filename="mes_tickets.csv")
        assert path.endswith("/export-results/mes_tickets.csv")

    def test_unknown_step_fails_loudly(self):
        """Une faute de frappe doit lever ici, pas produire un chemin S3
        plausible que personne n'a jamais écrit."""
        with pytest.raises(KeyError, match="prune-code"):
            artifact("prune-code", "mapping_lvl4", **RUN)

    def test_unknown_key_lists_what_exists(self):
        with pytest.raises(KeyError) as e:
            artifact("prune-codes", "mapping_niv4", **RUN)
        assert "mapping_lvl4" in str(e.value)


class TestConsumers:
    def test_answers_who_breaks_if_i_change_this(self):
        """`mapping_lvl4` est la sortie la plus consommée du pipeline : elle
        était désignée 5 fois par 4 mécanismes différents."""
        assert consumers("prune-codes", "mapping_lvl4") == [
            "classify-rag-notices", "export-results", "reconcile-llm", "reconcile-sirus",
        ]

    def test_an_orphan_output_has_no_consumer(self):
        """`retrieved_codes` est écrit puis jamais relu — le registre le dit."""
        assert consumers("classify-rag-notices", "retrieved_codes") == []


class TestExternalInputs:
    def test_resolves_outside_the_run(self):
        path = external("nomenclature_coicop")
        assert "workflow_runs" not in path
        assert path.endswith("coicop-2018_envoi_rmes_20251022.csv")

    def test_unknown_external_fails(self):
        with pytest.raises(KeyError):
            external("nomenclature_2026")


class TestBucketOverride:
    def test_env_var_wins(self, monkeypatch):
        monkeypatch.setenv("COICOP_BUCKET", "bac-a-sable")
        assert bucket() == "bac-a-sable"
        assert artifact("prune-codes", "mapping_lvl4", **RUN).startswith("s3://bac-a-sable/")

    def test_default_comes_from_the_registry(self, monkeypatch):
        monkeypatch.delenv("COICOP_BUCKET", raising=False)
        assert bucket() == "projet-budget-famille"
