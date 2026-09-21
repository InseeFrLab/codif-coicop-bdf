"""Tests des URLs MLflow.

Seules les fonctions pures sont testées : `find_runs` a besoin d'un serveur, et
son contrat est de ne jamais lever — c'est vérifié ici sans serveur, ce qui est
précisément le cas qui doit se dégrader proprement.

Lancer depuis `common/` : `uv run pytest tests/test_tracking.py`
"""

from codif_common.tracking import (
    TAG_RUN_ID,
    TAG_STEP,
    find_runs,
    run_url,
    url_from_artifact_uri,
)

TRACKING = "https://projet-budget-famille-mlflow.user.lab.sspcloud.fr"


class TestRunUrl:
    def test_open_source_ui_lives_behind_a_hash(self):
        assert run_url(TRACKING, "15", "abc123") == f"{TRACKING}/#/experiments/15/runs/abc123"

    def test_trailing_slash_does_not_double(self):
        """Un `MLFLOW_TRACKING_URI` terminé par `/` dans le secret Kubernetes
        produirait sinon `https://hôte//#/experiments/…`."""
        assert run_url(TRACKING + "/", "15", "abc") == run_url(TRACKING, "15", "abc")

    def test_databricks_has_no_hash(self):
        url = run_url("databricks://scope", "15", "abc")
        assert "#" not in url and url.endswith("/ml/experiments/15/runs/abc")


class TestArtifactUri:
    def test_extracts_experiment_and_run_from_a_model_uri(self):
        """C'est ainsi que les runs d'entraînement de classify-ttc et de
        reconcile-sirus se retrouvent : ils n'ouvrent aucun run pendant le
        pipeline, mais l'URI de leur modèle porte les deux identifiants."""
        uri = "mlflow-artifacts:/15/312f69aff83943de97b3a91b1f4185fa/artifacts/model"
        assert url_from_artifact_uri(TRACKING, uri) == (
            f"{TRACKING}/#/experiments/15/runs/312f69aff83943de97b3a91b1f4185fa"
        )

    def test_tolerates_the_trailing_slash_of_params_yaml(self):
        """`argo/params.yaml` écrit l'URI avec un `/` final."""
        uri = "mlflow-artifacts:/15/abc/artifacts/model/"
        assert url_from_artifact_uri(TRACKING, uri).endswith("/experiments/15/runs/abc")

    def test_other_uri_shapes_degrade_to_none(self):
        for uri in ["runs:/abc/model", "models:/nom/1", "/chemin/local", "", None]:
            assert url_from_artifact_uri(TRACKING, uri) is None, uri


class TestDegradation:
    def test_unreachable_tracking_server_yields_no_run_rather_than_raising(self):
        """Un rapport d'évaluation ne doit pas échouer parce qu'un tableau de
        liens n'a pas pu être rempli."""
        assert (
            find_runs(
                "run-inexistant", "classify-rag-notices",
                experiment_names=["experience-inexistante"],
            )
            == []
        )

    def test_tag_name_does_not_collide_with_the_indexing_tag(self):
        """`index.run_id` existe déjà dans les deux étapes RAG et désigne le run
        d'indexation de la vector DB, pas le run de pipeline."""
        assert TAG_RUN_ID == "pipeline.run_id" != "index.run_id"

    def test_the_step_tag_is_what_makes_the_lookup_unambiguous(self):
        """Toutes les étapes d'un run portent le même `pipeline.run_id` : sans le
        nom de l'étape, la recherche renverrait celle qui a fini en dernier."""
        assert TAG_STEP == "pipeline.step"
