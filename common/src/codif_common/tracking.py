"""Retrouver un run MLflow et en fabriquer une URL cliquable.

POURQUOI CE MODULE
Un rapport d'évaluation dit ce qu'a produit chaque brique, jamais *quel* run l'a
produite. Or rien dans le pipeline ne relie les deux : aucune étape ne persiste
son identifiant de run MLflow — ni dans un parquet, ni dans un artefact S3, ni
dans un paramètre de sortie Argo — et les `run_name` des deux étapes RAG ne
portent qu'un horodatage (`run_{timestamp}`), qui ne désigne rien.

Le raccordement se fait donc par un tag posé au moment du run, `pipeline.run_id`,
et retrouvé ici par recherche. La notation pointée suit celle déjà en place dans
le dépôt (`git.commit`, `index.run_id`).

`mlflow` est importé **dans** les fonctions : `codif-common` est le socle dont
tous les modules dépendent, et il ne doit pas leur imposer mlflow. Les deux
fonctions de construction d'URL sont pures et n'en ont pas besoin du tout.
"""

from __future__ import annotations

import logging
import re
from typing import Sequence

logger = logging.getLogger(__name__)

# Tags par lesquels une étape se rattache à son run de pipeline. À ne pas
# confondre avec `index.run_id`, déjà posé par les deux étapes RAG, qui désigne
# le run d'INDEXATION de la vector DB — un autre workflow, un autre run.
TAG_RUN_ID = "pipeline.run_id"
TAG_RUN_DATE = "pipeline.run_date"
# Le tag du run seul ne suffit pas : toutes les étapes d'un même run de pipeline
# le portent, et une recherche sur ce seul critère renverrait celle qui a fini
# en dernier, quelle qu'elle soit. Le nom de l'étape est ce qui rend le
# rattachement univoque.
TAG_STEP = "pipeline.step"

# `mlflow-artifacts:/{experiment_id}/{run_id}/artifacts/...`, la forme que prennent
# `classify-ttc-model-uri` et la colonne `sirus_model_uri`. Les deux identifiants
# y sont, donc un modèle suffit à retrouver le run qui l'a entraîné — sans tag et
# sans recherche.
_ARTIFACT_URI = re.compile(r"^mlflow-artifacts:/+([^/]+)/([^/]+)/artifacts")


def run_url(tracking_uri: str, experiment_id: str, run_id: str) -> str:
    """URL de la page d'un run dans l'UI MLflow.

    Forme reprise de `MlflowClient._get_run_url` : l'UI open-source est une SPA
    servie derrière un `#`, Databricks non. Le `rstrip` n'est pas cosmétique —
    un `MLFLOW_TRACKING_URI` terminé par `/` dans le secret Kubernetes
    produirait `https://hôte//#/experiments/…`.
    """
    host = str(tracking_uri).rstrip("/")
    if host.startswith("databricks"):
        return f"{host}/ml/experiments/{experiment_id}/runs/{run_id}"
    return f"{host}/#/experiments/{experiment_id}/runs/{run_id}"


def url_from_artifact_uri(tracking_uri: str, artifact_uri: str) -> str | None:
    """URL du run d'entraînement désigné par une URI d'artefacts MLflow.

    C'est ainsi que les runs de `classify-ttc` et de `reconcile-sirus` se
    retrouvent : ils n'ouvrent pas de run pendant le pipeline (ils ne font
    qu'inférer), mais le modèle qu'ils chargent porte dans son URI l'expérience
    et le run de son entraînement.

    Renvoie ``None`` sur une URI d'une autre forme (`runs:/`, `models:/`, un
    chemin local) plutôt que de lever : la traçabilité se dégrade, elle ne casse
    pas le rapport.
    """
    if not artifact_uri:
        return None
    m = _ARTIFACT_URI.match(str(artifact_uri).strip())
    if not m:
        return None
    return run_url(tracking_uri, m.group(1), m.group(2))


def find_runs(
    run_id: str, step: str, *, experiment_names: Sequence[str] | None = None
) -> list:
    """Runs MLflow d'une étape donnée pour un run de pipeline donné, du plus
    récent au plus ancien.

    Renvoie une liste, et pas un run, parce qu'il peut y en avoir plusieurs :
    les étapes du pipeline sont sous `retryStrategy: limit "2"`, donc un run de
    pipeline qui a réessayé laisse deux ou trois runs MLflow légitimes portant
    le même tag. Choisir en silence donnerait un lien arbitraire.

    ``experiment_names=None`` cherche dans **toutes** les expériences. C'est le
    défaut voulu : le nom de l'expérience des étapes RAG n'est pas fixe — il
    vient de leur `config.yaml` (`test`, `rag-annotation`), sauf quand le
    paramètre Argo `rag-experiment` le surcharge, ce que fait la passe smoke
    (`codif-coicop-smoke-notices`). Le coder en dur donnerait un tableau vide
    sur un run smoke, sans rien signaler.

    Ne lève jamais : MLflow injoignable, expérience absente ou filtre refusé
    renvoient une liste vide.
    """
    try:
        import mlflow

        # `search_all_experiments` est ignoré si `experiment_names` est fourni ;
        # sans l'un ni l'autre, mlflow retombe sur l'expérience *active* — donc
        # `Default` dans un processus de rendu, et un résultat vide qui ressemble
        # à « pas de run » au lieu de « mauvais périmètre ».
        return mlflow.search_runs(
            experiment_names=list(experiment_names) if experiment_names else None,
            search_all_experiments=not experiment_names,
            # Clé entre guillemets doubles, valeur entre simples : la clé est
            # pointée, et l'analyseur de mlflow ne coupe que sur le PREMIER
            # point (`tags.` / le reste). Sans les guillemets, `pipeline.run_id`
            # passe encore côté client, mais le filtre est réanalysé par le
            # serveur avec sa propre version de mlflow — la forme citée est la
            # seule sur laquelle compter.
            filter_string=(
                f"tags.\"{TAG_RUN_ID}\" = '{run_id}' and tags.\"{TAG_STEP}\" = '{step}'"
            ),
            order_by=["attributes.start_time DESC"],
            max_results=5,
            output_format="list",
        )
    except Exception as exc:  # noqa: BLE001 — la traçabilité ne casse pas un rapport
        logger.warning(
            "recherche MLflow impossible pour run_id=%s étape=%s : %s", run_id, step, exc
        )
        return []


def find_run_url(
    tracking_uri: str,
    run_id: str,
    step: str,
    *,
    experiment_names: Sequence[str] | None = None,
) -> tuple[str | None, int]:
    """``(url du run le plus récent, nombre de runs trouvés)``.

    Le compte est renvoyé pour que l'appelant puisse signaler une reprise sur
    échec (« 2 runs ») au lieu de faire croire qu'il n'y en a qu'un.
    """
    runs = find_runs(run_id, step, experiment_names=experiment_names)
    if not runs:
        return None, 0
    info = runs[0].info
    return run_url(tracking_uri, info.experiment_id, info.run_id), len(runs)
