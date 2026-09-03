"""Nommage et manifeste des collections Qdrant.

Depuis la scission de l'indexation en pipelines Argo autonomes, chaque build
produit une collection au **nom unique** au lieu d'écraser un nom fixe partagé
par tous les runs. Le nom seul ne peut pas porter tout ce qui détermine la
validité d'un index (modèle d'embedding, stratégie de découpage, périmètre
de la KB, CSV source) : deux collections de même dimension bâties avec des
réglages différents sont indistinguables, et interroger la mauvaise ne lève
aucune erreur — le RAG rend juste de moins bons résultats. D'où le manifeste,
écrit à côté de la collection et relu par le consommateur au démarrage.

Ce fichier a vécu un temps **dupliqué à l'identique** dans `rag_notices` et
`rag_annotations`, faute de module partagé — avec le risque que les deux
copies divergent en silence, alors que le consommateur relit le manifeste
écrit par le constructeur : un écart de schéma ou de règle de nommage y casse
la validation au démarrage. Les deux modules importent désormais d'ici.
"""

import json
import subprocess
from typing import Any, Dict, Optional

import duckdb
import pandas as pd

# Séparateur entre segments du nom. Doublé pour rester lisible même quand les
# segments contiennent eux-mêmes des tirets (`codif-abc12`, `2026-09-02`).
SEP = "__"


def build_collection_name(
    base: str,
    run_date: str,
    run_id: str,
    mode: Optional[str] = None,
    sample_size: Optional[int] = None,
) -> str:
    """Compose le nom unique d'une collection.

    Forme : ``{base}[__{mode}]__{run_date}__{run_id}[__sample{N}]``

    `mode` porte le périmètre de la KB — 'full' (tous les produits annotés) ou
    'train' (l'ancien split, transitoire) — et n'est renseigné que pour les
    annotations, dont le contenu en dépend réellement. Les notices dérivent d'un
    CSV statique et n'ont pas de périmètre : leur en inventer un serait mensonger.

    `sample_size` ajoute un suffixe visible : sans lui, un index de test à 100
    points et un index complet portent des noms de même forme, et rien ne
    signale qu'on vient de recopier un jouet dans les paramètres de production.

    Les segments sont sûrs en chemin d'URL (Qdrant expose le nom dans ses URLs) :
    tirets et chiffres uniquement, jamais ':' ni '/'.
    """
    parts = [base]
    if mode:
        parts.append(mode)
    parts.extend([run_date, run_id])
    name = SEP.join(parts)
    if sample_size:
        name = f"{name}{SEP}sample{sample_size}"
    return name


def manifest_uri(manifests_root: str, collection_name: str) -> str:
    """Clé déterministe du manifeste : connaître le nom de la collection suffit
    à le retrouver, sans index ni catalogue. Un `aws s3 ls` trié par date
    répond à « quel est le dernier index ? »."""
    return f"{manifests_root.rstrip('/')}/{collection_name}.json"


def write_manifest(
    con: duckdb.DuckDBPyConnection,
    manifests_root: str,
    manifest: Dict[str, Any],
) -> str:
    """Écrit le manifeste sur S3 et renvoie son URI.

    Passe par DuckDB — dont la connexion porte déjà les identifiants et
    l'endpoint S3 — plutôt que par boto3 (absent des dépendances des deux
    modules) ou s3fs (qui exigerait un `AWS_S3_ENDPOINT` que les templates Argo
    d'indexation n'injectent pas).

    Le manifeste est volontairement **plat** : plus simple à relire, à grepper,
    et lisible depuis R si `classify-lcs` en a besoin un jour.
    """
    uri = manifest_uri(manifests_root, manifest["collection_name"])
    df = pd.DataFrame([manifest])
    con.register("_manifest", df)
    try:
        # Une ligne → un objet JSON. `read_json_auto` le relit tel quel.
        con.sql(f"COPY _manifest TO '{uri}' (FORMAT JSON)")
    finally:
        con.unregister("_manifest")
    return uri


def read_manifest(
    con: duckdb.DuckDBPyConnection,
    manifests_root: str,
    collection_name: str,
) -> Dict[str, Any]:
    """Relit le manifeste d'une collection. Lève si absent — c'est le cas
    d'un nom de collection inventé, ou d'un index bâti avant cette mécanique."""
    uri = manifest_uri(manifests_root, collection_name)
    rows = con.sql(f"SELECT * FROM read_json_auto('{uri}')").to_df()
    if rows.empty:
        raise ValueError(f"Manifeste vide : {uri}")
    record = rows.iloc[0].to_dict()
    # pandas remonte les manquants en NaN ; le JSON d'origine avait None.
    return {k: (None if pd.isna(v) else v) for k, v in record.items()}


def validate_collection(
    con: duckdb.DuckDBPyConnection,
    client_qdrant: Any,
    collection_name: Optional[str],
    manifests_root: str,
    expected_dim: int,
    expected_embedding_model: str,
    param_name: str,
    index_pipeline: str,
    expected_strategy: Optional[str] = None,
) -> Dict[str, Any]:
    """Vérifie qu'une collection est utilisable, AVANT tout travail coûteux.

    À appeler avant `mlflow.set_experiment` : échouer à l'intérieur d'un
    `mlflow.start_run` laisserait un run FAILED qui pollue l'expérience.

    Le précédent à ne pas reproduire est `reconcile-sirus-model-uri` : oublié,
    il fait échouer le pipeline après ~2 h sur un `argument --model-uri:
    expected one argument` qui ne dit ni ce qu'il fallait renseigner, ni où.
    Chaque message ci-dessous nomme le paramètre et le pipeline à lancer.

    Renvoie le manifeste, que l'appelant peut journaliser.
    """
    if not collection_name:
        raise ValueError(
            f"Paramètre `{param_name}` vide. Renseigner le nom de la collection "
            f"Qdrant à interroger, produit par `{index_pipeline}` (son log de fin "
            "affiche la ligne à recopier dans le fichier de paramètres)."
        )

    try:
        manifest = read_manifest(con, manifests_root, collection_name)
    except Exception as e:
        raise ValueError(
            f"Aucun manifeste pour la collection « {collection_name} » sous "
            f"{manifests_root}. Soit le nom est erroné, soit l'index a été bâti "
            f"avant la mise en place des manifestes : relancer `{index_pipeline}`. "
            f"({e})"
        ) from e

    if not client_qdrant.collection_exists(collection_name):
        raise ValueError(
            f"La collection « {collection_name} » a un manifeste mais n'existe pas "
            f"dans Qdrant — supprimée ? Relancer `{index_pipeline}`."
        )

    info = client_qdrant.get_collection(collection_name)

    # Vecteur anonyme dans les deux constructeurs ; on reste défensif au cas où
    # quelqu'un passerait un jour à des vecteurs nommés (le type devient un dict).
    vectors = info.config.params.vectors
    actual_dim = vectors.size if hasattr(vectors, "size") else vectors[""].size
    if actual_dim != expected_dim:
        raise ValueError(
            f"Dimension incompatible pour « {collection_name} » : la collection "
            f"est en {actual_dim}, ce run attend {expected_dim} "
            f"(embedding.model_len). Index bâti avec un autre modèle d'embedding."
        )

    actual_model = manifest.get("embedding_model")
    if actual_model != expected_embedding_model:
        raise ValueError(
            f"Modèle d'embedding incompatible pour « {collection_name} » : index "
            f"bâti avec « {actual_model} », ce run interroge avec "
            f"« {expected_embedding_model} ». Même dimension ne veut pas dire même "
            "espace vectoriel — les résultats seraient silencieusement faux."
        )

    if expected_strategy is not None:
        actual_strategy = manifest.get("strategy")
        if actual_strategy != expected_strategy:
            raise ValueError(
                f"Stratégie de découpage incompatible pour « {collection_name} » : "
                f"index bâti en « {actual_strategy} », ce run attend "
                f"« {expected_strategy} »."
            )

    count = client_qdrant.count(collection_name=collection_name, exact=True).count
    if count == 0:
        raise ValueError(
            f"La collection « {collection_name} » est vide. Indexation "
            f"interrompue ? Relancer `{index_pipeline}`."
        )

    status = str(getattr(info, "status", "")).lower()
    if "red" in status:
        raise ValueError(
            f"La collection « {collection_name} » est en statut RED côté Qdrant : "
            "inutilisable."
        )

    return {**manifest, "point_count_live": count}


def git_sha() -> Optional[str]:
    """SHA du commit exécuté, pour rapprocher un index du code qui l'a bâti.
    Les trois pipelines clonent une branche : sans ça, un écart de version
    entre l'indexation et la classification est invisible."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None


def log_paste_banner(logger, lines: Dict[str, str]) -> None:
    """Affiche les lignes de paramètres prêtes à recopier.

    Calqué sur `reconcile-sirus/main.py`, qui a établi la convention : la fin du
    job d'entraînement imprime littéralement ce qu'il faut coller dans le
    fichier de paramètres, plutôt que de laisser l'opérateur reconstituer un
    identifiant depuis les logs.
    """
    logger.info("=" * 72)
    logger.info("Index construit. Pour l'utiliser, mettre dans le fichier de")
    logger.info("paramètres du pipeline de classification :")
    for key, value in lines.items():
        logger.info("    %s: %s", key, value)
    logger.info("=" * 72)


def dumps(manifest: Dict[str, Any]) -> str:
    """Rendu lisible du manifeste pour les logs."""
    return json.dumps(manifest, indent=2, ensure_ascii=False, default=str)
