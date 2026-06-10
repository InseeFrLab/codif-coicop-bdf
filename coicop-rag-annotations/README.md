# coicop-rag-annotations

RAG de codification COICOP **basé sur des exemples annotés** (descriptions de
produits déjà codifiées), par opposition à [`coicop-rag`](../coicop-rag/) qui
récupère des **notices** de la nomenclature.

Idée : pour coder un produit, on récupère dans une base vectorielle les produits
*déjà annotés* les plus proches sémantiquement, et on fournit ces couples
`(description → code)` comme exemples few-shot à un LLM.

## Architecture

```
0_build_annotation_vector_db.py        1_run_rag.py
   import annotations TRAIN                charge le test set (amont, prune-annotations)
   prune (niveau 4)                        embed + retrieve (Qdrant)
   (+ suggester) → index  ───────────►     prompt few-shot → LLM
   embed → Qdrant                          parse → évaluation (accuracy/recall par niveau)
```

- `scripts/0_build_annotation_vector_db.py` — importe les annotations **de train**
  (le split train/test est déjà fait en amont par `preprocessing`), prune les codes
  au niveau 4, ajoute optionnellement les exemples du **suggester** à l'index, embed
  **toutes** les descriptions et les charge dans Qdrant. Pas de split ici, pas de
  test produit. Le suggester passe par le **même** `prune_annotation_lvl4`.
- `scripts/1_run_rag.py` — exécute le RAG sur le **test d'évaluation produit en
  amont** (`prune-annotations`, déjà pruné) et évalue (accuracy + recall de retrieval
  par niveau COICOP, exportés dans MLflow + `report.txt`).
- `prompts/annotation_rag.md` — template de prompt (sections `<<<SYSTEM>>>` /
  `<<<USER>>>`). Source de vérité locale ; peut être migré vers Langfuse.
- `src/coicop_rag_annotations/` — utilitaires (embeddings, prunning, génération LLM
  concurrente, parsing, évaluation, chargement du prompt).

## Configuration

Tout est dans `config/config.yaml`. Paramètres clés :

| Clé | Rôle |
|---|---|
| `annotations.s3_path` | annotations **train** à indexer (sortie de codif-regex) |
| `annotations.product_col` | colonne texte à embedder (`l_pr_product`) |
| `suggester.enabled` | ajouter les exemples du suggester à l'index |
| `suggester.s3_path` + `*_col` | source du suggester et noms de colonnes |
| `qdrant.collection_name` | collection Qdrant (`coicop_annotations`) |
| `retrieval.size` | nombre d'exemples annotés récupérés par produit |
| `llm.use_langfuse` | `false` → prompt local ; `true` → Langfuse |
| `eval.s3_path_test` | jeu de test d'évaluation (pruné, produit en amont) |
| `eval.levels` | profondeur d'évaluation (niveaux 1..levels) |

## Exécution (Onyxia, secrets via Vault)

Variables d'environnement attendues (mêmes que `coicop-rag`, injectées par Vault) :
`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `LLMLAB_URL`, `LLMLAB_API_KEY`,
`QDRANT_URL`, `QDRANT_API_KEY`, `QDRANT_API_PORT`, `MLFLOW_TRACKING_URI`,
`MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD` (+ `LANGFUSE_*` si
`use_langfuse: true`).

```bash
cd coicop-rag-annotations
uv sync
mkdir -p logs

# 1. Construire la base vectorielle + figer le test set
uv run scripts/0_build_annotation_vector_db.py --run-id <ID> --run-date <YYYY-MM-DD>

# 2. Lancer le RAG et évaluer (sur le test set held-out)
uv run scripts/1_run_rag.py --run-id <ID> --run-date <YYYY-MM-DD> --sample_size 100
```

Le couple `--run-id` / `--run-date` doit correspondre à un run pour lequel les
annotations brutes (`annotations.s3_path`) et la table de mapping de prunning
(`coicop.path_mapping_lvl4`) existent déjà sur S3.

## Prérequis upstream

- annotations **train** sur S3 (sortie de `codif-regex`) → indexées ;
- jeu de **test pruné** pour l'évaluation (`prune-annotations`, `eval.s3_path_test`) ;
- table de mapping de prunning niveau 4 (produite par `0_prunning_coicop.py` de `coicop-rag`) ;
- source du suggester (CSV `liste_produits_fr_copain.csv`) si `suggester.enabled`.

## Limites de cette première version

- Le texte du suggester est utilisé tel quel (pas la normalisation `l_pr_product`
  du preprocessing) ; cohérence des **codes** garantie, pas du prétraitement texte.
- Embeddings recalculés à chaque run (pas de cache).
- Évaluation volontairement compacte (accuracy + recall par niveau) ; le
  reporting riche de `coicop-rag` n'est pas repris.
- Prompt local par défaut ; le pousser dans Langfuse pour le versionner.
