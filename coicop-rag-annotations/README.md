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
- `scripts/1_run_rag.py` — codifie l'input et exporte les prédictions. En mode
  **évaluation** (opt-in, `--skip-eval false`), calcule en plus accuracy + recall de
  retrieval par niveau COICOP (MLflow + `report.txt`). En **production** (défaut),
  prédictions seulement.
- `prompts/annotation_rag.md` — template de prompt (sections `<<<SYSTEM>>>` /
  `<<<USER>>>`). Source de vérité locale ; peut être migré vers Langfuse.
- `src/coicop_rag_annotations/` — utilitaires (embeddings, prunning, génération LLM
  concurrente, parsing, évaluation, chargement du prompt).

### Représentation texte enrichie par le lieu d'achat

Le code COICOP d'un produit dépend du **lieu d'achat** (« café » en supermarché ≠ « café »
au comptoir d'un bar). On indexe donc une représentation canonique unique
(`build_location_text`) qui combine produit + magasin + type de magasin quand on en
dispose — par ex. `café - magasin ou lieu d'achat : Super U (type : supermarché)`.

Cette **même** représentation sert partout : texte embeddé dans Qdrant, texte de la
requête (test), et exemples few-shot du prompt → l'espace d'embedding requête/index est
cohérent et le LLM voit le lieu d'achat dans les exemples. Quand l'info manque (ex. le
suggester), on retombe sur le seul libellé produit.

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
| `data.s3_path_input` | produits à codifier (test labellisé en éval, données prod sinon) |
| `data.s3_path_predictions` | sortie : prédictions du RAG |
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

# 1. Construire la base vectorielle (index = annotations train + suggester)
uv run scripts/0_build_annotation_vector_db.py --run-id <ID> --run-date <YYYY-MM-DD>

# 2a. PRODUCTION (défaut) : codifie l'input, exporte les prédictions, pas d'évaluation
uv run scripts/1_run_rag.py --run-id <ID> --run-date <YYYY-MM-DD>

# 2b. ÉVALUATION (opt-in) : idem + métriques vs labels (input labellisé requis)
uv run scripts/1_run_rag.py --run-id <ID> --run-date <YYYY-MM-DD> --skip-eval false
```

### Modes production / évaluation

`1_run_rag.py` produit **toujours** des prédictions (`data.s3_path_predictions`).
L'**évaluation** (accuracy + recall par niveau) est **opt-in** via `--skip-eval false`
— cohérent avec la convention `skip-report` de `argo/codif-pipeline.yaml` (paramètre
booléen string, testé par `!= "true"`). En production, l'input n'a pas de labels et
l'évaluation est ignorée (un garde-fou la saute aussi s'il manque la colonne `code`).

Le couple `--run-id` / `--run-date` doit correspondre à un run pour lequel l'input
(`data.s3_path_input`) et la table de mapping de prunning (`coicop.path_mapping_lvl4`)
existent déjà sur S3.

## Prérequis upstream

- annotations **train** sur S3 (sortie de `codif-regex`) → indexées ;
- input à codifier (`data.s3_path_input`) : test pruné labellisé en mode évaluation,
  données à codifier en production ;
- table de mapping de prunning niveau 4 (produite par `0_prunning_coicop.py` de `coicop-rag`) ;
- source du suggester (CSV `liste_produits_fr_copain.csv`) si `suggester.enabled`.

## Rapport d'évaluation (mode `--skip-eval false`)

Loggé dans MLflow (scalaires `log_metrics` + tables `log_table` + `report.txt`) :

- **Accuracy / recall par niveau** (1..`eval.levels`) sur `all_parsed` et `parsed_and_codable`.
- **Accuracy par catégorie COICOP niveau 1** (`eval/accuracy_by_level1.json`) : support et
  précision par grande catégorie, au niveau 1 et au niveau cible.
- **Accuracy par source des données de test** (accuracy globale + niveau 1 par `source`) —
  **uniquement dans `report.txt`** (pas en scalaire ni en table MLflow).
- **Distorsion de distribution** niveaux 1 et 2 (`eval/distribution_level_{1,2}.json`) :
  parts vraies vs prédites par catégorie + `diff`, et indicateurs agrégés **TV distance**
  et **KL** (détecte les sur/sous-prédictions systématiques d'une catégorie).
- **Fiabilité de la confiance LLM** : **AUROC** confiance↔justesse, courbe de **calibration**
  (accuracy par bin), et **balayage de seuils** (couverture vs accuracy si on filtre sur
  `confidence ≥ t`) → indique si la confiance permet d'écarter les mauvaises prédictions.
- **Fiabilité du flag `codable`** : accuracy et couverture pour `codable` True/False/manquant,
  et **lift** d'accuracy si on ne garde que `codable=True`.

## Limites de cette première version

- Le texte du suggester est utilisé tel quel (pas la normalisation `l_pr_product`
  du preprocessing) ; cohérence des **codes** garantie, pas du prétraitement texte.
- Embeddings recalculés à chaque run (pas de cache).
- Prompt local par défaut ; le pousser dans Langfuse pour le versionner.
