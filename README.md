# codif-coicop-bdf

Pipeline d'orchestration pour la codification automatique des produits de l'enquête Budget de Famille (BDF) selon la nomenclature COICOP.

## Pipeline

Le projet compte **trois workflows Argo**. La construction des bases vectorielles, coûteuse et
rarement à refaire, est sortie du pipeline de classification : elle a ses propres workflows, et
les collections produites sont désignées par paramètre au lancement — même organisation que
l'entraînement de `classify-ttc` et de `reconcile-sirus`.

```
① argo/index-notices-pipeline.yaml
   prune-codes (--only nomenclature) ─→ index-notices          → collection Qdrant, nom unique

② argo/index-annotations-pipeline.yaml
   build-datasets ─→ prune-codes (--only kb) ─→ index-annotations   → collection Qdrant, nom unique

        └── les deux noms sont recopiés dans argo/params.yaml ──┐
                                                                ▼
③ argo/codif-pipeline.yaml   (input_file : OBLIGATOIRE — un seul mode)
   Le DAG ci-dessous tourne DEUX FOIS : smoke sur 100 lignes (~8 min), puis pour de vrai.
   Si le smoke échoue, le vrai run ne démarre pas. Échappatoire : -p skip-smoke=true

build-datasets
  └─→ classify-regex ─┬─→ classify-lcs ─────────────────────────────┐
                      ├─→ classify-ttc ─────────────────────────────┤
                      └─→ prune-codes ─┬─→ classify-rag-notices ────┤
                                       └─→ classify-rag-annotations ┘
                                                                    │
                                       ┌────────────────────────────┘
                                       │  les 4 classifieurs convergent
                                       └─→ reconcile-llm  OU  reconcile-sirus   (exclusifs : paramètre `reconciliation`)
                                               └─→ export-results ─→ report  (skip-report)
                                                          └─→ evaluate  (facultative : label-column)
```

Le pipeline ③ n'a **qu'un seul mode** : il code le fichier désigné par `input_file`, qui est obligatoire. L'étape **`prune-codes`** centralise tout le pruning et produit les artefacts lus par l'aval.

Il y avait auparavant deux modes, dérivés du seul fait qu'`input_file` soit vide ou non, et testés à treize endroits. Cette dualité venait de ce que le pipeline avait été construit **pour être évalué**, à une époque où il n'existait qu'un jeu annoté : il fallait le couper en deux pour mesurer sans fuite. Des données fraîches annotées arrivent désormais régulièrement, donc toute la base historique sert de KB sans découpage, et **l'évaluation devient une opération d'après-coup** : l'étape facultative `evaluate`, qui ne tourne que si le fichier d'entrée porte une colonne d'étiquettes (`label-column`).

② n'a **pas** de paramètre `input_file` — celui-ci désigne les produits *à classer*, ce qui ne concerne pas la constitution d'une base de produits déjà annotés.

### build-datasets

Construit le dataset d'annotations à partir des sources brutes (COPAIN, historique, suggester).

- Code dans [`build-datasets/`](./build-datasets/) (ex-repo `construction-dataset`)
- Exporte le dataset consolidé sur S3

### prune-codes

Étape **unique** de pruning (troncature niveau 4 + élagage des hiérarchies linéaires → code canonique). Produit tous les artefacts prunés sous `…/{run}/prune-codes/`, lus par l'aval.

- Code dans [`prune-codes/`](./prune-codes/) (`scripts/main.py`)
- Sorties : `nomenclature_pruned`, `mapping_lvl4`, `annotations_train_pruned` (la KB annotée), `annotations_test_pruned` (le jeu à coder), `suggester_pruned`. Les suffixes `train`/`test` sont hérités du split supprimé : ils désignent aujourd'hui la KB et le jeu à coder.

### index-notices *(workflow ① — hors pipeline de classification)*

Encode les notices COICOP **prunées** dans une base vectorielle Qdrant.

- Code dans [`rag-notices/`](./rag-notices/) (`scripts/0_create_vector_db.py`)
- Lit `prune-codes/nomenclature_pruned.parquet` ; embeddings via VLLM, index Qdrant
- Autonome : la nomenclature dérive d'un CSV statique, donc `prune-codes --only nomenclature` suffit — ni `build-datasets` ni `classify-regex`

### index-annotations *(workflow ② — hors pipeline de classification)*

Encode la **KB d'annotations** (+ suggester) dans une vector DB Qdrant, pour la RAG sur exemples annotés.

- Code dans [`rag-annotations/`](./rag-annotations/) (`scripts/0_build_annotation_vector_db.py`)
- La KB, ce sont les **produits déjà annotés** : `annotations_full` + `suggester` au sens de `build-datasets`, prunés. `classify-regex` n'est pas dans la chaîne — la KB n'a pas à être filtrée des produits que la regex sait coder
- Il y avait ici une option `kb-scope` (`full` / `train`), du temps où la KB était un demi-jeu. Ce split n'existait que faute de jeu de test indépendant ; les nouveaux produits annotés en fournissent un, donc toute la base historique sert de KB. Supprimée

### Les collections Qdrant

Chaque indexation crée une collection au **nom unique** — `{base}__{run_date}__{run_id}[__sampleN]` — et un manifeste JSON à côté (modèle d'embedding, dimension, stratégie, nombre de points, sha git), que les étapes `classify-rag-*` relisent au démarrage pour valider ce qu'elles interrogent.

Auparavant deux noms fixes (`coicop_lineage`, `coicop_annotations_without_copain_2017`) étaient partagés par tous les runs et détruits/recréés à chaque indexation : une réindexation cassait la base que lisait un run concurrent.

### classify-regex

Codification des libellés produits par approche regex.

- Code dans [`classify-regex/`](./classify-regex/) (ex-repo `regex_codif`)

### classify-lcs

Codification des libellés produits par approche LCS (Longest Common Subsequence) en R.

- Code dans [`classify-lcs/`](./classify-lcs/)

### classify-rag-notices

Code le jeu à coder via un RAG sur les **notices** de la nomenclature COICOP.

- Code dans [`rag-notices/`](./rag-notices/) (`scripts/2_run_rag.py`)
- Récupère les notices proches depuis Qdrant, génère via VLLM (`VLLM_GENERATION_URL`)
- Paramètres et compteurs MLflow (`MLFLOW_TRACKING_URI`), traces Langfuse (`LANGFUSE_BASE_URL`). Il ne calcule plus d'accuracy : c'est `evaluate` qui mesure

### classify-rag-annotations

Code le jeu à coder via un RAG sur des **exemples déjà annotés** (few-shot).

- Code dans [`rag-annotations/`](./rag-annotations/) (`scripts/1_run_rag.py`)
- Récupère les annotations proches depuis Qdrant. Prédictions seules : les métriques sont calculées par `evaluate`

### classify-ttc

Prédictions TTC via un classifieur pré-entraîné.

- Code dans [`classify-ttc/`](./classify-ttc/) (ex-repo `coicop_bdf_classifier`, bientôt archivé en amont)
- L'étape argo utilise actuellement l'image pré-construite `ghcr.io/micedre/coicop_bdf_classifier:latest`
- Étape destinée à être supprimée à terme

### reconcile-llm

Arbitrage final des prédictions par un LLM-as-judge : fusionne les sorties de `classify-lcs`, `classify-rag-notices`, `classify-rag-annotations` et `classify-ttc` et sélectionne le meilleur code COICOP par observation.

- Code dans [`reconcile-llm/`](./reconcile-llm/) (`uv run main.py reconcile-llm`)
- Entrées :
  - `s3://.../classify-lcs/raw_test_LCS.parquet`
  - `s3://.../classify-rag-notices/predictions.parquet`
  - `s3://.../rag-annotation/predictions.parquet`
  - `s3://.../classify-ttc/predictions.parquet`
- Sortie : `s3://.../reconcile-llm/predictions.parquet`
- Utilise un endpoint OpenAI-compatible (`LLMLAB_API_KEY`, optionnellement `LLMLAB_URL` pour un backend non-OpenAI)
- Court-circuit consensus : si les trois sources convergent (et que la confiance TTC ≥ 0.90), aucune requête LLM n'est émise
- Filtrage de nomenclature : seules les sections COICOP pertinentes sont envoyées au prompt (réduction ×4–10 du nombre de tokens)
- Reprise automatique : relancer l'étape avec le même `run_id`/`run_date` reprend les observations non traitées depuis le fichier de sortie existant

### report

Rapport **de production** Quarto (HTML auto-contenu) sur la sortie de conciliation. Il ne suppose **aucune vérité terrain** : il décrit ce que le pipeline a produit, sans le noter.

- Code dans [`report/`](./report/) (`prediction_report.qmd`) — Quarto + Python (pandas, duckdb, matplotlib, seaborn)
- **Activé par défaut** : le YAML pose `skip-report: "false"`, c'est-à-dire « ne pas sauter ». Passer `-p skip-report=true` pour s'en dispenser
- Entrée : `s3://.../reconcile-{llm,sirus}/predictions.parquet`
- Sortie : `s3://.../report/report.html`
- Contenu : volumétrie et couverture, profondeur des codes prédits, accord/désaccord des quatre classifieurs, distribution des confiances, durée de chaque étape

### evaluate *(facultative)*

Mesure la qualité du run. **Ne tourne que si le fichier d'entrée portait une colonne d'étiquettes** (`label-column`) — sinon la tâche est `Skipped`, ce qui est le cas nominal de production.

- Code dans [`evaluate/`](./evaluate/) (`main.py` + `evaluation_report.qmd`)
- Déclenchement : `-p label-column=code`. `-p skip-eval=true` la saute malgré tout
- Entrées — **trois** artefacts, pas un :
  - `s3://.../reconcile-{llm,sirus}/predictions.parquet` — les 4 classifieurs et la conciliation
  - `s3://.../classify-rag-notices/retrieved_codes.parquet` et `s3://.../classify-rag-annotations/predictions.parquet` — le **recall de retrieval** : le seul indicateur qui dise si un RAG échoue à *retrouver* ou à *générer*
  - le livrable d'`export-results` — l'**accuracy de bout en bout, regex comprise** : le chiffre métier. Le parquet de conciliation ne l'a pas, car les lignes captées par la regex n'entrent jamais dans la chaîne
- Sorties : `s3://.../evaluate/evaluation_report.html` **et** les métriques dans MLflow
- Contenu : accuracy par niveau COICOP (1 à 4) pour **LCS, RAG notices, RAG annotations, TTC, conciliation**, selon les deux conventions (stricte et inclusive) ; accuracy par `shop`, `shop_type_name` et quartile de `budget` ; matrice de confusion ; calibration (accuracy par bucket de confiance, AUROC) ; coût et latence de l'arbitrage LLM. Avec `-p eval-source-column=…`, une ventilation **par provenance du produit**
- Méthodologie *accuracy par niveau* : tronquer la vérité et la prédiction aux `k` premiers segments ; les observations dont la vérité a moins de `k` niveaux sont exclues du dénominateur à ce niveau
- Elle **échoue** si la vérité canonique `code_lvl4` est absente, plutôt que de se rabattre sur `code` : comparer des prédictions canoniques à une vérité brute sous-estime l'accuracy sur près d'un quart des postes

## Contrôles automatiques

Quatre vérifications mécaniques tournent à chaque envoi sur GitHub
([`.github/workflows/checks.yml`](./.github/workflows/checks.yml)), en moins d'une minute.
Elles répondent à une seule question : *quelqu'un a-t-il renommé quelque chose en oubliant
un endroit ?* Les mêmes en local :

```bash
uv lock --check                                    # verrou à jour (sinon TOUTES les étapes Argo échouent)
uv run --with ruff ruff check .                    # nom utilisé sans être déclaré
uv run --with pyyaml python scripts/check_pipeline.py .   # cohérence des workflows et des paramètres
argo lint --offline argo/codif-pipeline.yaml       # validité du schéma Argo
```

Ce ne sont **pas** des tests métier : elles ne disent rien de la qualité de la
codification. Elles attrapent le code ou la configuration qui *ne peut pas fonctionner* —
un import manquant, une clé de paramètre mal orthographiée qu'`argo submit` accepterait en
silence, un `pyproject.toml` modifié sans relancer `uv lock`.

## Structure du dépôt

Ce dépôt rassemble le code de toutes les étapes du pipeline, auparavant dispersé dans plusieurs repos.

| Dossier | Origine | Rôle |
|---|---|---|
| [`argo/`](./argo/) | — | Workflows Argo : `codif-pipeline.yaml` (classification), `index-notices-pipeline.yaml` et `index-annotations-pipeline.yaml` (bases vectorielles), `ttc-pipeline.yaml`, `rbac.yaml` |
| [`build-datasets/`](./build-datasets/) | `construction-dataset` | Étape `build-datasets` |
| [`classify-regex/`](./classify-regex/) | `regex_codif` | Étape `classify-regex` |
| [`prune-codes/`](./prune-codes/) | — | Étape `prune-codes` (pruning unifié) |
| [`rag-notices/`](./rag-notices/) | `coicop-rag` | Étape `index-notices` (workflow ①) et `classify-rag-notices` (workflow ③) |
| [`rag-annotations/`](./rag-annotations/) | — | Étape `index-annotations` (workflow ②) et `classify-rag-annotations` (workflow ③) |
| [`classify-lcs/`](./classify-lcs/) | `stats-annotations` | Étape `classify-lcs` (R) |
| [`classify-ttc/`](./classify-ttc/) | `coicop_bdf_classifier` | Étape `classify-ttc` |
| [`reconcile-llm/`](./reconcile-llm/) | — | Étape `reconcile-llm` (arbitrage LLM) |
| [`export-results/`](./export-results/) | — | Étape `export-results` (livrable utilisateur) |
| [`report/`](./report/) | — | Rapport Quarto **de production** (étape `report`, activée par défaut) |
| [`evaluate/`](./evaluate/) | — | Rapport Quarto **d'évaluation** (étape `evaluate`, facultative) |
| [`common/`](./common/) | — | Socle partagé (`codif_common`) : registre d'artefacts, métriques, chemins S3. Aucune étape Argo |

Chaque sous-dossier Python conserve son propre `pyproject.toml` (ses dépendances lui
appartiennent), mais tous sont membres d'un même **workspace `uv`**.

## Environnement Python

Un `pyproject.toml` à la racine déclare les 10 modules Python comme membres d'un workspace, ce
qui donne **un seul `uv.lock`** pour tout le dépôt : une seule version de `pandas`, `duckdb`,
`pyarrow`… partagée par toutes les étapes. C'est nécessaire parce que les étapes se passent des
Parquet : avec un lock par module, `classify-regex` écrivait en pandas 3 ce que `classify-ttc` relisait
en pandas 2.

```bash
# Travailler sur un module : n'installe que SES dépendances, aux versions du lock racine
cd prune-codes/ && uv sync --locked
uv run scripts/main.py --help

# Ajouter une dépendance à un module, depuis n'importe où dans le dépôt
uv add --package prune polars

# Faire monter un paquet pour TOUT le dépôt (le lock est commun)
uv lock --upgrade-package duckdb
```

L'environnement (`.venv`) vit à la racine du workspace et est reconstruit par chaque `uv sync` :
enchaîner deux modules est normal et rapide. `--locked` fait échouer la commande si le lock ne
correspond plus aux `pyproject.toml` — c'est ce que fait le pipeline, plutôt que de re-résoudre
les dépendances en silence au démarrage d'une étape.

`classify-lcs/` (R) n'est pas concerné : ses dépendances sont installées par `R/main.R`.

Python ≥ 3.13 partout (`.python-version` à la racine).

## Lancer le workflow

### Prérequis

Le secret Kubernetes `secret-codif-coicop-bdf` doit exister dans le namespace et contenir les clés suivantes :

```
AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN,
AWS_S3_ENDPOINT, AWS_ENDPOINT_URL,
VLLM_EMBEDDING_URL, VLLM_EMBEDDING_API_KEY,
VLLM_GENERATION_URL, VLLM_GENERATION_API_KEY,
QDRANT_URL, QDRANT_API_KEY, QDRANT_API_PORT,
LANGFUSE_BASE_URL, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY,
MLFLOW_TRACKING_URI, MLFLOW_TRACKING_USERNAME, MLFLOW_TRACKING_PASSWORD,
OLLAMA_URL, OLLAMA_API_KEY,
DDC_ENCRYPTION_KEY,
LLMLAB_API_KEY, LLMLAB_URL   # requis pour reconcile-llm (LLMLAB_URL optionnel)
```

### Avec la CLI Argo

**Préalable, une fois** : construire les deux bases vectorielles. Chaque workflow affiche en fin
d'exécution la ligne exacte à recopier dans [`argo/params.yaml`](./argo/params.yaml).

```bash
argo submit argo/index-notices-pipeline.yaml --watch
argo submit argo/index-annotations-pipeline.yaml --watch
```

Puis la classification :

```bash
# Pipeline complet (les noms de collections viennent de params.yaml)
argo submit argo/codif-pipeline.yaml --parameter-file argo/params.yaml

# Sur un autre fichier à coder
argo submit argo/codif-pipeline.yaml --parameter-file argo/params.yaml \
  -p input_file=s3://.../workflow_inputs/mon_fichier.csv \
  -p text_column=NAT_DEP -p shop_column=MAG_DEP -p budget_column=MONT_DEP

# Limiter le volume pour tester (sampling centralisé à classify-regex, hérité par tous
# les classifieurs)
argo submit argo/codif-pipeline.yaml --parameter-file argo/params.yaml -p sample-observations=100

# Mesurer le run : uniquement si le fichier d'entrée porte une colonne d'étiquettes
argo submit argo/codif-pipeline.yaml --parameter-file argo/params.yaml \
  -p label-column=code -p eval-source-column=source

# Modèle spécifique pour classify-rag-notices
argo submit argo/codif-pipeline.yaml --parameter-file argo/params.yaml -p classify-rag-model=openai/gpt-oss-120b

# Désactiver le rapport de production (activé par défaut dans le YAML)
argo submit argo/codif-pipeline.yaml --parameter-file argo/params.yaml -p skip-report=true
```

> `argo submit` accepte **en silence** un nom de paramètre inconnu : une faute de frappe dans un
> `-p` ne produit ni erreur ni effet.

Voir aussi la fiche [`argo/argo_helper.md`](./argo/argo_helper.md) et le fichier de paramètres [`argo/params.yaml`](./argo/params.yaml).

### Paramètres disponibles

| Paramètre | Défaut | Description |
|---|---|---|
| `input_file` | *(csv BDF)* | **Obligatoire.** Le fichier à coder. Le pipeline n'a plus qu'un mode. |
| `sample-observations` | *(vide)* | Plafonne le jeu à coder. Échantillonné **une fois** à `classify-regex`, hérité par tous les classifieurs. Pour plafonner la KB indexée, c'est `kb-sample-size` du workflow ②. |
| `classify-rag-notices-collection` | *(vide)* | **Obligatoire.** Collection Qdrant produite par `index-notices-pipeline.yaml`. |
| `classify-rag-annotations-collection` | *(vide)* | **Obligatoire.** Collection Qdrant produite par `index-annotations-pipeline.yaml`. |
| `classify-rag-model` | `gemma4-26b-moe` | Modèle LLM pour `classify-rag-notices` / `classify-rag-annotations` |
| `reconcile-llm-model` | `gemma4-26b-moe` | Modèle LLM utilisé par `reconcile-llm` |
| `reconcile-llm-concurrency` | `5` | Nombre d'appels LLM parallèles de `reconcile-llm` |
| `skip-report` | `false` | Si `false`, génère le rapport de production après `export-results` |
| `label-column` | *(vide)* | Nom de la colonne d'étiquettes dans `input_file`. **Vide = pas d'évaluation** : l'étape `evaluate` est sautée. Non vide, `build-datasets` la recopie dans `code`, qui la porte jusqu'au bout de la chaîne. |
| `eval-source-column` | *(vide)* | Nom de la colonne de provenance du produit. Ajoute une ventilation de l'accuracy par source au rapport d'évaluation. Ne restreint **jamais** le périmètre codé. |
| `skip-eval` | `false` | `true` saute l'étape `evaluate` même si `label-column` est renseignée. |
| `eval-experiment` | `codif-coicop-eval` | Expérience MLflow de l'étape `evaluate`. |
