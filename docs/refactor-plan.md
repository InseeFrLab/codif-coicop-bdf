# Plan de refactor — pipeline codif-coicop-bdf

> **Document de travail, non publié.** Le projet Quarto ne rend que les `.qmd` listés dans
> `_quarto.yml` ; ce fichier est en `.md` et explicitement exclu. Il n'apparaît pas sur le site.
>
> Dernière mise à jour : 2026-09-01. À maintenir au fil des chantiers.

## Pourquoi ce document

Le dépôt a été consolidé en monorepo (voir `archive/refactor-plan.md` pour cette étape,
terminée). L'objectif de la campagne en cours est différent : amener le pipeline à des
**standards de production** — reproductible, lisible, avec un contrat explicite entre étapes —
sans sacrifier la facilité de prise en main.

Cinq chantiers ont été identifiés à partir d'un audit du dépôt. Deux sont faits, trois restent.
Ce document sert à reprendre le travail dans une autre session ou par une autre personne.

---

## État d'avancement

| Chantier | Objet | État |
|---|---|---|
| 3 | Cohérence des dépendances (workspace uv) | **fait** — commit `b40d349` |
| 5a | Vocabulaire unique des étapes, modules, paquets, paramètres | **fait** — commit `3ccacf8` |
| 1 | Reproductibilité de l'exécution | partiel |
| 5b | YAML déclaratif, homogénéité des modules | à faire |
| 2 | Contrat de données entre étapes | à faire |
| 4 | CI et tests | à faire |

### Vocabulaire en vigueur depuis `3ccacf8`

Le nom d'une étape est aussi le **dossier S3** où elle écrit, le **dossier du module**, et le
nom de son projet Python. Une seule chaîne à retenir par étape.

| Étape / dossier / projet | Rôle |
|---|---|
| `build-datasets` | construit annotations (full/train/test), suggester, observations, QA |
| `classify-regex` | codification par règles regex ; **sampling centralisé** ici |
| `prune-codes` | définit l'espace de codes canonique et y reprojette tout |
| `classify-lcs` (R) | plus longue sous-chaîne commune |
| `index-notices` / `index-annotations` | indexation Qdrant (skippables via `skip-index`) |
| `classify-rag-notices` / `classify-rag-annotations` | RAG sur notices / sur exemples annotés |
| `classify-ttc` | classifieur neuronal (modèle chargé depuis MLflow) |
| `reconcile-llm` / `reconcile-sirus` | conciliation, **exclusives** (paramètre `reconciliation`) |
| `export-results` | livrable utilisateur |
| `report` | rapport Quarto d'exactitude (opt-in) |

Paramètres Argo alignés : `classify-rag-model`, `classify-ttc-model-uri`,
`reconcile-llm-model`, `reconcile-llm-concurrency`, `reconcile-sirus-model-uri`, `skip-index`,
`reconciliation`.

**Épargnés volontairement** : les noms d'**expériences MLflow** (`rag-annotation`,
`codif-coicop-eval`, `test`) et le paramètre `report-experiment` qui les porte — l'historique des
métriques y vit ; le paquet R `sirus` ; les colonnes de données (`llm_code`, `sirus_proba`…) ;
la colonne « Origine » du `README.md`, qui documente les anciens dépôts.

**Conséquence à connaître** : les runs antérieurs à `3ccacf8` ont des dossiers S3 aux anciens
noms (`codif-lcs/`, `run-rag/`, `rag-annotation/`, `final-output/`…). Le code courant ne sait
plus les lire. Pour rejouer un rapport sur un run ancien : `git checkout b40d349`. Les noms de
métriques MLflow changent aussi (`duration_decide_coicop_seconds` →
`duration_reconcile_llm_seconds`), donc les séries historiques se coupent là.

---

## Chantier 1 — Reproductibilité de l'exécution *(partiel)*

**Acquis.** Les 22 invocations `uv` des workflows Argo utilisent `uv sync --locked --no-dev` :
une étape ne peut plus re-résoudre ses dépendances en silence au démarrage, et n'installe plus
ruff/pytest en production.

**Ce qui reste.** Les 13 steps font toujours `git clone --branch main` au runtime. Trois
conséquences :

1. un run n'est pas rejouable à l'identique (la branche bouge) ;
2. deux steps d'un même run peuvent exécuter deux versions du code — chaque step clone à *son*
   démarrage, et un pipeline dure ~2 h ;
3. `classify-lcs/R/main.R` fait **11 `install.packages()` non épinglés** depuis CRAN, à chaque
   exécution, en plein pipeline. C'est le point le plus fragile du dépôt.

**Deux options, par ordre d'ambition.**

- *Épingler le clone sur un commit* : remplacer le paramètre `git-branch` par `git-sha`. Petit,
  sans CI, règle (1) et (2). Ne règle ni la lenteur ni les `install.packages`.
- *Images construites en CI* : une image par module, taguée par SHA, poussée sur un registre
  (GHCR : le `GITHUB_TOKEN` du dépôt suffit à pousser, mais les paquets doivent être publics ou
  un `imagePullSecret` configuré côté cluster). Le patron existe déjà :
  `classify-ttc/Dockerfile` (multi-stage, `uv sync --frozen --no-dev`, venv recopié) — écrit
  mais jamais utilisé par le pipeline. Contexte de build = racine du dépôt (4 modules
  dépendent d'un module frère), `.dockerignore` obligatoire, et `data/` à exclure des images
  (vérifié : seules les commandes d'entraînement de `classify-ttc` le lisent).
  Contrepartie assumée : le cycle de test passe de « push + submit » à « push + attendre le
  build (3-5 min) + submit ». Faire construire les images **aussi sur les branches**.

**Décision utilisateur (2026-09-01)** : les images sont écartées pour le moment.

---

## Chantier 2 — Contrat de données entre étapes *(à faire — le plus rentable pour la lisibilité)*

### Le problème, mesuré

Chaque consommateur redéclare le chemin du producteur. `mapping_lvl4.parquet`, produit par
`prune-codes`, voit son chemin écrit **7 fois hors documentation** (`prune-codes/config.yaml`,
`rag-notices/config.yaml`, `export-results/main.py`, deux lignes du YAML Argo, le bilan) — et
par deux mécanismes différents selon le consommateur : argument CLI pour `reconcile-llm`, clé de
config pour `rag-notices`. Le nom du bucket est en dur à **76 endroits**.

Coût constaté : le renommage des étapes a demandé de modifier **162 segments de chemin dans
29 fichiers**.

Le même code est écrit plusieurs fois sans que ce soit un choix :

| Fonction | Copies |
|---|---|
| `expand_paths` | 4 (`prune_codes`, `classify-regex`, `rag_annotations`, `rag_notices`) |
| `create_duckdb_connection` | 3 |
| `truncate_code` | 4 |
| `setup_logging` | 3, et 2 modules sans logger (`report`, `export-results` utilisent `print`) |
| bloc `CREATE OR REPLACE SECRET` DuckDB (~20 lignes) | 5 fichiers |

### Brique 1 — le registre des artefacts

Un fichier de déclarations à la racine, **en YAML et non en Python** : `classify-lcs` est en R et
doit pouvoir le lire.

```yaml
# contracts.yaml
bucket: projet-budget-famille          # surchargeable par COICOP_BUCKET
layout: data/workflow_runs/{run_date}/{run_id}/{step}/

steps:
  prune-codes:
    outputs:
      mapping_lvl4: mapping_lvl4.parquet
      nomenclature: nomenclature_pruned.parquet
      annotations_train: annotations_train_pruned.parquet
      annotations_test: annotations_test_pruned.parquet
      suggester: suggester_pruned.parquet
  classify-lcs:
    inputs:  [classify-regex.test_without_regex]
    outputs: {predictions: raw_test_LCS.parquet}
```

Accès depuis Python (`artifact(run, "prune-codes", "mapping_lvl4")`) et depuis R. Les
`config.yaml` des modules cessent de porter des URI : ils référencent des clés.

Gains : renommer une sortie devient une ligne ; « qui consomme quoi » se lit au lieu de se
grepper ; et le repli « si ce dossier n'existe pas, essaie l'ancien nom » — nécessaire pour
relire les runs d'avant `3ccacf8` — s'écrit **une fois** au lieu de dix.

### Brique 2 — la plomberie dans `common/`

Nouveau membre du workspace (le patron existe : `prune-codes` est déjà une dépendance
`workspace = true` de trois modules). Il porte la connexion DuckDB/S3, la lecture/écriture
Parquet, `expand_paths`, `truncate_code`, le setup du logging. Les 5 copies du bloc de secrets
et les 4 copies d'`expand_paths` disparaissent.

### Brique 3 — validation de schéma aux frontières

Chaque étape déclare les colonnes qu'elle lit et écrit ; échec immédiat et lisible sinon.
Deux cas vécus qui le justifient :

- `report.qmd` casse sur `KeyError: "None of ['sources muettes'] are in the columns"` quand on le
  rend sur des données de production — au milieu du rapport, loin de la cause ;
- `truth_column()` choisit silencieusement `code_lvl4` ou, à défaut, `code` : deux sémantiques,
  un repli muet.

### Ordre proposé

1. `common/` + registre, puis migration de **deux modules seulement** (`export-results` et
   `report`, ceux qui dupliquent le bloc de secrets) — périmètre vérifiable en local.
2. Les 4 `config.yaml` qui recopient des URI, un par un.
3. La validation de schéma, en commençant par l'entrée du `report` et `classify-lcs` →
   `reconcile-*`.

**Préalable fortement recommandé** : le smoke run du chantier 4. Sans lui, ce chantier modifie
les chemins des 10 étapes sans filet.

---

## Chantier 4 — CI et tests *(à faire)*

### Où on en est

Un seul workflow GitHub (`publish-website.yml`, publication du site). Aucun lint, aucun test
exécuté, aucune config `[tool.ruff]` / `[tool.pytest]` — alors que ruff est déjà en dépendance de
développement de plusieurs modules. Aucun `CronWorkflow` Argo.

Modules sans aucun test : `build-datasets`, `classify-regex`, `rag-annotations`,
`export-results`, `classify-lcs`.

### Le partage des rôles CI / smoke run

La CI est une **barrière** (elle empêche un commit cassé d'entrer dans `main`) ; le smoke run est
une **alarme** (il signale après coup que la chaîne ne marche plus). Ils ne se remplacent pas.

- La CI ne peut pas atteindre le cluster : ni Qdrant, ni VLLM, ni MLflow, ni le secret
  `secret-codif-coicop-bdf`. Elle ne valide donc jamais qu'une étape codifie correctement.
- Le smoke run n'exerce **qu'une configuration** : `reconciliation: llm` *ou* `sirus`, mode
  production *ou* évaluation, avec ou sans `skip-index`. Le plantage du `report` en mode
  production ne serait jamais apparu dans un run nocturne en mode évaluation.
- Le smoke run vérifie que ça ne plante pas, **pas que le résultat est juste** : inverser la
  précédence regex/conciliation dans `export-results` produirait un livrable faux sans qu'aucune
  étape échoue.

### Ce que la CI aurait attrapé (bugs réels de la campagne de renommage)

| Bug | Job qui l'attrape |
|---|---|
| `reconcile-llm/main.py` importait `src.decide_coicop` après renommage du fichier (import différé, invisible au `--help`) | test qui importe chaque module |
| `report/main.py` lisait `args.ttc_model_uri` après renommage du drapeau | test qui construit les arguments |
| `prune: true` transformé en `prune-codes: true` — clé de config cassée | test qui charge les 5 `config.yaml` |
| 2 fichiers de test cassés depuis l'import dans le monorepo | `pytest` |
| liens de doc morts après renommage des pages | `quarto render` + contrôle des liens |
| références de templates/dépendances Argo pendantes | `argo lint` + script de cohérence |
| `pyproject.toml` modifié sans relock → étape en échec au démarrage (à cause de `--locked`) | `uv lock --check` |

### Les quatre jobs

1. **Lint** : `ruff check` + `ruff format --check` (config à créer à la racine).
2. **Tests** : `pytest` sur les modules qui en ont + `uv lock --check`.
3. **Cohérence du pipeline** : YAML valide, chaque tâche pointe un template existant, chaque
   dépendance existe, chaque `{{workflow.parameters.X}}` est déclaré, aucun `uv sync` sans
   `--locked`.
4. **Documentation** : `quarto render docs` + échec sur lien interne mort.

### Le smoke run

`CronWorkflow` Argo, nocturne, `sample-annotations=100`. Deux runs pour couvrir les deux
conciliations (`reconciliation: llm`, puis `sirus` avec `reconcile-sirus-model-uri` renseigné).
Coût : appels LLM + GPU, marginal sur 100 observations mais non nul.

### Tests à écrire en priorité

- **`classify-regex`** : 43 règles dans `config/rules.yaml`, logique pure sur des chaînes, aucune
  dépendance externe. Meilleur rapport valeur/effort du dépôt.
- **`export-results`** : c'est le livrable utilisateur. Sa précédence (regex avant conciliation)
  et la restauration des noms de colonnes d'origine se testent sur trois lignes fabriquées.

---

## Chantier 5b — YAML déclaratif et homogénéité *(à faire)*

- **`envFrom: secretRef`** : les 5 variables viennent du même Secret. ~410 lignes de plomberie
  recopiée 14 fois → 28.
- **`WorkflowTemplate`** pour le boilerplate commun (image, resources, env).
- **81 lignes de logique métier en shell** dans le YAML (`ARGS="$ARGS --input-file …"`, la règle
  prod/éval) à remonter dans les CLI, où elle devient testable. Le YAML redevient déclaratif.
- **Squelette de module identique** : `export-results` et `report` n'ont pas de README ;
  `report`, `export-results` et quelques scripts utilisent `print()` au lieu d'un logger.
- **Mélange `snake_case` / `kebab-case`** dans les paramètres Argo : `input_file`, `run_id`,
  `text_column` d'un côté, `skip-index`, `classify-rag-model` de l'autre.

---

## Dette identifiée en cours de route

1. **Bug latent, correctif d'une ligne** : `rag-notices/scripts/2_run_rag.py` configure un
   `FileHandler` vers `logs/` **à l'import**, sans créer le dossier. Sur un clone neuf — ce que
   fait chaque run Argo — l'étape échoue au démarrage. Ajouter
   `Path("logs").mkdir(exist_ok=True)` avant `setup_logging()`.
2. **Deux fichiers de test cassés** dans `rag-notices`, depuis leur import dans le monorepo :
   `src/rag_notices/eval/test_metrics.py` (importe `eval.metrics`, chemin inexistant) et
   `tests/test_llms.py` (fixtures `client` / `model` que rien ne définit, aucun `conftest.py`).
3. **`torchtextclassifiers` plafonné à `<2`** dans `classify-ttc/pyproject.toml` : la résolution
   commune prendrait 2.0.0, et un saut majeur de la brique de classification n'avait pas sa place
   dans un commit d'harmonisation. À monter séparément, avec un test de prédiction contre un
   modèle MLflow existant.
4. **pandas bloqué en 2.x par mlflow** (`pandas<3` dans toutes les versions ≥ 3.10 sauf 3.11.0),
   alors que `build-datasets` et `classify-regex` tournaient en 3.0.x avant l'unification. Le jour
   où mlflow supporte pandas 3 : `uv lock --upgrade-package pandas`. Raison consignée dans le
   `pyproject.toml` racine.
5. **~109 Mo de CSV morts dans git** — *traité le 2026-09-02*. Quatre fichiers référencés par
   aucun code ont été supprimés : `synthetic+gtin.csv` (100 Mo), `synthetic_data_9899.csv`
   (8,1 Mo), `20260203-liste_produits_copain.csv`, `table_passage_coicop.csv`. Conservés parce
   qu'ils servent réellement : `synthetic_data.csv` (entraînement) et les trois
   `data/annotated/*.csv`, lus **par glob** et non par nom — `evaluation_report.py:123` fait
   `directory.glob("*.csv")`, ce qu'une recherche par nom de fichier ne voit pas.
   Reste ouvert : l'historique git porte toujours ces 109 Mo, donc un clone complet les
   télécharge. Les purger exige de réécrire l'historique — décision à part, qui casse les forks
   et les PR en cours. `git clone --depth 1`, ce que fait déjà le pipeline, l'évite.
6. **Aucun run Argo depuis le workspace et le renommage.** C'est le préalable à tout le reste.

---

## Comment vérifier (recettes utilisées, reproductibles en local)

```bash
# 1. Lock à jour et environnement par module
uv lock --check
cd <module>/ && uv sync --locked --no-dev      # n'installe QUE ce module

# 2. Suites de tests existantes
cd prune-codes     && uv run --locked --group dev pytest tests -q     # 11
cd reconcile-sirus && uv run --locked --group dev pytest tests -q     # 41
cd report          && uv run --locked --group dev pytest test_coicop_metrics.py -q   # 30
cd classify-ttc    && uv run --locked --group dev pytest tests -q     # 7

# 3. Les points d'entrée se chargent (détecte les imports cassés sans rien exécuter)
cd <module>/ && uv run --locked --no-dev python <entrypoint> --help
#   NB : rag-notices et rag-annotations exigent un dossier logs/ (cf. dette n°1)

# 4. Bout en bout sur données réelles, avec chiffres connus
cd report && uv run --locked --no-dev quarto render ../annexes/benchmarking/bilan-codification.qmd
#   run codif-sj4jz (AVANT renommage, donc nécessite git checkout b40d349) :
#   accuracy globale 77,2 % — 12 576/16 293 ; sirus 79,6 % ; regex 54,6 %
#   Toute différence sur ces chiffres est une régression.

# 5. Cohérence des workflows Argo
#   YAML valide ; chaque tâche → template existant ; chaque dépendance existe ;
#   chaque {{workflow.parameters.X}} déclaré ; aucun uv sync/run sans --locked.

# 6. Documentation
quarto render docs        # puis vérifier qu'aucun href interne ne pointe vers un fichier absent
```

**Ce qui ne peut pas être vérifié sans accès au cluster** : Qdrant, VLLM, MLflow, et donc les
étapes `index-*`, `classify-rag-*`, `classify-ttc` (chargement du modèle), `reconcile-sirus`
(chargement du modèle) et le log MLflow du `report`. Il faut un service dans le bon namespace,
avec le secret `secret-codif-coicop-bdf` monté.

Commande de validation attendue :

```bash
argo submit argo/codif-pipeline.yaml \
  -p git-branch=<branche> -p sample-annotations=100 -p skip-index=true
```

---

## Prochaine étape recommandée

Le **smoke run** (chantier 4) : il valide d'un coup le workspace et le renommage, que rien
d'autre ne couvre, et il devient le filet du chantier 2. Ensuite, dans l'ordre : la CI lint +
tests, puis la brique 1 du chantier 2 (le registre), qui porte l'essentiel du gain de lisibilité
et peut être livrée seule.
