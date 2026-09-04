# evaluate

Étape finale **facultative** du pipeline de codification : elle mesure la qualité d'un run
dont le fichier d'entrée portait des étiquettes.

```bash
cd evaluate/
uv sync --locked
uv run main.py --run-id <ID> --run-date <YYYY-MM-DD> \
  --input-file s3://.../workflow_inputs/mon_fichier.csv \
  --reconciliation llm \
  --source-column source        # facultatif : ventilation par provenance
```

Dans Argo, elle ne tourne que si :

```yaml
when: '"{{workflow.parameters.skip-eval}}" != "true" && "{{workflow.parameters.label-column}}" != ""'
```

Sans `label-column`, la tâche apparaît `Skipped` et le reste du run est inchangé. C'est le cas
nominal de production.

## Pourquoi ce module existe

Le pipeline a été construit **pour être évalué**, et la mesure s'était installée un peu
partout : un mode production/évaluation dérivé du seul fait qu'`input_file` soit vide ou non,
testé à treize endroits, plus un `--skip-eval`, un `train_test_split`, une option `kb-scope`,
et des calculs de métriques dans trois étapes de classification.

L'une d'elles, `rag-notices`, n'a jamais connu la dualité : elle évaluait **sans condition**, et
la garde qui devait ignorer les lignes sans vérité terrain était commentée. Chaque run de
production loguait donc dans MLflow une accuracy ≈ 0, silencieusement.

Rassembler la mesure ici rend le reste du pipeline monomode, et rend l'évaluation explicite :
on la demande, elle ne se déclenche pas toute seule.

## Trois entrées, pas une

Le parquet de conciliation ne suffit pas. Il porte les quatre classifieurs et la décision
finale, mais il lui manque deux choses : les codes récupérés par les RAG, retirés à la fusion,
et **les lignes captées par la regex**, qui n'entrent jamais dans la chaîne d'arbitrage.

| Entrée | Ce qu'elle apporte |
|---|---|
| `…/{run}/reconcile-{llm,sirus}/predictions.parquet` | les 4 classifieurs et la conciliation |
| `…/{run}/classify-rag-notices/retrieved_codes.parquet` et `…/{run}/classify-rag-annotations/predictions.parquet` | le **recall de retrieval** et l'accuracy de génération conditionnelle : le seul indicateur qui dise si un RAG échoue à *retrouver* ou à *générer* |
| le livrable d'`export-results` | l'**accuracy de bout en bout, regex comprise** — le chiffre métier |

Les deux premiers artefacts étaient écrits à chaque run et **jamais relus**.

Tous les chemins viennent du registre (`contracts.yaml` via `codif_common.contracts`) : aucune
URI S3 n'est construite à la main ici.

## L'étape échoue si `code_lvl4` est absent

Elle ne se rabat **pas** sur `code` avec un avertissement, comme le faisait l'ancien rapport.

`code_lvl4` — la vérité terrain sous forme canonique, tronquée au niveau 4 puis élaguée — naît
en un seul endroit du pipeline : [`reconcile-llm/`](../reconcile-llm/), et seulement si
`--mapping-file` lui est passé. L'oublier ne cassait rien : tout le rapport sortait avec une
accuracy sous-estimée sur près d'un quart des Postes, parce qu'une prédiction canonique exacte
(`01.3` face à une annotation `01.3.0.0.x`) était comptée fausse.

Repli acceptable dans un rapport qui ne mesure rien ; pas quand quelqu'un a explicitement
demandé une mesure. Le message d'échec liste les colonnes présentes et nomme la cause probable.
Testé par `tests/test_truth.py`.

## Sorties

| Destination | Contenu |
|---|---|
| `…/{run}/evaluate/evaluation_report.html` | le rapport, HTML auto-contenu |
| MLflow (`--experiment-name`, `codif-coicop-eval` par défaut) | les deux conventions d'accuracy par méthode et par niveau, `truth_shallower_than_niv<k>_count`, le recall de retrieval, `llm_prompt_tokens_total`, `llm_latency_s_mean` |

L'expérience est **distincte** de celle du rapport de production : les deux ne mesurent pas la
même chose et n'ont pas le même schéma de métriques.

## Le gabarit

`evaluation_report.qmd` (ex-`report/report.qmd`) lit ses entrées via des variables
d'environnement plutôt que des paramètres Quarto : `EVAL_DECIDE_PATH`, `EVAL_RETRIEVED_PATH`,
`EVAL_RAGANN_PATH`, `EVAL_DELIVERABLE_PATH`, `EVAL_RUN_ID`, `EVAL_RUN_DATE`,
`EVAL_SOURCE_COLUMN`. `main.py` les positionne avant d'appeler `quarto render` ; pour itérer
sur le gabarit en local, les exporter soi-même et rendre le `.qmd` directement.

## Dépendances de workspace

Trois bibliothèques de mesure, toutes paramétrées par nom de colonne, donc applicables à
n'importe quelle prédiction :

| Module | Ce qu'il apporte |
|---|---|
| [`common/`](../common/) | `codif_common.metrics` : accuracy par niveau (2 conventions), couverture, régimes, `final_decision` |
| [`rag-notices/`](../rag-notices/) | les 5 filtres, le recall de retrieval, l'accuracy de génération conditionnelle |
| [`rag-annotations/`](../rag-annotations/) | AUROC, distorsion de distribution (TV et KL), fiabilité du `codable`, `accuracy_by_source` |
| [`prune-codes/`](../prune-codes/) | vocabulaire de code COICOP (troncature, sentinelles d'abstention) |

## Tests

```bash
cd evaluate/ && uv run --locked pytest -q
```
