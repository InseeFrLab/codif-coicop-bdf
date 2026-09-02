# Lancer `index-annotations-pipeline`

Construit la **vector DB des annotations** (KB few-shot du RAG). Pipeline
autonome, hors du pipeline de classification — même principe que l'entraînement
de `classify-ttc` et de `reconcile-sirus`.

Ce qu'il indexe : les **produits déjà annotés**, soit `annotations_full` +
`suggester` au sens de `build-datasets`. Il n'y a pas de paramètre `input_file` :
celui-ci désigne les produits *à classer*, ce qui ne concerne pas la
constitution d'une base d'exemples.

Prérequis identiques à `codif-pipeline` : voir [`argo_helper.md`](./argo_helper.md).

## Lancer

```bash
# Cas normal — KB = tous les produits annotés + suggester
argo submit argo/index-annotations-pipeline.yaml --watch

# Essai rapide — plafonne la KB indexée, donc le temps d'embedding
argo submit argo/index-annotations-pipeline.yaml -p kb-sample-size=100 --watch

# Ancien split train (transitoire, voir plus bas)
argo submit argo/index-annotations-pipeline.yaml -p kb-scope=train --watch
```

Chaîne exécutée : `build-datasets → prune-codes (--only kb) → index-annotations`.
`classify-regex` n'en fait pas partie : la KB n'a pas à être filtrée des
produits que la regex sait coder.

## Récupérer le nom de la collection

L'étape affiche en fin d'exécution la ligne à recopier dans `argo/params.yaml` :

```
classify-rag-annotations-collection: coicop_annotations__full__2026-09-02__index-annotations-a7k2p
```

Si le workflow est déjà terminé :

```bash
argo logs <nom-du-workflow> | grep classify-rag-annotations-collection
```

## Paramètres

| Paramètre | Défaut | Rôle |
|---|---|---|
| `kb-scope` | `full` | `full` = tous les produits annotés ; `train` = ancien split |
| `kb-sample-size` | `""` (toute la KB) | Plafonne la KB indexée |
| `collection-name` | `""` (composé) | Force un nom exact |
| `git-branch` | `main` | Branche clonée par les steps |

`kb-sample-size` n'est pas le `sample-annotations` du pipeline de classification :
celui-ci ne plafonne que les observations à coder, jamais la KB.

## `kb-scope=train` : transitoire

Le split train/test n'existait que parce que les seuls produits annotés
disponibles servaient à la fois de KB et de jeu d'évaluation. Les nouveaux
produits annotés fournissent désormais un jeu de test indépendant, donc toute
la base historique peut servir de KB — c'est le défaut `full`. L'option `train`
reste le temps de la transition et sera supprimée.

Tant qu'elle existe : si le jeu de test d'un run d'évaluation provient du même
lot historique que la KB, une KB `full` contient ses réponses et l'accuracy est
gonflée. `classify-rag-annotations` émet un avertissement dans ce cas, sans
bloquer. Le `kb-scope` est visible dans le nom de la collection et dans son
manifeste.

Chaque exécution crée une collection de plus ; rien ne supprime les anciennes.
