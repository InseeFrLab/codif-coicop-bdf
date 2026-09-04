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
```

Chaîne exécutée : `build-datasets → prune-codes (--only kb) → index-annotations`.
`classify-regex` n'en fait pas partie : la KB n'a pas à être filtrée des
produits que la regex sait coder.

## Récupérer le nom de la collection

L'étape affiche en fin d'exécution la ligne à recopier dans `argo/params.yaml` :

```
classify-rag-annotations-collection: coicop_annotations__2026-09-02__index-annotations-a7k2p
```

Si le workflow est déjà terminé :

```bash
argo logs <nom-du-workflow> | grep classify-rag-annotations-collection
```

## Paramètres

| Paramètre | Défaut | Rôle |
|---|---|---|
| `kb-sample-size` | `""` (toute la KB) | Plafonne la KB indexée |
| `collection-name` | `""` (composé) | Force un nom exact |
| `git-branch` | `main` | Branche clonée par les steps |

`kb-sample-size` n'est pas le `sample-annotations` du pipeline de classification :
celui-ci ne plafonne que les observations à coder, jamais la KB.

## Il y avait ici une option `kb-scope`

Le split train/test n'existait que parce que les seuls produits annotés
disponibles servaient à la fois de KB et de jeu d'évaluation : il fallait bien
en réserver une moitié pour mesurer. Les nouveaux produits annotés fournissent
désormais un jeu indépendant, donc **toute la base historique sert de KB** et
l'option a été supprimée, avec le segment `__full__` / `__train__` du nom de
collection.

Reste la vigilance qu'elle couvrait : si le fichier évalué provient du même lot
historique que la KB, le RAG y retrouve ses réponses et l'accuracy est gonflée,
sans qu'aucune étape n'échoue. Plus rien ne le détecte.
