# Fixtures du test d'équivalence Python ↔ `sirus.predict`

**Ces fichiers sont volontairement versionnés.** Ce ne sont pas des artefacts de
modèle — ceux-là vivent dans MLflow — mais des **fixtures de test**, produites
une fois par `Rscript R/make_golden.R` et figées.

Elles sont dans le dépôt parce que `tests/test_scorer_golden.py` doit pouvoir
tourner **sans R** : le dépôt n'a pas de CI qui installe le paquet `sirus`, et
celui-ci a été archivé du CRAN. Sans ces fichiers, la seule preuve que le scorer
Python reproduit `sirus.predict` disparaîtrait.

| Fichier | Rôle |
|---|---|
| `rules.json` | Modèle jouet exporté par R, avec 12 règles |
| `expected_proba.csv` | Les features de 42 lignes **et** la prédiction de `sirus.predict` pour chacune, à 17 chiffres significatifs. La référence bit-à-bit |
| `rules_empty.json` | Modèle sans aucune règle : le scorer doit retomber sur `mean` |
| `threshold_roundtrip.csv` | Chaque seuil de règle et son reformatage : détecte une perte de précision à l'export |

Le jeu de `expected_proba.csv` est **construit**, pas échantillonné : il pose une
ligne exactement sur chaque seuil de règle, plus ses deux voisins immédiats.
C'est là que `<` se distingue de `>=`, donc là qu'un seuil arrondi basculerait
dans la mauvaise branche.

## Régénérer

```bash
cd reconcile-sirus/ && Rscript R/make_golden.R
```

À ne faire que si le scorer ou l'exporteur de règles changent **volontairement**,
et **à relire dans le diff** : un golden réajusté en silence sur l'implémentation
ne prouve plus rien.
