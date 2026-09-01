# sirus — conciliation COICOP par règles interprétables

Alternative au juge LLM (`reconcile-llm/`) pour choisir le code COICOP final
parmi les candidats des 4 classifieurs de base. Là où le juge produit une
décision opaque et un appel LLM par produit, SIRUS produit **une vingtaine de
règles à seuils lisibles** — relisibles, auditables, et figeables comme
convention de codage — plus un score par produit.

L'étape s'arrête là : elle livre **un code et un score**, sans verdict sur le
sort à leur réserver. Décider à partir de quel score un code est exploitable
sans relecture est une question métier, qui s'instruit sur la section
« Calibration de SIRUS » du rapport d'évaluation.

## Deux moitiés, volontairement séparées

```
  HORS PIPELINE — ./train.sh, lancé à la main sur un run passé
 ┌──────────────────────────────────────────────────────────────────────┐
 │  build-table (Python)  →  fit_sirus.R (R)  →  finalize (Python)       │
 │      table candidat        20 règles           auto-contrôle,         │
 │      level + split         + rules.json        mesures, MLflow        │
 └───────────────────────────────┬──────────────────────────────────────┘
                                 │  URI MLflow, à recopier dans params.yaml
                                 ▼
  DANS LE PIPELINE — étape reconcile-sirus, quand `conciliation: sirus`
  4 classifieurs ──→ reconcile-sirus (Python pur) ──→ reconcile-sirus/predictions.parquet
                          charge rules.json,              sirus_code, sirus_proba,
                          moyenne des règles              sirus_n_candidats
```

**L'entraînement n'est pas une étape du pipeline, et c'est délibéré.** Il produit
un modèle destiné à un run *futur* : mis dans le DAG, il ne servirait jamais le
run dans lequel il tourne. Le détour par MLflow puis par `reconcile-sirus-model-uri` est ce
qui rend structurellement impossible de prédire sur les lignes qui ont servi à
l'entraînement, et donc de lire des chiffres flatteurs. C'est l'organisation
retenue pour TTC (`classify-ttc-model-uri`).

## Utilisation

### 1. Entraîner (à la main, sur un run d'évaluation passé)

```bash
cd reconcile-sirus/
./train.sh 2026-06-29/codif-vvkv9
```

Le run désigné doit être un run d'**évaluation** terminé (soumis avec
`input_file` vide) : l'entraînement a besoin de la vérité terrain, qu'un run de
production ne porte pas.

Réglages optionnels, par variables d'environnement (les défauts conviennent
dans la quasi-totalité des cas) :

```bash
NUM_RULE=20 MAX_DEPTH=2 SEED=42 EXPERIMENT=codif-coicop-sirus \
  ./train.sh 2026-06-29/codif-vvkv9
```

Le script vérifie une seule chose avant de travailler : que
`MLFLOW_TRACKING_URI` soit définie. C'est le seul échec qui n'arriverait qu'au
dernier pas, après plusieurs minutes d'ajustement R. Tout le reste (identifiants
S3, chemins inexistants) fait échouer la première étape en quelques secondes,
donc ne mérite pas de contrôle séparé.

Il affiche en fin d'exécution l'URI à recopier :

```
Modèle SIRUS loggué. Pour l'utiliser, mettre dans argo/params.yaml :
    reconcile-sirus-model-uri: mlflow-artifacts:/12/9f3c.../artifacts/model
```

Les artefacts restent dans `artifacts/<run_id>/` : si le log MLflow échoue
(réseau, URI), il suffit de relancer la 3ᵉ étape seule, **sans réajuster le
modèle** :

```bash
uv run main.py finalize --artifacts-dir artifacts/codif-vvkv9 \
  --experiment codif-coicop-sirus --run-id codif-vvkv9 --run-date 2026-06-29
```

À vérifier dans MLflow avant de s'en servir :

| Métrique | Ce qu'elle dit |
|---|---|
| `accuracy_product` | Accuracy réelle, mesurée sur les 20 % tenus à l'écart |
| `upper_bound` | Plafond structurel : part des produits pour lesquels **au moins un** classifieur a proposé le bon code. SIRUS ne peut rien au-delà |
| `proba_min` / `proba_max` | Plage de scores atteignable. **Le nombre le plus utile** : la sortie étant une moyenne de sorties de règles, elle n'atteint jamais 0 ni 1, et cette plage dit sur quelle étendue le modèle se prononce |

Artefacts joints : `rules.json` (le modèle), `rules_printed.txt` (**les règles en
clair — le premier fichier à ouvrir quand une décision surprend**),
`threshold_sweep.json` (volume/fiabilité pour chaque seuil envisageable —
**information de comparaison entre entraînements**, pas un réglage : le pipeline
n'applique aucun seuil).

`rules.json` est la seule représentation du modèle.

### 2. Coder un run avec SIRUS

```bash
argo submit argo/codif-pipeline.yaml --parameter-file argo/params.yaml \
  -p conciliation=sirus \
  -p reconcile-sirus-model-uri=mlflow-artifacts:/12/9f3c.../artifacts/model
```

`reconcile-llm` est alors *Skipped* ; `export-results` et `report` lisent
`reconcile-sirus/`. Les deux conciliations sont exclusives.

## Exploiter le score

`reconcile-sirus` **n'applique aucun seuil**. La sortie porte, par produit :

| Colonne | Sens |
|---|---|
| `sirus_code` | Code retenu — l'argmax du score parmi les candidats. `NA` si aucun candidat |
| `sirus_proba` | Score du candidat retenu |
| `sirus_n_candidats` | Nombre de candidats scorés — un `0` explique un `NA` |

Ce n'est pas un oubli : un verdict calculé ici serait entièrement déduit du
score, donc sans information propre, tout en figeant dans le parquet un réglage
que l'aval ne pourrait plus rediscuter sans réentraîner.

Pour décider à partir de quel score un code est exploitable sans relecture,
lire la section **« Calibration de SIRUS »** du rapport d'évaluation. Elle donne
l'accuracy par tranche de score et, surtout, l'arbitrage volume/fiabilité pour
chaque seuil envisageable.

Deux propriétés du score à connaître avant de s'en servir :

- c'est une **moyenne non pondérée des sorties des règles**, donc une
  combinaison convexe de valeurs toutes strictement entre 0 et 1 : elle
  n'atteint jamais 0 ni 1 et reste tirée vers le taux de base. **0,5 n'a donc
  rien de particulier** ;
- **la plage atteignable est une propriété du modèle ajusté, pas du problème.**
  Sur un modèle donné elle peut valoir `[0.28, 0.77]` : une exigence à 0,8
  écarterait 100 % du volume, une exigence à 0,3 n'écarterait rien. Un seuil lu
  sur un modèle ne se transpose pas au suivant.

## Lire une règle

`rules_printed.txt` donne des lignes comme :

```
if conf_ragann < 0.9 then 0.134 (n=16219) else 0.898 (n=11869)
```

Parmi les candidats du train vérifiant la condition, 13,4 % étaient le bon code ;
89,8 % pour ceux ne la vérifiant pas. La probabilité finale d'un candidat est la
**moyenne, sur toutes les règles, du nombre que chacune lui attribue**.

Deux points contre-intuitifs, et importants :

- **Aucune règle ne se tait.** Une règle non vérifiée n'est pas ignorée : elle
  contribue sa valeur `else`. On divise donc toujours par le nombre total de
  règles. Gonfler `--num-rule` dilue le signal avec des règles dont le `else`
  est proche de 0,5.
- **L'agrégation n'est pas pondérée.** Le champ `rule.weights` existe dans
  l'objet R en classification (rempli à `1/K`) et `sirus.print` l'affiche, mais
  `sirus.predict` **ne le lit jamais** — le remplacer par des valeurs arbitraires
  ne change pas les prédictions. La régression ridge à coefficients positifs est
  l'agrégation de la variante *régression*. L'article (Bénard et al. 2021, EJS
  15:427-505, éq. 3.3) dit « we simply average » et précise avoir testé une
  agrégation linéaire régularisée puis l'avoir écartée : elle dégradait la
  stabilité sans gagner en accuracy. **Ne pas « corriger » `src/scorer.py` en y
  ajoutant des poids.**

## Pourquoi entraîner en R et prédire en Python

L'entraînement a besoin du paquet R `sirus`, qui n'existe qu'en R. La prédiction,
elle, est une moyenne : la réimplémenter en Python évite trois ennuis sur le
chemin de production, qui est celui qu'on emprunte à chaque run —

1. **le paquet a été archivé du CRAN** (2026-01-15) et doit être compilé depuis
   la source d'archive, avec un correctif maison (`R/sirus-0.3.3-cxx17.patch`) ;
2. **il abandonne sur tout `NA`** (`sirus:::data.check`) : une division COICOP
   inédite tuerait l'étape, là où le scorer Python écarte le candidat et fait
   basculer le produit en reprise ;
3. l'étape de prédiction devient rapide et sans compilation.

L'équivalence n'est pas supposée, elle est **prouvée deux fois** :

- `tests/test_scorer_golden.py` compare au bit près à une référence produite par
  R, sur un jeu **construit** pour poser une ligne exactement sur chaque seuil ;
- `train.sh` refait la comparaison **à chaque entraînement**, sur le modèle
  qu'il vient d'ajuster, et échoue s'il y a divergence.

Une subtilité qui a coûté un aller-retour : `numpy.mean` et `mean()` de R ne
donnent pas le même dernier bit (R accumule en `long double` avec une passe de
correction). `src/scorer.py::_r_mean` reproduit l'algorithme de R, ce qui rend
l'égalité bit-à-bit atteignable — et donc la divergence détectable.

## Développer

```bash
cd reconcile-sirus/
uv sync --locked --group dev
uv run pytest                    # tests golden + invariants de la table candidat
Rscript R/install_deps.R         # une fois par environnement, ~1 min 30 à vide
                                 # (appelé aussi par train.sh, idempotent)
Rscript R/make_golden.R          # régénère tests/golden/ — à relire dans le diff
```

| Fichier | Rôle |
|---|---|
| `train.sh` | Orchestre l'entraînement hors pipeline : construit les chemins S3 depuis `<date>/<run_id>` et enchaîne les 3 étapes. 45 lignes, **aucune logique métier** — elle vit dans les sous-commandes et dans `fit_sirus.R` |
| `src/candidates.py` | **Le code partagé** entre entraînement et prédiction : table candidat-level, sentinelles, ordre des features. Un écart entre les deux chemins serait invisible ; d'où le module unique et le hash embarqué dans `rules.json` |
| `src/scorer.py` | Miroir de `sirus.predict` pour `type="classif"` |
| `src/train.py` | Split par produit, auto-contrôle R↔Python, calibration, MLflow |
| `R/fit_sirus.R` | Ajustement 80 % (mesure) puis 100 % (modèle livré) |
| `R/export_rules.R` | Sérialisation en `rules.json`. **Tous les flottants en chaînes `%.17g`** : `jsonlite` par défaut arrondit à 4 chiffres et `digits = NA` à 15, aucun des deux ne round-trippe |
| `R/install_deps.R` | Télécharge l'archive CRAN + applique le patch texte. **N'écrase pas `repos`** : les services SSP Cloud sont configurés sur un dépôt de binaires Linux (Posit Package Manager), et forcer `cloud.r-project.org` ferait recompiler chaque paquet depuis les sources — duckdb passe alors de 7 secondes à plus de 28 minutes |

## Limites connues

- **Un changement de RAG-ANN invalide le modèle.** Sur le modèle
  expérimental, 13 des 20 règles portaient sur RAG-ANN, et `conf_ragann` y était
  quasi binaire (`-1` ou `1.0`). Un simple ajustement de prompt en amont
  repointe donc l'essentiel du modèle, en silence. `reconcile-sirus` compare les
  distributions du run à celles de l'entraînement et avertit en cas de dérive,
  mais la seule vraie réponse est de réentraîner. Un réentraînement sur un
  RAG-ANN dégradé produit un modèle *correctement calibré sur un pipeline moins
  bon* — c'est le comportement voulu, mais ça ressemblera à une régression.
- **SIRUS ne crée pas de candidat.** Si aucun classifieur n'a proposé le bon
  code, aucun réglage ne le retrouvera : c'est ce que mesure `upper_bound`.
- **Une division COICOP inédite coûte du volume.** Un `code_candidat_n1` absent
  du modèle rend le candidat non scorable (compté et loggué en ERROR, jamais
  absorbé en silence). Un réentraînement l'intègre.
- **Classification binaire uniquement.** Pas de version multi-classe, ni dans
  l'article ni dans le paquet : d'où le cadrage « ce candidat est-il correct ? »
  plutôt que « quel code ? ».
- **Le rapport de production** (`report/prediction_report.qmd`) saute ses
  sections d'accord avec le juge en mode SIRUS ; le rapport d'évaluation, lui,
  est complet dans les deux modes.

## Références

- Bénard, C., Biau, G., da Veiga, S., Scornet, E. (2021). *SIRUS: Stable and
  Interpretable RUle Set for Classification*. Electronic Journal of Statistics
  15:427-505. [doi:10.1214/20-EJS1792](https://doi.org/10.1214/20-EJS1792)
- Extension régression : AISTATS 2021, PMLR 130:937-945.
- Paquet (archivé) : [gitlab.com/drti/sirus](https://gitlab.com/drti/sirus)
