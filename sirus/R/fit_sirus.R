#!/usr/bin/env Rscript
# =============================================================================
# Ajustement du modèle SIRUS de conciliation COICOP.
#
# Entrée  : le parquet de features candidat-level produit par
#           `main.py build-table` (colonnes : id, code_candidat, les 10 features,
#           correcte, et `split` valant "train"/"test").
# Sorties : dans --out-dir,
#           rules_eval.json     modèle ajusté sur les 80 % (pour mesurer)
#           proba_eval_R.csv    ses prédictions sur les 20 %, référence de
#                               l'auto-contrôle Python (voir main.py finalize)
#           rules.json          modèle LIVRÉ, réajusté sur 100 % des données
#           rules_printed.txt   sirus.print() du modèle livré — la fiche métier
#
# Usage : Rscript R/fit_sirus.R --features=... --out-dir=... [--num-rule=20]
#                              [--max-depth=2] [--seed=42]
#         (depuis le dossier `sirus/`)
# =============================================================================

# Un warning R ne met PAS le code de sortie à non-zéro : sans ceci, `train.sh`
# (en `set -e`) enchaînerait sur la suite en ayant silencieusement dérapé.
options(warn = 2)

suppressPackageStartupMessages({
  library(DBI)
  library(duckdb)
  library(jsonlite)
  library(sirus)
})

source(file.path("R", "export_rules.R"))

# --- Arguments (style du dépôt : --flag=valeur, cf. stats-annotations/R/main.R)
args <- commandArgs(trailingOnly = TRUE)
arg <- function(nom, defaut = NULL) {
  motif <- paste0("^--", nom, "=")
  trouve <- grep(motif, args, value = TRUE)
  if (length(trouve) == 0) {
    if (is.null(defaut)) stop(sprintf("argument --%s manquant", nom))
    return(defaut)
  }
  sub(motif, "", trouve[1])
}

chemin_features <- arg("features")
out_dir         <- arg("out-dir")
num_rule        <- as.integer(arg("num-rule", "20"))
max_depth       <- as.integer(arg("max-depth", "2"))
seed            <- as.integer(arg("seed", "42"))

dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

FEATURES <- c(
  "vote_lcs", "vote_rag", "vote_ragann", "vote_ttc", "nb_votants",
  "conf_rag", "conf_ragann", "conf_ttc", "dist_lcs", "code_candidat_n1"
)

# --- Chargement ---------------------------------------------------------------
# Parquet plutôt que CSV : il porte les types et les doubles EXACTS nativement.
# Un CSV imposerait un formatage à 17 chiffres significatifs à l'écriture et un
# `colClasses` à la lecture — deux mécanismes silencieux dont dépendrait
# l'exactitude des seuils appris. duckdb est servi en binaire par le dépôt
# configuré dans l'image (quelques secondes).
con <- dbConnect(duckdb::duckdb())
on.exit(dbDisconnect(con, shutdown = TRUE), add = TRUE)
feat <- dbGetQuery(con, sprintf("SELECT * FROM read_parquet('%s')", chemin_features))

stopifnot(
  "colonnes de features manquantes" = all(FEATURES %in% names(feat)),
  "colonne `correcte` absente : l'entraînement exige des données labellisées" =
    "correcte" %in% names(feat),
  "colonne `split` absente : le découpage 80/20 est fait par main.py build-table" =
    "split" %in% names(feat)
)

# Niveaux triés (et non l'ordre par fréquence de m$bins) : le scoring fait un
# `%in%` sur les libellés, donc l'ordre n'a aucun effet sur la prédiction, mais
# une liste stable rend rules.json comparable d'un entraînement à l'autre.
niveaux <- sort(unique(as.character(feat$code_candidat_n1)))
feat$code_candidat_n1 <- factor(as.character(feat$code_candidat_n1), levels = niveaux)

# sirus:::data.check abandonne sur tout NA : autant échouer ici, avec un message
# qui dit quelle colonne, que dans les entrailles du paquet.
na_par_colonne <- vapply(feat[, FEATURES], function(x) sum(is.na(x)), integer(1))
stopifnot(
  "NA dans les features (data.check de sirus abandonnerait)" = sum(na_par_colonne) == 0
)

# Cible dégénérée : `sirus.fit` échoue alors au fond de `ranger.stab` sur
# « missing value where TRUE/FALSE needed », message qui ne dit rien de la cause.
# Autant échouer ici en la nommant. Le cas se produit quand la vérité terrain
# n'est pas comparable aux candidats — typiquement une colonne `code_lvl4` qui
# n'a pas été tronquée au même niveau qu'eux (mapping absent ou inadapté).
repartition <- table(factor(feat$correcte, levels = c(0, 1)))
if (any(repartition == 0)) {
  stop(sprintf(
    paste0(
      "cible dégénérée : `correcte` vaut %s sur les %d candidats. Aucun modèle ",
      "n'est ajustable. Cause la plus fréquente : la vérité terrain n'est pas ",
      "au même niveau de troncature que les codes candidats — vérifier que ",
      "`build-table` a bien reçu --mapping-file, et que `code_lvl4` porte des ",
      "codes de niveau 4 (répartition observée : %s)."
    ),
    if (repartition[["1"]] == 0) "0 partout" else "1 partout",
    nrow(feat),
    paste(sprintf("%s=%d", names(repartition), as.integer(repartition)), collapse = " ")
  ))
}

train <- feat[feat$split == "train", ]
test  <- feat[feat$split == "test", ]
cat(sprintf(
  "Features : %d candidats (%d train / %d test), %d produits, %d modalités de division\n",
  nrow(feat), nrow(train), nrow(test), length(unique(feat$id)), length(niveaux)
))

metadonnees_communes <- function(m) {
  list(
    features_sha256 = arg("features-sha256", ""),
    hyperparams = list(
      num_rule_requested = num_rule,
      # Peut être INFÉRIEUR à num_rule : le filtrage des dépendances linéaires
      # en écarte. Ne jamais supposer num_rule règles en aval.
      num_rule_selected = length(m$rules),
      max_depth = max_depth,
      # num.trees est choisi adaptativement par le critère d'arrêt de stabilité
      # (on ne le fixe pas), il varie donc d'un entraînement à l'autre.
      num_trees = m$num.trees,
      q = 10L,
      seed = seed
    )
  )
}

# --- Modèle d'évaluation : ajusté sur les 80 % -------------------------------
cat("\nAjustement du modèle d'évaluation (80 %)...\n")
m_eval <- sirus.fit(
  as.data.frame(train[, FEATURES]), train$correcte,
  type = "classif", num.rule = num_rule, max.depth = max_depth,
  # seed explicite : sans lui sirus.fit n'est pas reproductible (le script
  # expérimental d'origine ne le passait pas).
  seed = seed, verbose = FALSE
)
cat(sprintf("  %d règles retenues, %d arbres\n", length(m_eval$rules), m_eval$num.trees))

# Référence de l'auto-contrôle : les probabilités que R calcule sur le test
# 20 %. `main.py finalize` recalcule les mêmes avec le scorer Python et échoue
# si un seul écart apparaît. Le test golden prouve l'équivalence au jour où on
# l'a figé ; ceci la prouve sur CE modèle, avec CETTE version de R.
proba_R <- sirus.predict(m_eval, as.data.frame(test[, FEATURES]))
write.csv(
  data.frame(
    id = test$id,
    code_candidat = test$code_candidat,
    proba_R = g17(proba_R),
    stringsAsFactors = FALSE
  ),
  file.path(out_dir, "proba_eval_R.csv"),
  row.names = FALSE
)
writeLines(
  export_rules(m_eval, niveaux, metadonnees_communes(m_eval)),
  file.path(out_dir, "rules_eval.json")
)

# --- Modèle livré : réajusté sur 100 % des données ---------------------------
# L'évaluation ci-dessus sert à ESTIMER accuracy et calibration. Cette
# estimation obtenue, il n'y a plus de raison de réserver 20 % : le modèle livré
# utilise tout ce qui est disponible.
cat("\nRéajustement du modèle livré (100 %)...\n")
m_final <- sirus.fit(
  as.data.frame(feat[, FEATURES]), feat$correcte,
  type = "classif", num.rule = num_rule, max.depth = max_depth,
  seed = seed, verbose = FALSE
)
cat(sprintf("  %d règles retenues, %d arbres\n", length(m_final$rules), m_final$num.trees))

writeLines(
  export_rules(m_final, niveaux, metadonnees_communes(m_final)),
  file.path(out_dir, "rules.json")
)

# Pas de `saveRDS` du modèle : `rules.json` porte tout ce dont la prédiction a
# besoin (conditions, sorties, niveaux de facteur) plus la fréquence de chaque
# règle pour l'audit, et `rules_printed.txt` en donne la forme lisible. Un
# `.rds` serait une seconde représentation du même objet, dans un format
# binaire propre à R et couplé à une version du paquet — or `sirus` est archivé
# du CRAN, donc relire ce fichier dans deux ans supposerait de réussir à le
# réinstaller. Dans un pipeline essentiellement Python, c'est une charge et non
# une assurance : la fidélité de rules.json est garantie par le test golden et
# par l'auto-contrôle rejoué à chaque entraînement.

# La liste de règles en clair : c'est l'objet du chantier (des règles qu'un
# métier peut relire et figer), et le premier fichier à ouvrir quand une
# décision du pipeline surprend.
con_txt <- file(file.path(out_dir, "rules_printed.txt"), open = "wt")
sink(con_txt)
sirus.print(m_final)
sink()
close(con_txt)

cat("\nÉcrit dans", out_dir, ":\n")
cat("  rules.json rules_eval.json proba_eval_R.csv rules_printed.txt\n")
