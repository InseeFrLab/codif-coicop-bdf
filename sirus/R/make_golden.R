#!/usr/bin/env Rscript
# =============================================================================
# OUTIL DE DÉVELOPPEMENT — ne tourne JAMAIS dans Argo.
#
# Régénère les fixtures de tests/golden/ qui prouvent que src/scorer.py
# reproduit sirus.predict. À relancer uniquement quand le scorer ou l'exporteur
# de règles changent volontairement, et à relire dans le diff : un golden
# réajusté en silence sur l'implémentation ne prouve plus rien.
#
# Usage : Rscript R/make_golden.R      (depuis le dossier `sirus/`)
# =============================================================================

options(warn = 2)
suppressPackageStartupMessages({
  library(jsonlite)
  library(sirus)
})
source(file.path("R", "export_rules.R"))

set.seed(11)
out_dir <- file.path("tests", "golden")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

FEATURES <- c(
  "vote_lcs", "vote_rag", "vote_ragann", "vote_ttc", "nb_votants",
  "conf_rag", "conf_ragann", "conf_ttc", "dist_lcs", "code_candidat_n1"
)
NIVEAUX <- c("01", "02", "03", "04", "05", "06", "07", "08",
             "09", "10", "11", "12", "13", "98", "99")

# --- Un jeu d'entraînement jouet, juste assez riche pour produire des règles
# variées : conditions numériques dans les deux sens, et au moins une condition
# factorielle.
n <- 900
tirer_conf <- function(n) {
  # Mélange de la sentinelle -1 et de vraies confiances : c'est cette forme qui
  # fait que SIRUS coupe parfois exactement sur la sentinelle.
  ifelse(runif(n) < 0.45, -1, runif(n))
}
train <- data.frame(
  vote_lcs    = rbinom(n, 1, 0.5),
  vote_rag    = rbinom(n, 1, 0.5),
  vote_ragann = rbinom(n, 1, 0.5),
  vote_ttc    = rbinom(n, 1, 0.5),
  conf_rag    = tirer_conf(n),
  conf_ragann = tirer_conf(n),
  conf_ttc    = tirer_conf(n),
  dist_lcs    = ifelse(runif(n) < 0.4, 1.5, runif(n, 0, 0.8)),
  code_candidat_n1 = factor(sample(NIVEAUX, n, replace = TRUE), levels = NIVEAUX),
  stringsAsFactors = FALSE
)
train$nb_votants <- train$vote_lcs + train$vote_rag + train$vote_ragann + train$vote_ttc
train <- train[train$nb_votants >= 1, ]
train <- train[, FEATURES]

# Cible corrélée aux features, pour que les règles aient du sens.
score_latent <- with(
  train,
  0.9 * nb_votants + 1.4 * pmax(conf_ttc, 0) + 1.1 * pmax(conf_ragann, 0) -
    1.2 * pmin(dist_lcs, 1) + rnorm(nrow(train), 0, 0.5)
)
y <- as.integer(score_latent > median(score_latent))

m <- sirus.fit(
  as.data.frame(train), y,
  type = "classif", num.rule = 12, max.depth = 2, seed = 11, verbose = FALSE
)
cat(sprintf("modèle golden : %d règles, %d arbres\n", length(m$rules), m$num.trees))

# --- Le jeu à scorer : CONSTRUIT, pas échantillonné -------------------------
# Il doit exercer chaque branche de chaque règle, et surtout les valeurs situées
# EXACTEMENT sur les seuils : c'est là que `<` se distingue de `>=`, et donc là
# qu'un seuil arrondi à l'export basculerait dans la mauvaise branche.
seuils <- do.call(rbind, lapply(seq_along(m$rules), function(k) {
  do.call(rbind, lapply(m$rules[[k]], function(split) {
    if (split[2] %in% c("<", ">=")) {
      data.frame(var = split[1], seuil = as.numeric(split[3]), stringsAsFactors = FALSE)
    } else {
      NULL
    }
  }))
}))
seuils <- unique(seuils)
cat(sprintf("%d seuil(s) numérique(s) distinct(s) à border\n", nrow(seuils)))

ligne_base <- function() {
  data.frame(
    vote_lcs = 1L, vote_rag = 1L, vote_ragann = 1L, vote_ttc = 1L, nb_votants = 4L,
    conf_rag = 0.5, conf_ragann = 0.5, conf_ttc = 0.5, dist_lcs = 0.4,
    code_candidat_n1 = "01", stringsAsFactors = FALSE
  )
}

lignes <- list()
# (a) pour chaque seuil : la valeur exacte, et ses deux voisins immédiats
for (i in seq_len(nrow(seuils))) {
  v <- seuils$var[i]
  t <- seuils$seuil[i]
  for (val in c(t, nextafter_down <- t - .Machine$double.eps * max(abs(t), 1),
                t + .Machine$double.eps * max(abs(t), 1))) {
    r <- ligne_base()
    r[[v]] <- val
    lignes[[length(lignes) + 1]] <- r
  }
}
# (b) chaque niveau de facteur connu
for (lev in NIVEAUX) {
  r <- ligne_base()
  r$code_candidat_n1 <- lev
  lignes[[length(lignes) + 1]] <- r
}
# (c) les 4 sentinelles actives, puis aucune
r <- ligne_base()
r$conf_rag <- -1; r$conf_ragann <- -1; r$conf_ttc <- -1; r$dist_lcs <- 1.5
r$vote_lcs <- 0L; r$vote_rag <- 0L; r$vote_ragann <- 0L; r$vote_ttc <- 1L; r$nb_votants <- 1L
lignes[[length(lignes) + 1]] <- r
# (d) les extrêmes de nb_votants
for (nv in c(1L, 4L)) {
  r <- ligne_base(); r$nb_votants <- nv
  lignes[[length(lignes) + 1]] <- r
}

test <- do.call(rbind, lignes)
test$code_candidat_n1 <- factor(test$code_candidat_n1, levels = NIVEAUX)
test <- test[, FEATURES]
cat(sprintf("jeu à scorer : %d lignes\n", nrow(test)))

proba <- sirus.predict(m, as.data.frame(test))

# --- Écriture -------------------------------------------------------------
writeLines(
  export_rules(m, NIVEAUX, list(hyperparams = list(
    num_rule_requested = 12L, num_rule_selected = length(m$rules),
    max_depth = 2L, num_trees = m$num.trees, q = 10L, seed = 11L
  ))),
  file.path(out_dir, "rules.json")
)

sortie <- test
sortie$code_candidat_n1 <- as.character(sortie$code_candidat_n1)
# TOUTES les colonnes flottantes en chaînes 17 chiffres, pas seulement proba_R.
# `write.csv` n'écrit que 15 chiffres significatifs par défaut : les lignes
# posées EXACTEMENT sur un seuil seraient arrondies, les bornes s'effondreraient,
# et Python scorerait des features différentes de celles que R a vues — le test
# échouerait pour une mauvaise raison, ou pire, passerait en ne testant rien.
for (col in setdiff(FEATURES, "code_candidat_n1")) {
  # Y compris vote_* et nb_votants : les lignes bordant le seuil sur
  # `nb_votants` portent 2 - eps et 2 + eps, qui ne sont pas des entiers.
  sortie[[col]] <- g17(as.numeric(sortie[[col]]))
}
sortie$proba_R <- g17(proba)
write.csv(sortie, file.path(out_dir, "expected_proba.csv"), row.names = FALSE)

# Round-trip des seuils : chaque chaîne du JSON doit se relire à l'identique.
# Si `g17` était un jour remplacé par un arrondi, ce fichier le révélerait.
rt <- data.frame(
  threshold_string = seuils$seuil_chr <- vapply(
    seq_len(nrow(seuils)), function(i) sprintf("%.17g", seuils$seuil[i]), character(1)
  ),
  reformatted = vapply(
    seq_len(nrow(seuils)),
    function(i) sprintf("%.17g", as.numeric(sprintf("%.17g", seuils$seuil[i]))),
    character(1)
  ),
  stringsAsFactors = FALSE
)
write.csv(rt, file.path(out_dir, "threshold_roundtrip.csv"), row.names = FALSE)

# Modèle sans aucune règle : le scorer doit retomber sur `mean`.
vide <- fromJSON(file.path(out_dir, "rules.json"), simplifyVector = FALSE)
vide$rules <- list()
writeLines(
  toJSON(vide, auto_unbox = TRUE, pretty = TRUE, digits = NA),
  file.path(out_dir, "rules_empty.json")
)

cat("\nFixtures écrites dans", out_dir, "\n")
