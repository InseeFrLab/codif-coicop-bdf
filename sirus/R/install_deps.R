#!/usr/bin/env Rscript
# =============================================================================
# Dépendances R de l'entraînement, seul chemin qui a besoin de R : la prédiction
# est en Python (cf. src/scorer.py), et aucune étape du pipeline Argo n'installe
# ce paquet.
#
# À lancer depuis le dossier `sirus/` : Rscript R/install_deps.R
#
# Convention du dépôt (cf. stats-annotations/R/main.R) : installation au
# runtime, idempotente, gardée par installed.packages(). Les services SSP Cloud
# n'embarquent pas ces paquets et ils ne persistent pas entre deux relances du
# service.
# =============================================================================

# NE PAS écraser `repos` : l'image Onyxia est configurée (Rprofile.site) sur
# Posit Package Manager, qui sert des BINAIRES Linux précompilés. Forcer
# cloud.r-project.org — comme le faisait le script expérimental d'origine — fait
# recompiler chaque paquet depuis les sources : duckdb passe de 7 secondes à plus
# de 28 minutes, et l'arbre de ggplot2 s'alourdit d'autant.
#
# Repli seulement si aucun dépôt n'est configuré (autre image, R nu).
if (!nzchar(getOption("repos")[["CRAN"]]) || getOption("repos")[["CRAN"]] == "@CRAN@") {
  options(repos = c(CRAN = "https://cloud.r-project.org"))
}
cat("Dépôt R utilisé :", getOption("repos")[["CRAN"]], "\n")

# duckdb + DBI : pour lire le parquet des features écrit par `main.py
# build-table`. Servis en binaire par le dépôt configuré (7 secondes) — c'est
# précisément pour ça qu'il ne faut pas écraser `repos` ci-dessus.
#
# Les dépendances de `sirus` lui-même doivent être listées ICI : on l'installe
# depuis une source locale avec `repos = NULL`, et sous cette forme
# `install.packages` NE RÉSOUT PAS les dépendances. Un oubli se traduit par
# "ERROR: dependency 'x' is not available for package 'sirus'".
#
# sirus 0.3.3 : Imports Rcpp, Matrix, ROCR, ggplot2, glmnet ; LinkingTo Rcpp,
# RcppEigen. Matrix est un paquet « recommended » livré avec R, mais on le liste
# quand même — le lister ne coûte rien s'il est déjà là.
deps <- c(
  "duckdb", "DBI",
  "jsonlite", "Rcpp", "Matrix", "ROCR", "ggplot2", "glmnet", "RcppEigen"
)
manquants <- setdiff(deps, rownames(installed.packages()))
if (length(manquants) > 0) {
  cat("Installation des dépendances CRAN manquantes :", paste(manquants, collapse = ", "), "\n")
  install.packages(manquants)
} else {
  cat("Dépendances CRAN déjà présentes.\n")
}

# --- Le paquet `sirus` --------------------------------------------------------
# `install.packages("sirus")` est IMPOSSIBLE : le paquet a été archivé du CRAN
# le 2026-01-15 ("as issues were not addressed in time") et renvoie
# "package 'sirus' is not available". L'archive reste servie à une URL canonique
# et stable, et le dernier tag upstream (gitlab.com/drti/sirus) est le même
# 0.3.3 de 2022 : il n'existe aucune version plus récente à préférer.
#
# La source d'archive NE COMPILE PAS telle quelle sur les toolchains récentes :
#   error: call of overloaded 'make_unique<sirus::TreeClassification>' is ambiguous
# Le paquet fournit son propre polyfill `make_unique` pour C++11, mais R ignore
# désormais sa directive `CXX_STD = CXX11` et compile en C++17+, où
# std::make_unique existe déjà — les deux candidats deviennent ambigus par ADL.
# Second problème du même ordre : les macros nues de Rinternals.h (`error`,
# `length`) polluent des en-têtes standard inclus indirectement.
#
# Le correctif tient en 2 fichiers, versionné en clair dans
# R/sirus-0.3.3-cxx17.patch (55 lignes) plutôt qu'en tarball binaire : il est
# ainsi relisible en revue. Appliqué inconditionnellement — il est neutre sur
# une vraie compilation C++11 (gardé par `#if __cplusplus < 201402L`), donc
# déterministe plutôt que dépendant de la toolchain de l'image.
if (!"sirus" %in% rownames(installed.packages())) {
  patch <- normalizePath(file.path("R", "sirus-0.3.3-cxx17.patch"), mustWork = TRUE)
  url <- "https://cran.r-project.org/src/contrib/Archive/sirus/sirus_0.3.3.tar.gz"

  td <- tempfile("sirus_src_")
  dir.create(td)
  tarball <- file.path(td, "sirus_0.3.3.tar.gz")
  cat("Téléchargement de la source d'archive CRAN...\n")
  utils::download.file(url, tarball, mode = "wb", quiet = TRUE)
  utils::untar(tarball, exdir = td)

  src <- file.path(td, "sirus")
  stopifnot("source sirus non extraite" = dir.exists(src))

  # `git` est nécessaire pour appliquer le patch. Il est présent partout où ce
  # dépôt a pu être cloné, donc la supposition est sûre. `git apply -p1`
  # fonctionne hors dépôt git dès lors que les chemins du patch sont relatifs au
  # répertoire courant.
  cat("Application de", basename(patch), "...\n")
  wd <- setwd(src)
  statut <- system2("git", c("apply", "-p1", shQuote(patch)))
  setwd(wd)
  stopifnot("application du patch sirus échouée" = statut == 0)

  cat("Compilation de sirus depuis la source patchée...\n")
  install.packages(src, repos = NULL, type = "source")
} else {
  cat("sirus déjà installé.\n")
}

# stopifnot() et non warning() : un warning ne met PAS le code de sortie à
# non-zéro, et `train.sh` (en `set -e`) enchaînerait sur l'ajustement avec un
# paquet manquant, pour échouer plus loin sans dire pourquoi.
stopifnot("sirus ne s'est pas installé correctement" = "sirus" %in% rownames(installed.packages()))
cat("\nToutes les dépendances R sont prêtes.\n")
