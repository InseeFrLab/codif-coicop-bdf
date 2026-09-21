###############################################################################'
# Analyse des fichiers d'annotations de la campagne de test de l'enquête BdF
###############################################################################'

###############################################################################'
# 0 - Paramètres ---------------------------------------------------------------
if(!"aws.s3" %in% installed.packages()) install.packages("aws.s3", repos = "https://cloud.R-project.org")
if(!"Rcpp" %in% installed.packages()) install.packages("Rcpp")
if(!"duckdb" %in% installed.packages()) install.packages("duckdb")
if(!"DBI" %in% installed.packages()) install.packages("DBI")
if(!"glue" %in% installed.packages()) install.packages("glue")
if(!"stopwords" %in% installed.packages()) install.packages("stopwords")
if(!"dplyr" %in% installed.packages()) install.packages("dplyr")
if(!"ggplot2" %in% installed.packages()) install.packages("ggplot2")
if(!"arrow" %in% installed.packages()) install.packages("arrow")
if(!"future" %in% installed.packages()) install.packages("future")
if(!"furrr" %in% installed.packages()) install.packages("furrr")
# Lecture de contracts.yaml, le registre partagé avec les modules Python.
if(!"yaml" %in% installed.packages()) install.packages("yaml")

# Parse CLI args: --run-id=<id> --run-date=<YYYY-MM-DD>
args <- commandArgs(trailingOnly = TRUE)
parse_arg <- function(name) {
  hit <- grep(paste0("^--", name, "="), args, value = TRUE)
  if (length(hit) == 0) stop(sprintf("Missing required arg: --%s", name))
  sub(paste0("^--", name, "="), "", hit[1])
}
parse_arg_opt <- function(name, default = NA) {
  hit <- grep(paste0("^--", name, "="), args, value = TRUE)
  if (length(hit) == 0) return(default)
  sub(paste0("^--", name, "="), "", hit[1])
}
run_id <- parse_arg("run-id")
run_date <- parse_arg("run-date")
sample_size <- suppressWarnings(as.integer(parse_arg_opt("sample-size")))
message(sprintf("run_id=%s, run_date=%s, sample_size=%s", run_id, run_date,
                ifelse(is.na(sample_size), "NA", sample_size)))

# Chemins lus dans le registre partagé (../contracts.yaml) plutôt qu'écrits en
# dur ici. La syntaxe {run_date}/{run_id} du registre est celle de glue::glue()
# comme celle de str.format() côté Python : aucune conversion.
contracts <- yaml::read_yaml("../contracts.yaml")
BUCKET <- Sys.getenv("COICOP_BUCKET", unset = contracts$bucket)
run_root <- glue::glue(contracts$run_root)

# `aws.s3` veut bucket et objet séparés, contrairement aux URI complètes des
# configs Python : le registre stocke donc des chemins relatifs.
artifact <- function(step, key) {
  rel <- contracts$steps[[step]]$outputs[[key]]
  if (is.null(rel)) stop(glue::glue("Artefact {step}.{key} absent de contracts.yaml"))
  glue::glue("{run_root}/{rel}")
}

path <- artifact("classify-regex", "test_without_regex")
sug_path <- artifact("classify-regex", "train_without_regex")
lcs_output_dir <- glue::glue("{run_root}/classify-lcs")
  
source("R/fonctions.R", encoding = "UTF-8")

con <- duckdb::dbConnect(duckdb::duckdb(), dbdir = ":memory:")
DBI::dbExecute(con, "INSTALL httpfs;")
DBI::dbExecute(con, "INSTALL icu; LOAD icu;")

DBI::dbExecute(con, sprintf("
  CREATE SECRET my_s3_secret (
    TYPE S3,
    KEY_ID '%s',
    SECRET '%s',
    ENDPOINT '%s',
    SESSION_TOKEN '%s',
    REGION 'us-east-1'
  )",
    Sys.getenv("AWS_ACCESS_KEY_ID"),
    Sys.getenv("AWS_SECRET_ACCESS_KEY"),
    Sys.getenv("AWS_S3_ENDPOINT"),
    Sys.getenv("AWS_SESSION_TOKEN")
))

###############################################################################'
# 1 - Import des tables --------------------------------------------------------

# on charge le jeu à coder (observations)
message(sprintf("Chargement observations : s3://%s/%s", BUCKET, path))
observations <- DBI::dbGetQuery(con, glue::glue(
        " SELECT *
          FROM read_parquet('s3://{BUCKET}/{path}')
        ")
    )

# Échantillonnage optionnel, directement sur le fichier d'input (pas de
# déduplication préalable). Même logique que classify-rag-notices : graine 42, sans remise.
if (!is.na(sample_size) && sample_size > 0 && sample_size < nrow(observations)) {
  set.seed(42)
  observations <- observations[sample(nrow(observations), sample_size), , drop = FALSE]
  message(sprintf("✓ Sampling applied: %d lignes", sample_size))
}

# En mode prédiction, le suggester est dans un fichier dédié (bypass classify-regex).
# On essaie ce fichier en premier ; si absent (mode normal), on lit depuis raw_train_without_regex.
#
# Ce chemin était FAUX : il cherchait `{run_root}/suggester.parquet` alors que
# build-datasets écrit sous `build-datasets/`. Le tryCatch ci-dessous avalait
# l'erreur, la branche n'était donc JAMAIS prise — sans un message. Le registre
# rend le décalage impossible : `artifact()` lève si la clé n'existe pas.
suggester_override <- artifact("build-datasets", "suggester")
use_override <- tryCatch({
  DBI::dbGetQuery(con, glue::glue(
    "SELECT COUNT(*) > 0 AS has_rows FROM read_parquet('s3://{BUCKET}/{suggester_override}')"
  ))$has_rows
}, error = function(e) FALSE)

if (isTRUE(use_override)) {
  message(sprintf("Chargement suggester (override) : s3://%s/%s", BUCKET, suggester_override))
  suggester <- DBI::dbGetQuery(con, glue::glue(
    "SELECT * FROM read_parquet('s3://{BUCKET}/{suggester_override}')"
  ))
} else {
  message(sprintf("Chargement suggester : s3://%s/%s (WHERE source = 'suggester')", BUCKET, sug_path))
  suggester <- DBI::dbGetQuery(con, glue::glue(
    "SELECT * FROM read_parquet('s3://{BUCKET}/{sug_path}') WHERE source = 'suggester'"
  ))
}
suggester$s_pr_product <- as.character(suggester$s_pr_product)
suggester$code <- as.character(suggester$code)

###############################################################################'
# 2 - Retraitements ------------------------------------------------------------

depenses <- observations |>
  dplyr::select(id, s_pr_product) |>
  dplyr::rename(product = s_pr_product)
depenses$s_pr_product |> unique() |> length() # 5533 produits différents sur 7804 lignes

liste_produits <- suggester |>
  dplyr::select(s_pr_product, code) |>
  dplyr::distinct()
liste_produits$s_pr_product |> unique() |> length() # 6498 produits différents sur 6609 lignes

###############################################################################'
# 3 - Analyse ------------------------------------------------------------------

# calcul de distances entre deux libellés
source("R/calcul_distances.R", encoding = "UTF-8")

# analyse de la codification avec la LCS
source("R/calcul_lcs_libel.R", encoding = "UTF-8")
