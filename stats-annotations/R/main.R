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

# Parse CLI args: --run-id=<id> --run-date=<YYYY-MM-DD>
args <- commandArgs(trailingOnly = TRUE)
parse_arg <- function(name) {
  hit <- grep(paste0("^--", name, "="), args, value = TRUE)
  if (length(hit) == 0) stop(sprintf("Missing required arg: --%s", name))
  sub(paste0("^--", name, "="), "", hit[1])
}
run_id <- parse_arg("run-id")
run_date <- parse_arg("run-date")
message(sprintf("run_id=%s, run_date=%s", run_id, run_date))

BUCKET <- "projet-budget-famille"
run_root <- glue::glue("data/workflow_runs/{run_date}/{run_id}")
path <- glue::glue("{run_root}/codif-regex/raw_test_without_regex.parquet")
sug_path <- glue::glue("{run_root}/codif-regex/raw_train_without_regex.parquet")
lcs_output_dir <- glue::glue("{run_root}/codif-lcs")
  
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

# on charge le jeu de test
data <- DBI::dbGetQuery(con, glue::glue(
        " SELECT *
          FROM read_parquet('s3://{BUCKET}/{path}')
        ")
    )

suggester <- DBI::dbGetQuery(con, glue::glue(
  " SELECT *
    FROM read_parquet('s3://{BUCKET}/{sug_path}')
    WHERE source = 'suggester'
        ")
)
suggester$s_pr_product <- as.character(suggester$s_pr_product)
suggester$code <- as.character(suggester$code)

###############################################################################'
# 2 - Retraitements ------------------------------------------------------------

depenses <- data |>
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
