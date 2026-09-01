###############################################################################'
# Calcul distances entre les libellés
###############################################################################'




# retraitement des produits labellisés et de ceux du suggester pour retirer les stopswords
# Table des libellés

DBI::dbWriteTable(con, 
                  "dep", 
                  depenses, 
                  overwrite = TRUE)

# Suggester
DBI::dbWriteTable(con, 
                  "sug", 
                  liste_produits, 
                  overwrite = TRUE)

# Compilation du noyau C++ (la fonction filtre en interne les paires distance > 0.8)
cache_dir <- file.path(getwd(), "rcpp_cache")
dir.create(cache_dir, showWarnings = FALSE, recursive = TRUE)
Rcpp::sourceCpp("./C/distance_gcd_batch_cpp.cpp", cacheDir = cache_dir)

# --- Produit cartésien calculé EN FLUX, par lots de libellés de test ---
# On NE matérialise PLUS l'intégralité du produit dep x sug (des dizaines de
# millions de lignes), qui faisait exploser la mémoire (OOMKilled) puis le disque
# (spill DuckDB -> "no space left on device").
#
# On itère sur les couples DISTINCT (id, product) du jeu de test. Comme chaque id
# n'apparaît alors que dans UN seul lot, l'union des lots est STRICTEMENT
# identique au produit global dédupliqué de l'ancienne version
# (même ensemble de paires (id, s1, s2, code) -> mêmes distances -> même résultat).
# Pour chaque lot on calcule la distance LCS et on n'accumule que les paires
# retenues (distance <= 0.8), peu nombreuses.
DBI::dbExecute(con, "CREATE OR REPLACE TABLE dep_u AS
                     SELECT DISTINCT id, product FROM dep")
DBI::dbExecute(con, "CREATE OR REPLACE TABLE dep_rn AS
                     SELECT *, row_number() OVER () AS rn FROM dep_u")
n_dep <- as.integer(DBI::dbGetQuery(con, "SELECT COUNT(*) AS n FROM dep_rn")$n)

dep_batch <- 200L  # libellés de test par lot (~ dep_batch x |sug| paires en mémoire à la fois)
cat(sprintf("Libellés de test distincts : %d, traités par lots de %d\n", n_dep, dep_batch))

t_deb <- Sys.time()

res_list <- list()
starts <- if (n_dep > 0L) seq(1L, n_dep, by = dep_batch) else integer(0)
for (start in starts) {
  end <- start + dep_batch - 1L
  df_batch <- DBI::dbGetQuery(con, sprintf("
    SELECT DISTINCT d.id,
                    d.product        AS s1,
                    sug.s_pr_product AS s2,
                    sug.code
    FROM dep_rn d
    CROSS JOIN sug
    WHERE d.rn BETWEEN %d AND %d
      AND d.product <> sug.s_pr_product", start, end))
  if (nrow(df_batch) > 0L) {
    res_list[[length(res_list) + 1L]] <-
      distance_gcd_batch_cpp(df_batch$id, df_batch$s1, df_batch$s2, df_batch$code)
  }
}

results <- dplyr::bind_rows(res_list)
if (ncol(results) == 0) {
  results <- data.frame(
    id               = character(),
    s1               = character(),
    s2               = character(),
    code             = character(),
    distance         = numeric(),
    prop_in_s1       = numeric(),
    prop_in_s2       = numeric(),
    common_substring = character()
  )
}

t_end <- Sys.time()
cat(sprintf("Temps de calcul : %s\n", format(t_end - t_deb)))
cat(sprintf("Résultats retenus (distance <= 0.8) : %s lignes\n",
            format(nrow(results), big.mark = " ")))

# ------- Séquentiel -------------

#t_deb <- Sys.time()
# Application de la fonction C++ ligne par ligne
#results <- purrr::pmap_dfr(df, distance_gcd_string_cpp)
#t_end <- Sys.time()
#sprintf("temps de calcul : %s",  t_end-t_deb)
