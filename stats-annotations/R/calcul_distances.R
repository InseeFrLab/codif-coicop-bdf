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

# Mise en forme des libellés sous forme de produits cartésien pour analyser la disatnce de chaque libellé avec les autres
# calcul de distance par la plus longue sous chaine en commun entre deux libellé
system.time({
  DBI::dbExecute(con, "CREATE OR REPLACE TABLE libel_pc AS
                     SELECT DISTINCT dep.id,
                                    dep.product AS s1,
                                    sug.s_pr_product AS s2,
                                    sug.code
                    FROM dep
                    CROSS JOIN sug
                    WHERE dep.product <> sug.s_pr_product") 
})


# calcul de la sous chaine commune la plus longue entre deux libellés
df <- DBI::dbGetQuery(
  con,
  sprintf("
          SELECT *
          FROM libel_pc")
)

# --- Parallélisation ---
n_cores <- parallel::detectCores()
cat(sprintf("Cores disponibles : %d\n", n_cores))

chunk_size <- 500000L
chunks <- split(seq_len(nrow(df)), ceiling(seq_len(nrow(df)) / chunk_size))
cat(sprintf("Nombre de chunks : %d (de %s lignes chacun)\n",
            length(chunks), format(chunk_size, big.mark = " ")))

t_deb <- Sys.time()

cache_dir <- file.path(getwd(), "rcpp_cache")
dir.create(cache_dir, showWarnings = FALSE, recursive = TRUE)
Rcpp::sourceCpp("./C/distance_gcd_batch_cpp.cpp", cacheDir = cache_dir)

# Trouver le .so compilé
so_file <- list.files(cache_dir, pattern = "\\.so$", recursive = TRUE, full.names = TRUE)
cat("DLL trouvée :", so_file, "\n")

res_list <- parallel::mclapply(chunks, function(idx) {
  if (!is.loaded("sourceCpp_1_distance_gcd_batch_cpp")) {
    dyn.load(so_file)
  }
  distance_gcd_batch_cpp(
    df$id[idx], df$s1[idx], df$s2[idx], df$code[idx]
  )
}, mc.cores = n_cores)

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
