###############################################################################'
# Calcul distances entre les libellés
###############################################################################'

###############################################################################'
# Analyse des fichiers d'annotations de la campagne de test de l'enquête BdF
###############################################################################'

###############################################################################'
# 0 - Paramètres ---------------------------------------------------------------

source("fonctions.R", encoding = "UTF-8")

###############################################################################'
# 1 - Analyse ---------------------------------------------------------------

## Analyse des magasins renseignés ---------------------------------------------
# 2743 enseignes ou type de magasins renseignés dans le fichier d'annotations (sur 15695 lignes)
nb_magasin <- depenses |>
  remove_noise("store") |>
  dplyr::select(store) |>
  dplyr::distinct() |> 
  nrow()

table_occurences <- table(depenses$store) |> as.data.frame() |> dplyr::filter(Freq > 1)

# calcul de distance entre les magasins pour regrouper les problèmes d'orthographe (long à faire tourner) -------

# calcul du produit cartésien entre tous les noms d'enseigne
test <- expand.grid(depenses$store |> unique(), depenses$store |> unique(), stringsAsFactors = FALSE) |>
  dplyr::rename(s1 = Var1,
                s2 = Var2) |>
  dplyr::filter(s1 != s2) |>
  dplyr::mutate(dplyr::across(.cols = dplyr::everything(),.fns = ~stringi::stri_trans_general(.x, "Latin-ASCII"))) |>
  dplyr::mutate(dplyr::across(.cols = dplyr::everything(),.fns = ~stringr::str_trim(.x))) 

# calcul de la sous chaine commune la plus longue
t_deb <- Sys.time()
resultat <- test |> purrr::pmap_dfr(distance_gcd_string_cpp)
t_end <- Sys.time()

enseignes_communes <- resultat |>
  dplyr::filter(stringi::stri_length(common_substring) > 2)
