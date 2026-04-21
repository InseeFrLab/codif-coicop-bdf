###############################################################################'
# Fonctions
###############################################################################'

###############################################################################'
# Calcul des distances ---------------------------------------------
###############################################################################'

#' Calcul de la distance du plus grand sous-mot commun (longest commun substring)
#' @param s1 chaine de character 1
#' @param s2 chaine de character 2
#' @return liste comprenant le score LCS et la plus longue sous chaine de character en commun entre s1 et s2
distance_gcd_string <- function(s1, s2, p) {
  stopifnot(is.character(s1) && is.character(s2))
  n1 <- nchar(s1); n2 <- nchar(s2)
  m <- matrix(0, n1 + 1, n2 + 1)
  maxlen <- 0
  endpos <- 0
  for (i in seq_len(n1)) {
    for (j in seq_len(n2)) {
      if (substr(s1, i, i) |> stringr::str_to_upper() == substr(s2, j, j) |> stringr::str_to_upper()) {
        m[i + 1, j + 1] <- m[i, j] + 1
        if (m[i + 1, j + 1] > maxlen) {
          maxlen <- m[i + 1, j + 1]
          endpos <- i
        }
      }
    }
  }
  substring_common <- if (maxlen > 0) substr(s1, endpos - maxlen + 1, endpos) else ""
  distance <- 1 - maxlen / max(n1, n2)
  
  return(list(distance = distance, common_substring = substring_common))
}

# version C

Rcpp::cppFunction('
List distance_gcd_string_cpp(std::string id, std::string s1, std::string s2, std::string code) {
  int n1 = s1.size();
  int n2 = s2.size();
  
  // --- Longest Common Substring ---
  std::vector<std::vector<int>> m(n1 + 1, std::vector<int>(n2 + 1, 0));
  int maxlen = 0;
  int endpos = 0;

  for (int i = 0; i < n1; i++) {
    for (int j = 0; j < n2; j++) {
      if (toupper(s1[i]) == toupper(s2[j])) {
        m[i + 1][j + 1] = m[i][j] + 1;
        if (m[i + 1][j + 1] > maxlen) {
          maxlen = m[i + 1][j + 1];
          endpos = i;
        }
      }
    }
  }

  std::string substring_common = (maxlen > 0) ? s1.substr(endpos - maxlen + 1, maxlen) : "";
  double distance = 1.0 - ((double)maxlen / std::max(n1, n2));
  
  // proportion de s2 couverte par la common substring
  double prop_in_s2 = (n2 > 0) ? ((double)maxlen / n2) : 0.0;
  
  // Si distance > 0.8 → on retourne une ligne vide, qui sera ignorée
  if (distance > 0.8) {
    return Rcpp::List();
  }
  
  return List::create(_["id"] = id,
                      _["s1"] = s1,
                      _["s2"] = s2,
                      _["code"] = code,
                      _["distance"] = distance,
                      _["common_substring"] = substring_common,
                      _["prop_in_s2"] = prop_in_s2);
}
')

###############################################################################'
# Annotation ------------------------------------------------------
###############################################################################'

#' Fonction d'annotation des erreurs comises par le modèle
#' @param df data.frame comportant le résultat
#' @return data.frame comportant les annnotations
#' @import dplyr, glue
codif_console <- function(df){
  vars_to_print <- c("product_libel", "coicop_libel", "product_sug", "code", "predict_ok")
  
  df <- df |>
    dplyr::mutate(coicop_trop_precis = NA_character_)
  
  for (i in seq_len(nrow(df))) {
    ligne <- df[i, vars_to_print] |> as.list()
    
    # print propre
    cat(glue::glue("{ligne[[1]]} | {ligne[[2]]} | {ligne[[3]]} | {ligne[[4]]} | {ligne[[5]]}\n"))
    
    saisie <- readline(prompt = 'Entrer un code (ou taper "STOP" pour arrêter) : ')
    
    if (toupper(saisie) == "STOP") {
      cat("Arrêt demandé. Fin du remplissage.\n")
      break
    }
    
    df$coicop_trop_precis[i] <- saisie
  }
  return(df)
}

###############################################################################'
# Graphiques ------------------------------------------------------------------
###############################################################################'

#' Graphique qui affiche les évolutions du nombre de bonnes pred et le nombre de libellés selon le taux prop_in_s2
#' @param df data.frame comprenant les infos sur la prédiction LCS
graph_bon_pred_lcs <- function(df){
  df <- df |> dplyr::filter(predict_ok == 1)
  ggplot2::ggplot(df, ggplot2::aes(x = prop)) +
    # Série 1 : taux pour predict_ok = 1
    ggplot2::geom_line(
      data = df,
      ggplot2::aes(y = Freq / libel_tot, color = "Taux de bonnes prédictions"),
      linewidth = 1
    ) +
    # Série 2 : libel_tot (axe secondaire)
    ggplot2::geom_line(
      ggplot2::aes(y = libel_tot/ max(libel_tot), color = "Nb de libellés"),
      linewidth = 1,
      linetype = "dashed"
    ) +
    ggplot2::scale_y_continuous(
      name = "Taux de bonnes prédictions",
      limits = c(0, 1), 
      sec.axis = ggplot2::sec_axis(~ . * max(df$libel_tot),
                                   name = "Nb de libellés")
    ) +
    ggplot2::scale_color_manual(values = c("Taux de bonnes prédictions" = "blue",
                                           "Nb de libellés" = "red")) +
    ggplot2::labs(x = "prop", color = "") +
    ggplot2::theme_minimal()
  
}

###############################################################################'
# Evaluation / calcul de métriques ---------------------------------
###############################################################################'

#' Fonction de comparaison de codes COICOP entre la prédiction et le vrai code
#' @param coicop1
#' @param coicop2
#' @return chaine de caractère du code en commun
compar_coicop <- function(coicop1, coicop2){
  if(is.na(coicop1) | is.na(coicop2)){
    return(NA_character_)
  }
  mchar <- min(nchar(coicop1), nchar(coicop2))
  steps <- seq(from = 2, to = 12, by = 2)
  code_com <- NA_character_
  for (k in steps) {
    if (substr(coicop1, 1, k) == substr(coicop2, 1, k)) {
      code_com <- substr(coicop1, 1, k)
    } else {
      return(code_com)
    }
  }
  return(code_com)
}

#' Comptage des bonnes et mauvaises prédictions à partir d'un seuil de similarité entre la LCS et le libellé du suggester
#' @param df data.frame comportant les valeurs prédictes et la comparaison donnée prédite / donnée réelle
#' @param seuil_prop valeur du seuil d'analyse de la variable prop_in_s2 entre 0 et 1)
#' @return tablea de contingence avec les bonnes et mauvaises prédictions
comptage_pred_prop_in_s2 <- function(df, seuil_prop){
  tableau <- table(df[df$prop_in_s2 >=seuil_prop, "predict_ok"], useNA = "ifany") |> 
    as.data.frame() |> 
    dplyr::mutate(prop = seuil_prop)
  tableau$libel_tot <- sum(tableau$Freq, na.rm = TRUE)
  return(tableau)
}

#' Fonction qui analyse si la prédiction est égale à la valeur réelle sur le niveau de la coicop défini en entrée
#' @param df data.frame comportant les libellés comparés et la varibale de comparaison (code coicop en commun)
#' @param var_compar variable de comparaison entre deux code coicop
#' @param var_long_max_coicop variable de comparaison entre deux code coicop
#' @param lvl_coicop niveau de la coicop à comparer
#' @return data.framme comportant la variable predictive
#' @import dplyr, stringr
eval_pred_nb_pos_coicop <- function(df, var_compar, var_long_max_coicop, lvl_coicop){
  stopifnot(var_compar |> is.character())
  result <- df |>
    dplyr::mutate(
      comp_tmp = stringr::str_sub(.data[[var_compar]], 1, lvl_coicop),
      valid_len = nchar(comp_tmp) == lvl_coicop | nchar(.data[[var_compar]]) == .data[[var_long_max_coicop]] ,
      predict_ok = as.integer(!is.na(comp_tmp) & valid_len)
    )
  return(result)
}


#' Calcul taux de bonnes prédictions par code COICOP
#' @param df data.frame avec les variables code et predict_ok
#' @return pour chaque code coicop, le taux de bonnes prédictions
calcul_tx_pred_code <- function(df){
  tx_pred_code <- df |>
    dplyr::group_by(code, predict_ok)|>
    dplyr::summarise(total_cat = dplyr::n()) |>
    dplyr::ungroup() |>
    tidyr::complete(code, predict_ok = c(0,1)) |>
    dplyr::mutate(total_cat = tidyr::replace_na(total_cat, 0)) |>
    dplyr::left_join(df |>
                       dplyr::group_by(code)|>
                       dplyr::summarise(total = dplyr::n()), by = "code") |>
    dplyr::mutate(tx_bonne_pred = round(total_cat/total*100, 0)) |>
    dplyr::filter(predict_ok == 1)
  
  return(tx_pred_code)
}

#' Calcul le df avec le taux de bonne prédiction pour chaque code selon le niveau du code souhaité
#' @param df data.frame avec les variables code et predict_ok
#' @param taille longueur souhaité d'analyse de la variable "code" (nombres pairs de 2 à 12)
#' @return df avec le taux de bonnes prédictions par code
#' @import dplyr, stringr
calcul_df_bonne_pred_code <- function(df, taille){
  calcul <- df |> dplyr::select(code, predict_ok) |>
           dplyr::mutate(code = code |> stringr::str_sub(1, taille)) |>
           dplyr::mutate(predict_ok = tidyr::replace_na(predict_ok, 0)) |>
           calcul_tx_pred_code()
  
  }

#' Calcul les accuracy pour chaque niveau de la coicop
#' @param df
#' @return data.frame avec les accuracy pour chaque niveau
#' @import purrr, dplyr, stringr
calcul_accuracy_tous_lvl <- function(df){
  steps <- seq(from = 2, to = 12, by = 2)
  pred_coicop.list <- purrr::map(steps,
                                 ~ eval_pred_nb_pos_coicop(df = df,
                                                           var_compar = "comparaison",
                                                           var_long_max_coicop = "long_max_coicop",
                                                           .x))
  
  names(pred_coicop.list) <- paste0("coicop_pos_", c(1:6))
  
  # on calcule la table avec les accuracy sur tous les niveaux
  tab_accuracy <- purrr::map2_dfr(pred_coicop.list, names(pred_coicop.list), .f = ~ comptage_pred_prop_in_s2(df = .x, seuil_prop = 0) |>
                                       dplyr::mutate(nb_pos = .y,) |>
                                       dplyr::group_by(predict_ok, nb_pos) |>
                                       dplyr::mutate(
                                         accuracy = dplyr::if_else(predict_ok == 1,
                                                                   Freq / libel_tot,
                                                                   NA_real_)
                                       )) |> 
    dplyr::select(nb_pos, accuracy, predict_ok, Freq, libel_tot) |>
    dplyr::filter(predict_ok == 1)
  # on calcule les accuracy par niveau de coicop
  acc_by_lvl <- purrr::map2(pred_coicop.list, steps, calcul_df_bonne_pred_code)
  return(list(tab_accuracy = tab_accuracy, acc_by_lvl = acc_by_lvl))
}

###############################################################################'
# Fonctions d'import / export --------------------------------------------------
###############################################################################'

#' Lecture d'un fichier CSV avec détection auto du délimiter (, ou ;)
#' @param path chemin du fichier à importer
#' @return data.frame comportant les donnes importées
#' @import readr
lire_csv_auto <- function(path) {
  # Lire quelques lignes
  lignes <- readLines(path, n = 5)
  # Détection simple du séparateur
  sep <- if (sum(grepl(";", lignes)) > sum(grepl(",", lignes))) ";" else ","
  # Lecture
  readr::read_delim(path, delim = sep, show_col_types = FALSE)
}

#' Export de df dans le S3
#' @param df
#' @param output_names
#' @import aws.S3, stringr, arrow
export_liste_df <- function(df, output_names){
  aws.s3::s3write_using(
    x = df,
    FUN = arrow::write_parquet,
    object = stringr::str_glue("{lcs_output_dir}/eval/{output_names}.parquet"),
    bucket = BUCKET,
    opts = list("region" = "")
  )
}
