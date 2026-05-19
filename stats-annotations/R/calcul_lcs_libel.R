###############################################################################'
# Analyse prédictions distance LCS
###############################################################################'

###############################################################################'
# 0 - Test égalité stricte -------
egal_str <- data |>
  dplyr::select(id, source, annee, raw_product, s_pr_product, l_pr_product, shop, shop_type_name, budget, code) |>
  dplyr::inner_join(liste_produits |> dplyr::select(s_pr_product, code), by = "s_pr_product", suffix = c("", "_lcs")) |>
  dplyr::filter(s_pr_product != "")

egal_str$s_pr_product |> unique() |> length() # 498 produits identiques 
egal_str$id |> unique() |> length() # 1230 observations avec une égalité entre test set et suggester -> 15.8%
egal_str |> dplyr::filter(code != code_lcs) |> dplyr::select(s_pr_product, code, code_lcs)

# On doit mettre en forme les égalités strictes pour les ajouter à la table finale
egal_str <- egal_str |>
  dplyr::mutate(s1 = s_pr_product,
                s2 = s_pr_product,
                distance = 0.0,
                prop_in_s1 = 1.0,
                prop_in_s2 = 1.0,
                common_substring = s_pr_product,
                is_close_suggester = TRUE)
# il faut traiter les doublons d'id
egal_str <- egal_str[!duplicated(egal_str$id), ]

###############################################################################'
# 1 - Calcul distance LCS min pour chaque libellé ------
# on regarde d'abord les libellés dont plusieurs libellés du suggester ont un score minimal identique
plusieurs_res <- results |>
  dplyr::slice_min(order_by = distance, by = id, with_ties = TRUE) |>
  dplyr::add_count(id) |>
  dplyr::filter(n > 1) |>
  dplyr::select(-n)
plusieurs_res$s1 |> unique() |> length() # 2794 libels ayant au moins un doublon sur 7804 : 36% des libellés
plusieurs_res$s1 |> table() |> table()

# On va filter les résultats pour ne garder que le code final prédit pour les libellés
# à partir du suggester
extract_res <- results |>
  dplyr::arrange(id, distance, dplyr::desc(prop_in_s2)) |>
  dplyr::group_by(id) |>
  dplyr::slice_head(n = 1) |> # il peut y avoir ds doublons, alors on prend le premier de la liste
  dplyr::ungroup() |> 
  dplyr::left_join(data |> dplyr::select(id, source, annee, raw_product, s_pr_product, l_pr_product, shop, shop_type_name, budget, code), by = "id", suffix = c("_lcs", "")) |>
  dplyr::bind_rows(egal_str) |> # on ajoute les égalités strictes, qui n'ont pas été ajoutées
  dplyr::group_by(id) |>
  dplyr::slice_min(distance, n = 1) |>
  dplyr::ungroup()
extract_res$id |> unique() |> length() # 6231 libels ayant été classés par la méthode LCS

# Redressement 01/11 : la LCS a naturellement tendence à classer les libellés en 01 même s'ils ont été achétés dans un restaurant (catégorie 11)
# On applique donc un redressement 01 -> 11 en regardant le type de magasin
extract_res <- extract_res |>
  dplyr::mutate(code_lcs = dplyr::case_when(
    stringr::str_detect(shop_type_name, "Restauration professionnelle") ~ "11.1.2",
    stringr::str_detect(shop_type_name, "Restauration rapide") ~ "11.1.1.2",
    stringr::str_detect(shop_type_name, "Bars, cafés, salons de thé, glaciers") ~ "11.1.1.2",
    stringr::str_detect(shop_type_name, "Restauration classique") ~ "11.1.1.1",
    TRUE ~ code_lcs
  ))

# regardons les libellés qui n'ont pas été classés par la LCS
no_class_lcs <- data |>
  dplyr::anti_join(extract_res |> dplyr::select(id), by = "id")
no_class_lcs |> nrow() # 51 libels ayant été classés par la méthode LCS

# table finale à exporter
output_lcs <- extract_res |>
  dplyr::select(id, source, annee, raw_product, l_pr_product, s_pr_product, shop, shop_type_name, budget, code, code_lcs, s1, s2, distance, common_substring, prop_in_s1, prop_in_s2) |>
  dplyr::mutate(is_close_suggester = dplyr::if_else(prop_in_s2>=0.8, TRUE, FALSE)) |>
  dplyr::bind_rows(no_class_lcs |> dplyr::select(-n_obs, -coicop)) |>
  dplyr::mutate(method = "LCS") |>
  dplyr::rename(predict_code = code_lcs)

# Export de la table de comparaison avec la règle LCS
aws.s3::s3write_using(
  x = output_lcs,
  FUN = arrow::write_parquet,
  object = glue::glue("{lcs_output_dir}/raw_test_LCS.parquet"),
  bucket = BUCKET,
  opts = list("region" = "")
)

###############################################################################'
# 2 - Analyse des bonnes prédictions selon la valeur de prop_in_s2 ------

# on va comparer les codes entre le code prédit et le code annoté manuellement
comparaison <- output_lcs |>
  dplyr::mutate(
    comparaison = purrr::map2_chr(code, predict_code, compar_coicop),
    long_max_coicop = purrr::map2_int(code, predict_code, ~ { m <- max(nchar(.x), nchar(.y), na.rm = TRUE); if (is.infinite(m)) 0L else as.integer(m) }) # variable permettant de récupérer les bonne pred sur des positions inférieures
  ) |>
  dplyr::distinct() |> 
  dplyr::mutate(predict_ok = dplyr::if_else(code == comparaison, 1, 0))

table(comparaison$predict_ok) # 1991 bonne pred et 2926 erreurs
comparaison[is.na(comparaison$predict_ok),] |> nrow() # 2887 NA
# soit une accuracy de 3892/10195
# quelques chiffres en graphique :
# on simule, pour différentes valeurs du seuil prop_ins_2, l'évolution des bonnes pred

# on exporte la table comparaison, qui sert de base de travail pour l'analyse
aws.s3::s3write_using(
  x = comparaison,
  FUN = arrow::write_parquet,
  object = glue::glue("{lcs_output_dir}/analyse_codif_LCS.parquet"),
  bucket = BUCKET,
  opts = list("region" = "")
)

purrr::map_dfr(.x = seq(0, 1, length.out = 100), .f = ~ comptage_pred_prop_in_s2(df = comparaison, seuil_prop = .x)) |> graph_bon_pred_lcs()

## Analyse 1 : on regarde les libellés qui ont prop_s2 = 1 (la LCS entre le libellé et le suggester est égal au suggester) ----

sugg_compl <- comparaison |>
  dplyr::filter(prop_in_s2 == 1)
table(sugg_compl$predict_ok) # 983 mauvaise pred et 1243 bonne pred
sugg_compl[is.na(sugg_compl$predict_ok),] |> nrow() # 283 NA

# liste des libellés avec aucun point commun entre le code annoté et celui du suggester
sugg_compl[is.na(sugg_compl$predict_ok), "s1"]

# sugg_annote <- codif_console(sugg_compl)
# - confusion entre le produit et le nom du restaurant : 1 PIZZA -> restaurant ou produit ? -> voir avec nom du commerce (et type de commerce ?)

## Analyse 2 : on fixe le seuil à prop_in_s2 = 0.8 ----

sugg_08 <- comparaison |>
  dplyr::filter(prop_in_s2 >= 0.8)
table(sugg_08$predict_ok) # 1201 mauvaise pred et 1348 bonne pred
sugg_08[is.na(sugg_08$predict_ok),] |> nrow() # 524 NA

## Analyse 3 : on regarde les bonnes prédictions sur les différents niveaux de nomenclature ----
tab_acc_all <- calcul_accuracy_tous_lvl(comparaison)[[1]]

tab_acc_08 <- calcul_accuracy_tous_lvl(comparaison |> dplyr::filter(prop_in_s2 > 0.8))[[1]]

# 71 % d'accuracy sur 4 pos avec prop_in_s2 >0.8
# contre 38% d'accuracy sur 4 pos avec tous les libellés à classer

list(tab_acc_all, tab_acc_08) |> 
  purrr::map2(c("tab_acc_all", "tab_acc_08"), export_liste_df)

## Analyse 4 : Regarder le taux de bonne prédiction selon chaque code COICOP ----

bonne_pred_code <- calcul_accuracy_tous_lvl(comparaison)[[2]]

# Export de la table de comparaison pour une égalité stricte
bonne_pred_code |>
  purrr::map2(names(bonne_pred_code), export_liste_df)

## Analyse 5 : Origine des tickets



###############################################################################'
# 3 - Analyse des erreurs de prédiction ----------------------------------------

# Premier temps: les erreurs totales de pred (aucun niveau commun)
erreur_pred <- comparaison |>
  dplyr::filter(is.na(predict_ok))

df_lcs <- erreur_pred$common_substring |> table() |> as.data.frame() |>
  dplyr::mutate(Var1 = stringr::str_trim(Var1)) |>
  dplyr::group_by(Var1) |>
  dplyr::summarise(nb = sum(Freq, na.rm = T))

## cas 1 : restau (11) au lieu d'alimentaire (01)
erreur_pred_restau <- erreur_pred |>
  dplyr::filter(stringr::str_sub(code, 1, 2) == "11")

# Deuxième temps: les erreurs partielles de pred (au moins un niveau commun)
erreur_partielle <- comparaison |>
  dplyr::filter(predict_ok == 0)

## Origine des tickets
recap_origine <- table(erreur_pred$source) |> as.data.frame() |> # erreur totale
  dplyr::rename(erreur_totale = Freq) |>
  dplyr::inner_join(table(erreur_partielle$source) |> as.data.frame() |> # erreur partielle
                      dplyr::rename(erreur_partielle = Freq), by = "Var1") |>
  dplyr::inner_join(table(comparaison |> dplyr::filter(predict_ok == 1) |> dplyr::select(source)) |> # bonne prediction
                      as.data.frame() |> dplyr::rename(bonne_pred = Freq), 
                    by = c("Var1" = "source")) |>
  dplyr::inner_join(table(comparaison$source) |> as.data.frame() |> # total
                      dplyr::rename(total = Freq), by = "Var1") |> 
  dplyr::rename(origine_libel = Var1)

