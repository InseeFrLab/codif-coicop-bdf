###############################################################################'
## Test ajout unités stopwords -------------------------------------------------

# indication de euros dans les tickets de caisse : EUR
# 151 libellé avec la chaine "EUR" pour euros
eur <- depenses |>
  remove_noise("libel_dep") |>
  dplyr::filter(libel_dep |> stringr::str_detect(pattern = "\\b(EUR)\\b"))
