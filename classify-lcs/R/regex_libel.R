pattern_code_df <- tibble::tibble(
  pattern = c(
    "fruits? et l[eé]gumes?",
    "^l[eéèêë]gum[eéèêë]s?$",
    "^fruits?$",
    "\\b(divers\\s+)?courses?\\b",
    "^\\s*boulangerie\\s*$",
    "^\\s*billeterie\\s*$",
    "^\\s*restaurant\\s*$",
    "^\\s*resto$",
    "^carte bancaire$",
    "^alimentation?$",
    "^alimentaire$",
    "^courses alimentaires$",
    "^courses?$",
    "^reductions?.*",
    "^remises?.*",
    "^nourriture$",
    "^boissons?$",
    "^prelevement$",
    "^-10 % abonnement*",
    "^divers$",
    "^epicerie$",
    "^avantage carte 1028$",
    "^bon immediat$",
    "^rabais 30 %$",
    "^illisible$",
    "^[^a-zA-Z]*$",
    "^cantine$",
    "^cb$",
    "^marche$",
    "^surgeles?$",
    "^retrait$",
    "^boucher$",
    "^repas.*",
    "^article divers.*",
    "\\bdrive\\b",
    "\\bcantine\\b",
    "\\b^dejeune[s|r|rs]\\b"
  ),
  code = c(
    "01.1", "01.1.7", "01.1.6", "98.1", "01.1.1.3", "09.4",
    "11.1.1", "11.1.1", "98.3", "98.1.1", "98.1.1", "98.1.1",
    "98.1", "98.5", "98.5", "98.1", "98.1", "98.4", "98.5",
    "98.2", "98.1.1", "99", "98.5", "98.5", "98.4", "98.4",
    "11.1.2.1", "98", "98.1.1", "98.1.1", "99.2", "01.1.2.2",
    "11.1.1","98.1", "98.1", "11.1.2", "11.1.1"
  )
)


extract_reg <- extract_res |>
  dplyr::distinct() |>
  dplyr::mutate(
    code_regex = purrr::map_chr(
      product_libel,
      \(xx) {
        i <- which(
          stringr::str_detect(xx, stringr::regex(pattern_code_df$pattern, TRUE))
        )[1]
        if (is.na(i)) NA_character_ else pattern_code_df$code[i]
      }
    )
  )

extract_res_reg <- extract_reg |>
  dplyr::filter(is.na(code_regex))

# on regarde tous les libellés qui ont été codés par regex

cod_regex <- extract_reg |>
  dplyr::filter(!is.na(code_regex))

table(cod_regex$Origine)
