# =============================================================================
# Sérialisation d'un modèle sirus (type = "classif") vers rules.json, le contrat
# lu par src/scorer.py.
#
# Sourcé par fit_sirus.R et par make_golden.R.
# =============================================================================

# ATTENTION PRÉCISION — vérifié contre le modèle de production :
#
#   valeur R                      0.2538615560640732349285
#   toJSON() défaut               0.2539                     <- catastrophique
#   toJSON(digits = NA)           0.253861556064073          <- ne round-trip PAS
#   sprintf("%.17g", x)           0.25386155606407323        <- round-trip (identical())
#
# `digits = NA` est le piège : ça ressemble à « précision maximale » et ça perd
# silencieusement les 2 derniers chiffres. Un seuil arrondi fait basculer les
# lignes situées exactement dessus dans la mauvaise branche de la règle.
#
# Ne pas « simplifier » g17() : le test tests/test_scorer_golden.py
# (test_threshold_strings_roundtrip) échouerait, et c'est voulu.
g17 <- function(x) vapply(x, function(v) sprintf("%.17g", v), character(1), USE.NAMES = FALSE)

# Les SEUILS, eux, sont déjà des chaînes dans l'objet modèle : get.rule() stocke
# `split[4]` verbatim et get.rule.support() fait `as.numeric()` dessus au
# scoring. On les recopie tels quels — zéro conversion, donc zéro perte
# possible, et Python fait le même `float()` sur la même chaîne.
export_rules <- function(m, factor_levels, extra = list()) {
  stopifnot(
    "export_rules ne gère que type = 'classif'" = identical(m$type, "classif")
  )

  regles <- lapply(seq_along(m$rules), function(k) {
    conditions <- lapply(m$rules[[k]], function(split) {
      if (split[2] %in% c("<", ">=")) {
        list(var = split[1], op = split[2], value = split[3])   # verbatim
      } else if (split[2] == "=") {
        # Condition factorielle : get.rule.support fait `X %in% split[3:n]`,
        # donc un test d'appartenance sur les LIBELLÉS de niveaux. On l'exporte
        # comme "in" et non "=" : "=" invite à l'implémenter comme une égalité.
        list(var = split[1], op = "in", levels = as.character(split[3:length(split)]))
      } else {
        stop(sprintf("opérateur de condition inattendu : %s", split[2]))
      }
    })
    list(
      conditions = conditions,
      outputs    = g17(m$rules.out[[k]]$outputs),
      supp_size  = as.integer(m$rules.out[[k]]$supp.size),
      # Fréquence d'apparition de la règle dans la forêt (m$proba). C'est LA
      # quantité centrale de SIRUS : une règle est retenue parce qu'elle revient
      # dans plus de p0 des arbres, et cette fréquence mesure donc sa stabilité
      # au rééchantillonnage. Sans elle, un relecteur de rules.json ne peut pas
      # distinguer une règle robuste (~0,30) d'une règle marginale (~0,03).
      # Diagnostic uniquement : le scoring ne l'utilise pas (moyenne non
      # pondérée), d'où un nombre JSON et non une chaîne.
      frequency  = m$proba[[k]]
    )
  })

  kinds <- vapply(
    m$bins,
    function(b) if (identical(b$type, "categorical")) "factor" else "numeric",
    character(1)
  )

  obj <- c(
    list(
      schema_version = 1L,
      type           = m$type,
      # Documenté explicitement pour couper court à la confusion récurrente :
      # en classification sirus.predict fait rowMeans(), sans poids. Le champ
      # m$rule.weights existe (rempli à 1/K) mais n'est JAMAIS lu au scoring.
      # La ridge à coefficients positifs est l'agrégation de la variante
      # régression. On n'exporte donc ni poids ni intercept : un champ inutilisé
      # invite à l'implémenter.
      aggregation    = "mean",
      features       = m$data.names,
      feature_kind   = as.list(kinds),
      factor_levels  = list(code_candidat_n1 = as.character(factor_levels)),
      mean           = g17(m$mean),
      rules          = regles,
      # Recopiées ici pour qu'un relecteur de rules.json voie contre quoi les
      # seuils ont été appris, sans avoir à ouvrir le code Python.
      sentinels      = list(conf_missing = "-1", dist_lcs_missing = "1.5")
    ),
    extra
  )

  jsonlite::toJSON(obj, auto_unbox = TRUE, pretty = TRUE, null = "null", digits = NA)
}
