# stats-annotations

## Structure du dépôt

stats-annotations/
├── R
    ├── main.R # Script principal pour le calcul de la codification déterministe
    ├── analyse_libel.R # calculs d'indicateurs sur les doublons dans le fichier d'annotation et le suggester
    ├── calcul_distances.R # Script de calcul des distances de Levanshtein et de la LCS
    ├── calcul_lcs_libel.R # Script de codification des libellés dans la nomenclature
    ├── fonctions.R # fonctions principales utilisées
    ├── regex_libel # Script d'application des regex aux libellés issus du fichier d'annotation
    ├── analyse_magasins.R # Script d'analyse des noms des magasins
    ├── test_stopwords.R # Analyse de certains stopwords présents dans les libellés