# decide-coicop

Arbitre LLM (« LLM-as-judge ») qui produit la **décision COICOP finale** du
pipeline Budget de Famille (étape Argo `decide-coicop`).

Il fusionne les prédictions des quatre classifiers puis tranche :

| Source | Colonne(s) lue(s) |
|---|---|
| `codif-lcs`               | `predict_code` (+ distance, sous-chaîne, proportion) |
| `run-rag`                 | `code_predict`, `confidence`, `codable` |
| `run-rag-annotations`     | `code_predict`, `confidence`, `codable` |
| `run-ttc`                 | `predicted_code` (+ top-2/3, confiances) |

## Fonctionnement

1. **Normalisation** — les codes LCS, TTC et RAG-annotations sont tronqués au
   niveau 4 et élagués des hiérarchies linéaires (via `mapping_lvl4.parquet`
   produit par l'étape `prune`) : consensus, arbitrage et scoring aval
   raisonnent tous sur le même espace de codes. Seul le RAG notices garantit
   structurellement des codes prunés en amont. La vérité terrain, elle, est
   conservée **brute** dans `code` et dupliquée sous forme canonique dans
   **`code_lvl4`** — la sortie porte les deux. La logique est réutilisée du
   module [`prune`](../prune/) (aucune duplication). Le prompt du juge ne
   contient **pas** le code de référence annoté (pas de fuite de la vérité
   terrain en mode évaluation).
2. **Court-circuit consensus** — si LCS = RAG = RAG-annotations = TTC et que la
   confiance TTC ≥ 0,90, le code est retenu **sans appel LLM**.
3. **Arbitrage LLM** — sinon, un LLM tranche à partir du contexte d'achat et des
   prédictions (nomenclature filtrée par défaut, complète avec
   `--full-nomenclature`).

Reprise (`resume`) : relancer avec le même fichier de sortie reprend là où le
run précédent s'était arrêté.

## Installation / usage

Python ≥ 3.13. Paquets via le proxy Nexus INSEE (`[tool.uv]`).

```bash
uv sync
uv run main.py decide-coicop \
  --lcs-file  <s3://.../codif-lcs/raw_test_LCS.parquet> \
  --rag-file  <s3://.../run-rag/predictions.parquet> \
  --rag-annotations-file <s3://.../rag-annotation/predictions.parquet> \
  --ttc-file  <s3://.../run-ttc/predictions.parquet> \
  --mapping-file <s3://.../prune/mapping_lvl4.parquet> \
  --output-file <s3://.../decide-coicop/predictions.parquet> \
  --concurrency 5
```

## Secret / variables d'environnement

- `LLMLAB_API_KEY` (requis), `LLMLAB_URL` (optionnel) — endpoint LLM.
- Identifiants AWS/S3 pour lire/écrire les parquets sur MinIO.

La sortie `decide-coicop/predictions.parquet` contient les codes prédits
normalisés de chaque classifier et la décision (`llm_code`, `llm_model`,
`llm_confiance`, `llm_explication`). `llm_code` est lui aussi tronqué+élagué
en fin de batch (quand `--mapping-file` est fourni) ; le garde-fou de l'étape
aval [`final-output`](../final-output/) est donc une redondance idempotente.

En mode évaluation, deux colonnes portent la vérité terrain :

| Colonne | Contenu |
|---|---|
| `code` | code annoté **brut**, tel que saisi (souvent niveau 5) |
| `code_lvl4` | sa forme **canonique** : tronquée au niveau 4 puis élaguée — c'est la cible du scoring du [`report`](../report/) |
