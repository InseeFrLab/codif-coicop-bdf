# Upgrade Plan: `coicop-bdf-classifier` from torchTextClassifiers v1.0.4 → v2.0.0

**Scope:** the `coicop-bdf-classifier/` sub-project of `codif-coicop-bdf`, its Docker image, and the `run-ttc` / `decide-coicop` steps of the Argo pipeline.
**Basis:** verified API diff between the published `torchtextclassifiers==1.0.4` wheel (currently pinned in `uv.lock`) and the `v2.0.0` tag of `InseeFrLab/torchTextClassifiers`.

---

## 0. Why this upgrade changes the *model*, not just the dependency

v2.0.0 ships fixes that invalidate models trained under v1:

| v2 change (PR)                                                   | Effect on the COICOP model                                                                                                                                                                         |
|------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Validation data never checked (#91)                              | Early stopping (`patience=5`) and best-checkpoint selection were blind under v1 → **all deployed checkpoints must be retrained**, not just reloaded                                                |
| LabelAttention mask fixes (#74, #92, #96)                        | Padding tokens contaminated attention. With `max_seq_length=64` and very short `description_ean` labels, most of each sequence is padding → accuracy + explanation quality of the multi-head model |
| `predict` device alignment (#94)                                 | Faster GPU batch inference for `run-ttc` (`predict_batch_size=512`)                                                                                                                                |
| Captum device fix (#97)                                          | Restores gradient attributions                                                                                                                                                                     |
| S3 tokenizer load/save (#71, #73)                                | Pipeline artifacts live on S3                                                                                                                                                                      |
| `sample_weights` (#100)                                          | New lever for COICOP class imbalance and heterogeneous sources (DDC / COPAIN / synthetic)                                                                                                          |
| `from_model` + `contrib.MultiLevelTextClassificationModel` (#90) | Supported replacement for the hand-rolled 952-line `multihead_classifier.py`                                                                                                                       |

Consequence: the plan below treats **retraining as mandatory** and defines acceptance gates against the current production model.

---

## 1. Verified breaking changes (what will actually fail)

### 1.1 `ModelConfig`
- v1: `label_attention_config: Optional[LabelAttentionConfig]`
- v2: field removed; replaced by `n_heads_label_attention: Optional[int]` and `aggregation_method: Optional[str] = "mean"`. `num_classes` now accepts `int | list[int]` (multi-level support).
- **Impact:** any config dict pickled/serialized under v1 with `label_attention_config` will not round-trip through `ModelConfig.from_dict`. `basic_classifier.py` and `hierarchical_classifier.py` don't set that field, so their `ModelConfig(...)` constructor calls survive unchanged.

### 1.2 `TrainingConfig` — silent behavior change, then loud runtime error
- v2 adds `raw_categorical_inputs: bool = True` and `raw_labels: bool = True` **defaulting to True**.
- All three classifiers pass **integer-encoded** labels (`label_to_idx`) and, for the hierarchical model, integer categorical features. With `raw_labels=True` and no `ValueEncoder`, v2's `_check_Y` raises: *"Raw label encoding is enabled, but no value_encoder was provided."* Same for categorical inputs in the hierarchical cascade.
- **Fix (minimal path):** add `raw_labels=False` (and `raw_categorical_inputs=False` where categorical features are used) to every `TrainingConfig(...)` construction.

### 1.3 `predict()` signature
- v2: `predict(self, X_test, raw_categorical_inputs: bool = True, top_k=1, ..., device='cpu')`.
- A new positional parameter (`raw_categorical_inputs`) is inserted **before** `top_k`. All call sites in the repo use `top_k=` as a keyword, so no positional breakage — but every hierarchical `predict` call must now pass `raw_categorical_inputs=False`, and all call sites should pass `device=` explicitly (`"cuda"` when available) to benefit from #94.
- If a `ValueEncoder` is attached, predictions come back **inverse-transformed to raw labels (strings)** instead of integer indices — a return-type change that ripples into `predict.py`, `api.py` (`serve`), and `decide_coicop.py`. This is why ValueEncoder adoption is a separate, optional phase (Phase 5).

### 1.4 Components module — the multi-head classifier breaks at import time
- v1 `model.components.text_embedder` exports: `TextEmbedderConfig`, `TextEmbedder`, `LabelAttentionClassifier`.
- v2 replaces them with: `TokenEmbedderConfig`, `SentenceEmbedderConfig`, `TokenEmbedder`, `SentenceEmbedder`, `LabelAttention`.
- `multihead_classifier.py` imports `TextEmbedderConfig` and `LabelAttentionClassifier` directly → **ImportError under v2**. It also imports `Block` and `norm` from `model.components.attention` (still present in v2, verify signatures during Phase 4).

### 1.5 Checkpoint portability
- Module renames (`TextEmbedder` → `TokenEmbedder`/`SentenceEmbedder`) mean v1 `state_dict` keys will not match v2 module trees. Do **not** attempt checkpoint surgery; retrain (required anyway per §0).

### 1.6 Environment
- v2 `requires-python >= 3.11`; the project pins `>= 3.13` → compatible. v2's own CI moved to Python 3.14; staying on 3.13 is fine.

---

## 2. Phase plan

### Phase 0 — Freeze a baseline (½ day)
1. Tag the current repo state and the current model artifacts on S3 (`baseline-ttc-v1`).
2. Record baseline metrics using the existing evaluation path: per-level COICOP accuracy (levels 1–5), and — from the pipeline's `report` step — the **consensus rate** in `decide-coicop` (share of observations where LCS/RAG/TTC agree with TTC confidence ≥ 0.90) and resulting LLM call volume. These are the business KPIs the upgrade must not regress.
3. Freeze a fixed evaluation split (stratified by COICOP level-4 code) and store it on S3 so v1 and v2 models are compared on identical data.

### Phase 1 — Dependency bump on a branch (½ day)
1. Branch `upgrade/ttc-v2`.
2. `pyproject.toml`: `torchtextclassifiers>=2.0.0,<3`; run `uv lock && uv sync`.
3. Run the test suite. Expected failures: `multihead_classifier` import error (§1.4); `train()` runtime errors from `raw_labels` (§1.2). Catalogue anything unexpected.

### Phase 2 — Minimal compatibility: basic + hierarchical classifiers (1–2 days)
Goal: v1-equivalent behavior under v2, no functional redesign.
1. `basic_classifier.py`: add `raw_labels=False` to `TrainingConfig`; add `device=` (auto-detect CUDA) to the `predict` call.
2. `hierarchical_classifier.py`: add `raw_labels=False, raw_categorical_inputs=False` to `TrainingConfig`; add `raw_categorical_inputs=False, device=...` to all six `predict` call sites (including `_batched_predict` and beam search).
3. Save/load round-trip test for both classifiers: `torchTextClassifiers.load()` in v2 additionally looks for `value_encoder.pkl` via a `has_value_encoder` metadata flag — verify loading a freshly v2-saved model works, and confirm the custom pickle-based `save()`/`load()` wrappers in the repo still line up with v2's directory layout.
4. Smoke-train on a small sample (`sample_size=100`-style subset) and confirm: training runs, **validation loss is now actually evaluated each epoch** (visible in Lightning logs/MLflow — this is the #91 fix showing up), early stopping triggers correctly.

### Phase 3 — Retrain basic + hierarchical, apples-to-apples (1 day compute + review)
1. Retrain with **identical hyperparameters** to production (embedding_dim=128, n-grams 3–6, vocab 100k, same lr/epochs/patience) on the same training data.
2. Compare against baseline on the frozen split. Expect small-to-moderate gains purely from correct early stopping/checkpointing. Log everything to MLflow under an `ttc-v2-migration` experiment.

### Phase 4 — Rewrite the multi-head classifier (2–4 days, the main effort)
Two options; **Option A recommended**.

**Option A — port to `contrib.MultiLevelTextClassificationModel` (recommended).**
The hand-rolled architecture (shared NGram tokenizer → shared embedding + transformer `Block`s → per-level label-attention heads → per-level `ClassificationHead`) is exactly what v2's contrib module formalizes:
- Build one shared `TokenEmbedder` (configure attention layers via `TokenEmbedderConfig`/`AttentionConfig` to reproduce `n_attention_layers=2, n_heads=4`).
- Build one `SentenceEmbedder` per COICOP level with label attention (`n_heads_label_attention=4`, per-level class counts) and one `ClassificationHead` per level.
- Assemble with `MultiLevelTextClassificationModel(...)`, train via `MultiLevelCrossEntropyLoss` (supports the existing `loss_weights` idea), and wrap with `torchTextClassifiers.from_model(tokenizer, model)` to get `predict()/save()/load()` for free.
- Keep the repo-specific logic that must survive the port: per-level label filtering (`min_samples_per_level=50`, `min_samples_per_class=2`) and **parent–child masking at inference** (hierarchical consistency), which contrib does not provide — retain it as a thin post-processing layer over the per-level logits.
- Delete the bespoke Lightning module, dataset, and training loop (~600 of the 952 lines).

**Option B — mechanical rename (fallback if A hits blockers).**
Rename imports (`TextEmbedderConfig`→`TokenEmbedderConfig`+`SentenceEmbedderConfig`, `LabelAttentionClassifier`→`LabelAttention`/`SentenceEmbedder`) and adapt constructor calls. Faster but keeps re-implementing library internals — the class of bug (#92/#96) this upgrade exists to fix. Only use to unblock, then schedule A.

Acceptance for Phase 4: reproduce or beat the v1 multi-head model's per-level accuracy on the frozen split; verify label-attention explanations on a sample of short product labels no longer attribute mass to padding positions.

### Phase 5 (optional, separable) — `ValueEncoder` adoption
Replace the manual `label_to_idx`/`idx_to_label` bookkeeping (duplicated in all three classifiers) with a `ValueEncoder` built from the label sets, and drop `raw_labels=False`.
- **Benefit:** less code, labels persisted inside the model directory (`value_encoder.pkl`), decoding handled by the library.
- **Cost:** `predict()` return type changes from integer indices to raw label strings → touch `predict.py`, `evaluation/`, `api.py` (serve endpoint), and the parquet schema consumed by `decide-coicop`.
- **Recommendation:** ship Phases 1–4 first; do this as a follow-up PR with its own tests. Do not mix it into the migration diff.

### Phase 6 — Exploit the new capabilities (1–2 weeks, experimentation)
Now that parity is established, use v2's new levers, tracked in MLflow:
1. **`sample_weights`:** (a) inverse-sqrt class frequency to counter COICOP imbalance (many codes near the `min_samples_per_class=2` floor); (b) source-based weights — downweight LLM-synthetic and Circana-family-mapped labels relative to human-validated annotations. Run as a small grid.
2. **Label attention on the flat classifier:** `ModelConfig(n_heads_label_attention=4, aggregation_method=...)` — the INSEE internship benchmark found label attention competitive with mean pooling while adding per-class token explanations; on very short receipt texts this is now trustworthy thanks to the masking fixes.
3. **Calibration check:** recompute accuracy-vs-confidence buckets for the new TTC. If calibration improved, consider whether the 0.90 consensus threshold in `decide-coicop` can be lowered — every extra consensus case is one fewer LLM call.

### Phase 7 — Packaging and pipeline rollout (2–3 days)
1. Rebuild the classifier Docker image from the migrated source. **Important:** the Argo `run-ttc` step currently runs the legacy pre-built image `ghcr.io/micedre/coicop_bdf_classifier:latest`; switch `argo/pipeline.yaml` and `argo/ttc-pipeline.yaml` to an image built from this repo, pinned by digest, so the vendored source is actually what runs.
2. Publish the retrained model artifacts to S3/MLflow model registry with a `ttc-v2` version tag; keep v1 artifacts in place.
3. **Shadow run:** execute the full pipeline once with `-p sample_size=…` writing `run-ttc` output to a shadow path; diff v1-vs-v2 TTC predictions and the downstream `decide-coicop` outcomes (consensus rate, disagreement patterns, final accuracy via `skip-report=false`).
4. Cutover: point `run-ttc` at the new image/model. Roll back = repoint image tag + model path to the v1 artifacts frozen in Phase 0 (one-line Argo parameter change).

---

## 3. Acceptance gates

| Gate         | Criterion                                                                                                                                             |
|--------------|-------------------------------------------------------------------------------------------------------------------------------------------------------|
| G1 (Phase 2) | Test suite green under v2; save/load round-trip OK; validation metrics visibly logged during training                                                 |
| G2 (Phase 3) | Basic + hierarchical: per-level accuracy ≥ baseline on frozen split (levels 1–4; level 5 informational)                                               |
| G3 (Phase 4) | Multi-head: per-level accuracy ≥ v1 multi-head; attention attributions ignore padding on ≥ a manually reviewed sample of 50 short labels              |
| G4 (Phase 7) | Shadow pipeline: `decide-coicop` final accuracy ≥ baseline; consensus rate not degraded (target: improved); no schema breaks in `predictions.parquet` |

## 4. Risks and mitigations

| Risk                                                                                                                 | Mitigation                                                                                                                                                                                                                                                       |
|----------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Hidden v2 behavior changes beyond the documented diff (v2 is one day old)                                            | Pin exact version `==2.0.0`; smoke tests in Phase 2 before any retraining compute; report issues upstream (same org — short feedback loop)                                                                                                                       |
| Multi-head port drifts from original architecture semantics (e.g., RMSNorm placement, residual structure in `Block`) | Parameter-count and per-layer shape assertions against the v1 model; unit test comparing logits of a hand-assembled v2 model vs expected shapes                                                                                                                  |
| Return-type ripple if ValueEncoder sneaks in                                                                         | ValueEncoder isolated to Phase 5; Phases 1–4 keep integer labels (`raw_labels=False`)                                                                                                                                                                            |
| `run-ttc` is slated for eventual removal — sunk effort                                                               | Phases 1–3 are cheap (~3 days) and de-risk the library for other INSEE users of the same stack; Phase 4/6 effort is justified primarily by the `decide-coicop` cost savings — validate that hypothesis in the Phase 6 calibration check before investing further |
| Retraining cost                                                                                                      | Reuse existing Argo GPU workflow (`ttc-pipeline.yaml`); Phase 3 is a single retrain per architecture, grids only in Phase 6                                                                                                                                      |

## 5. Effort summary

| Phase                                   | Effort                            |
|-----------------------------------------|-----------------------------------|
| 0 Baseline freeze                       | 0.5 d                             |
| 1 Dependency bump                       | 0.5 d                             |
| 2 Minimal compat (basic + hierarchical) | 1–2 d                             |
| 3 Retrain + parity eval                 | ~1 d (mostly compute)             |
| 4 Multi-head port to contrib            | 2–4 d                             |
| 5 ValueEncoder (optional, follow-up)    | 1–2 d                             |
| 6 New-capability experiments            | 1–2 wk (parallelizable, optional) |
| 7 Packaging + shadow run + cutover      | 2–3 d                             |

**Critical path to production parity (Phases 0–4, 7): roughly 7–11 working days**, dominated by the multi-head rewrite. Phases 5–6 are value-add and can trail the cutover.
