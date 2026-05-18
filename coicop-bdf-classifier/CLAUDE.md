# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Hierarchical text classifier for COICOP (Classification of Individual Consumption According to Purpose) codes from French product descriptions. Built for INSEE (French National Institute of Statistics). Uses `torchtextclassifiers` as the core ML framework.

## Commands

All commands use `uv` as the package manager (with INSEE internal PyPI proxy).

```bash
# Install dependencies
uv sync

# Train hierarchical classifier (n-gram tokenizer, 5-level)
uv run python main.py train-hierarchical --data data/data-train.parquet --output checkpoints/hierarchical

# Train multi-head classifier (shared backbone, attention heads)
uv run python main.py train-multihead --data data/data-train.parquet --output checkpoints/multihead

# Train basic flat classifier
uv run python main.py train-basic --data data/data-train.parquet --output checkpoints/basic

# Fine-tune a pre-trained hierarchical model
uv run python main.py fine-tune-hierarchical --model checkpoints/hierarchical/hierarchical_model --data data/new.parquet --output checkpoints/fine-tuned

# Predict with hierarchical model (supports top-k)
uv run python main.py predict-hierarchical --model checkpoints/hierarchical/hierarchical_model "pain complet bio"

# Predict with multi-head model
uv run python main.py predict-multihead --model checkpoints/multihead/multihead_model "pain complet bio"

# Predict with basic model
uv run python main.py predict-basic --model checkpoints/basic/basic_model "pain complet bio"

# Generate evaluation report on annotated data
uv run python main.py evaluate-report --model checkpoints/basic/basic_model --data-dir data/annotated

# Start FastAPI server
uv run python main.py serve --model checkpoints/hierarchical/hierarchical_model
```

## Architecture

### Classification Approaches

**Hierarchical Classifier** (`src/hierarchical_classifier.py`): 5-level architecture using character n-gram tokenizer (3-6 grams). Each level N receives parent predictions from level N-1 as categorical features (parent code embedding + confidence bucket). Uses teacher forcing during training (default ratio 0.8).

**Multi-Head Classifier** (`src/multihead_classifier.py`): Shared backbone with per-level label-attention classification heads. Single model with transformer blocks shared across all COICOP levels.

**Basic Classifier** (`src/basic_classifier.py`): Single flat classifier using n-gram tokenizer. Predicts COICOP codes without hierarchical structure.

### Key Data Flow

1. **Data loading** (`src/data_preparation.py`): Reads parquet, applies text preprocessing (unidecode → lowercase → noise/punctuation/digit removal → dedup tokens → remove stopwords from `data/text/stopwords.json`), extracts 5 hierarchical levels from dotted COICOP codes (e.g., "01.1.2.3.4" → level1="01", level2="01.1", etc.), filters out technical codes 98.x/99.x.

2. **Training** (`src/train.py`): Orchestrates training for all classifier types, handles MLflow experiment tracking, runs optional post-training top-k evaluation.

3. **Prediction** (`src/predict.py`): `HierarchicalCOICOPPredictor`, `MultiHeadCOICOPPredictor`, and `BasicCOICOPPredictor`. Hierarchical and multi-head predictors share a common base class. All support single text, batch, DataFrame, and file-based prediction.

4. **API** (`src/api.py`): FastAPI server exposing `/predict`, `/predict/batch`, `/health`, `/model/info`. Model path set via `COICOP_MODEL_PATH` env var. Serves a frontend UI from `src/static/`.

### CLI Structure

`main.py` uses argparse with subcommands: `train-hierarchical`, `fine-tune-hierarchical`, `train-multihead`, `train-basic`, `predict-hierarchical`, `predict-multihead`, `predict-basic`, `evaluate-report`, `build-training-data`, `extract-ddc`, `serve`. Each subcommand has a `cmd_*` handler that imports the relevant module lazily.

### Project Structure

- `src/` — core package (classifiers, training, prediction, API, MLflow utils)
- `scripts/` — standalone utility scripts (accuracy analysis, data transforms, SQL)
- `data/` — training data, stopwords, COICOP definitions

## Key Conventions

- Python 3.13 required (`.python-version`)
- PyPI packages fetched through INSEE Nexus proxy (configured in `pyproject.toml` `[tool.uv]`)
- PyTorch is installed from a CPU-only index (`pytorch-cpu`)
- Langchain dependencies are optional: install with `uv sync --extra synth`
- Training data format: parquet with columns `product` (text), `code` (COICOP code), `coicop` (description)
- COICOP codes are dot-separated hierarchical strings (e.g., "01.1.2.3.4")
- Text preprocessing always applies `preprocess_text()` before training/prediction
- Models are serialized with pickle (metadata/mappings) + torchtextclassifiers built-in save/load (weights)
- Some code comments and documentation are in French
