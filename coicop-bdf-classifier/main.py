"""CLI entry point for COICOP BDF Classifier."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def cmd_train_hierarchical(args: argparse.Namespace) -> None:
    """Train the hierarchical multi-level classifier."""
    from src.train import train_hierarchical_classifier

    train_hierarchical_classifier(
        annotations_path=args.data,
        output_dir=args.output,
        ngram_min_n=args.ngram_min,
        ngram_max_n=args.ngram_max,
        ngram_num_tokens=args.ngram_vocab_size,
        embedding_dim=args.embedding_dim,
        parent_embedding_dim=args.parent_embedding_dim,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
        lr=args.lr,
        num_epochs=args.num_epochs,
        patience=args.patience,
        min_samples=args.min_samples,
        use_parent_features=args.use_parent_features,
        teacher_forcing_ratio=args.teacher_forcing_ratio,
        mlflow_experiment=args.mlflow_experiment,
        eval_data_path=args.eval_data,
        eval_top_k=args.eval_top_k,
        eval_text_column=args.eval_text_column,
        eval_filter_columns=args.eval_filter_columns,
        eval_code_column=args.eval_code_column,
        eval_beam_size=args.eval_beam_size,
        preprocess=args.preprocess,
        code_column=args.code_column,
        resume_from=args.resume,
        encryption_key=args.encryption_key,
        max_level=args.max_level,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        tokenizer_name=args.tokenizer,
    )


def cmd_fine_tune_hierarchical(args: argparse.Namespace) -> None:
    """Fine-tune a pre-trained hierarchical classifier on new data."""
    from src.train import fine_tune_hierarchical_classifier

    levels = None
    if args.levels:
        levels = [l.strip() for l in args.levels.split(",")]

    fine_tune_hierarchical_classifier(
        model_path=args.model,
        annotations_path=args.data,
        output_dir=args.output,
        levels=levels,
        lr=args.lr,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        teacher_forcing_ratio=args.teacher_forcing_ratio,
        mlflow_experiment=args.mlflow_experiment,
        eval_data_path=args.eval_data,
        eval_top_k=args.eval_top_k,
        eval_text_column=args.eval_text_column,
        eval_filter_columns=args.eval_filter_columns,
        eval_code_column=args.eval_code_column,
        eval_beam_size=args.eval_beam_size,
        preprocess=args.preprocess,
        code_column=args.code_column,
        encryption_key=args.encryption_key,
        max_level=args.max_level,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )


def cmd_train_basic(args: argparse.Namespace) -> None:
    """Train the basic flat classifier."""
    from src.train import train_basic_classifier

    train_basic_classifier(
        data_path=args.data,
        code_column=args.code_column,
        text_column=args.text_column,
        output_dir=args.output,
        ngram_min_n=args.ngram_min,
        ngram_max_n=args.ngram_max,
        ngram_num_tokens=args.ngram_vocab_size,
        embedding_dim=args.embedding_dim,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
        lr=args.lr,
        num_epochs=args.num_epochs,
        patience=args.patience,
        mlflow_experiment=args.mlflow_experiment,
        eval_data_path=args.eval_data,
        eval_top_k=args.eval_top_k,
        eval_text_column=args.eval_text_column,
        eval_filter_columns=args.eval_filter_columns,
        eval_code_column=args.eval_code_column,
        preprocess=args.preprocess,
        encryption_key=args.encryption_key,
        tokenizer_name=args.tokenizer,
    )


def cmd_fine_tune_basic(args: argparse.Namespace) -> None:
    """Fine-tune a pre-trained basic classifier on new data."""
    from src.train import fine_tune_basic_classifier

    fine_tune_basic_classifier(
        model_path=args.model,
        data_path=args.data,
        output_dir=args.output,
        code_column=args.code_column,
        text_column=args.text_column,
        lr=args.lr,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        mlflow_experiment=args.mlflow_experiment,
        eval_data_path=args.eval_data,
        eval_top_k=args.eval_top_k,
        eval_text_column=args.eval_text_column,
        eval_filter_columns=args.eval_filter_columns,
        eval_code_column=args.eval_code_column,
        preprocess=args.preprocess,
        encryption_key=args.encryption_key,
    )


def cmd_predict_basic(args: argparse.Namespace) -> None:
    """Predict COICOP codes using basic flat classifier."""
    from src.predict import BasicCOICOPPredictor

    predictor = BasicCOICOPPredictor(args.model)

    if args.file:
        predictor.predict_file(
            input_path=args.file,
            output_path=args.output,
            text_column=args.text_column,
            batch_size=args.batch_size,
            top_k=args.top_k,
        )
    else:
        predictions = predictor.predict(args.texts, top_k=args.top_k)
        for pred in predictions:
            print(f"\nText: {pred['text']}")
            print(f"Code: {pred['code']} (confidence: {pred['confidence']:.2f})")
            if "alternatives" in pred:
                for i, alt in enumerate(pred["alternatives"], start=2):
                    print(f"  top {i}: {alt['code']} (conf: {alt['confidence']:.2f})")


def cmd_predict_hierarchical(args: argparse.Namespace) -> None:
    """Predict COICOP codes using hierarchical classifier."""
    from src.predict import HierarchicalCOICOPPredictor

    predictor = HierarchicalCOICOPPredictor(args.model)

    if args.file:
        # File-based prediction
        predictor.predict_file(
            input_path=args.file,
            output_path=args.output,
            text_column=args.text_column,
            batch_size=args.batch_size,
            top_k=args.top_k,
            confidence_threshold=args.confidence_threshold,
            beam_size=args.beam_size,
        )
    else:
        # Text-based prediction
        if args.input:
            texts = [args.input]
        else:
            texts = args.texts

        predictions = predictor.predict(
            texts,
            top_k=args.top_k,
            confidence_threshold=args.confidence_threshold,
            beam_size=args.beam_size,
        )
        for pred in predictions:
            # Format hierarchical output nicely
            print(f"\nText: {pred['text']}")
            print(f"Final code: {pred['code']} (confidence: {pred['confidence']:.2f})")
            print(f"Combined confidence: {pred['combined_confidence']:.4f}")
            if "levels" in pred:
                print("Level breakdown:")
                for level_name, level_data in pred["levels"].items():
                    print(
                        f"  {level_name}: {level_data['code']} (conf: {level_data['confidence']:.2f})"
                    )
                    if "alternatives" in level_data:
                        for i, alt in enumerate(level_data["alternatives"], start=2):
                            print(
                                f"    top {i}: {alt['code']} (conf: {alt['confidence']:.2f})"
                            )


def cmd_train_multihead(args: argparse.Namespace) -> None:
    """Train the multi-head classifier with shared backbone."""
    from src.train import train_multihead_classifier

    loss_weights = None
    if args.loss_weights:
        loss_weights = [float(w) for w in args.loss_weights.split(",")]

    train_multihead_classifier(
        annotations_path=args.data,
        output_dir=args.output,
        ngram_min_n=args.ngram_min,
        ngram_max_n=args.ngram_max,
        ngram_num_tokens=args.ngram_vocab_size,
        embedding_dim=args.embedding_dim,
        max_seq_length=args.max_seq_length,
        n_attention_layers=args.n_attention_layers,
        n_attention_heads=args.n_attention_heads,
        n_kv_heads=args.n_kv_heads,
        n_label_attention_heads=args.n_label_attention_heads,
        batch_size=args.batch_size,
        lr=args.lr,
        num_epochs=args.num_epochs,
        patience=args.patience,
        min_samples=args.min_samples,
        max_level=args.max_level,
        loss_weights=loss_weights,
        mlflow_experiment=args.mlflow_experiment,
        eval_data_path=args.eval_data,
        eval_top_k=args.eval_top_k,
        eval_text_column=args.eval_text_column,
        eval_filter_columns=args.eval_filter_columns,
        eval_code_column=args.eval_code_column,
        preprocess=args.preprocess,
        code_column=args.code_column,
        encryption_key=args.encryption_key,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        tokenizer_name=args.tokenizer,
    )


def cmd_predict_multihead(args: argparse.Namespace) -> None:
    """Predict COICOP codes using multi-head classifier."""
    from src.predict import MultiHeadCOICOPPredictor

    predictor = MultiHeadCOICOPPredictor(args.model)

    if args.file:
        predictor.predict_file(
            input_path=args.file,
            output_path=args.output,
            text_column=args.text_column,
            batch_size=args.batch_size,
            top_k=args.top_k,
            confidence_threshold=args.confidence_threshold,
        )
    else:
        if args.input:
            texts = [args.input]
        else:
            texts = args.texts

        predictions = predictor.predict(
            texts, top_k=args.top_k, confidence_threshold=args.confidence_threshold
        )
        for pred in predictions:
            print(f"\nText: {pred['text']}")
            print(f"Final code: {pred['code']} (confidence: {pred['confidence']:.2f})")
            print(f"Combined confidence: {pred['combined_confidence']:.4f}")
            if "levels" in pred:
                print("Level breakdown:")
                for level_name, level_data in pred["levels"].items():
                    print(
                        f"  {level_name}: {level_data['code']} (conf: {level_data['confidence']:.2f})"
                    )
                    if "alternatives" in level_data:
                        for i, alt in enumerate(level_data["alternatives"], start=2):
                            print(
                                f"    top {i}: {alt['code']} (conf: {alt['confidence']:.2f})"
                            )


def cmd_classify_llm(args: argparse.Namespace) -> None:
    """Classify a file into COICOP codes using an LLM."""
    import asyncio

    from src.classifiers.llm_classifier import classify_llm

    context_columns = {}
    if args.price_column:
        context_columns["prix"] = args.price_column
    if args.shop_column:
        context_columns["enseigne"] = args.shop_column
    if args.shop_category_column:
        context_columns["categorie"] = args.shop_category_column

    asyncio.run(
        classify_llm(
            input_path=args.input,
            coicop_path=args.coicop,
            output_path=args.output,
            text_column=args.text_column,
            batch_size=args.batch_size,
            concurrency=args.concurrency,
            context_columns=context_columns or None,
        )
    )


def cmd_decide_coicop(args: argparse.Namespace) -> None:
    """Decide the final COICOP code using an LLM as judge over multiple model predictions."""
    import asyncio
    import json
    import sys

    from openai import OpenAI

    from src.decide_coicop import (
        build_prompt,
        call_llm_sync,
        get_observation,
        load_all_observations,
        load_nomenclature,
        print_result,
        run_batch,
        try_consensus_decision,
    )

    if not os.environ.get("LLMLAB_API_KEY"):
        logger.error("Variable LLMLAB_API_KEY non définie.")
        sys.exit(1)

    logger.info("Chargement des données...")
    df = load_all_observations(args.lcs_file, args.rag_file, args.ttc_file)

    logger.info("Chargement de la nomenclature...")
    nomenclature = load_nomenclature(args.nomenclature)

    if args.id is not None:
        try:
            obs = get_observation(df, args.id)
        except KeyError as exc:
            logger.error("%s", exc)
            sys.exit(1)

        api_kwargs: dict = {"api_key": os.environ["LLMLAB_API_KEY"]}
        if base_url := os.environ.get("LLMLAB_URL"):
            api_kwargs["base_url"] = base_url
        client = OpenAI(**api_kwargs)

        decision = try_consensus_decision(obs, nomenclature)
        if decision is not None:
            logger.info(
                "Consensus détecté pour id=%s — LLM ignoré (code=%s)",
                args.id,
                decision.coicop_code,
            )
        else:
            logger.info(
                "Appel au modèle %s pour id=%s (nomen=%s)...",
                args.model,
                args.id,
                "complète" if args.full_nomenclature else "filtrée",
            )
            prompt = build_prompt(obs, nomenclature, full_nomen=args.full_nomenclature)
            decision = call_llm_sync(prompt, args.model, client)

        if args.output == "json":
            print(
                json.dumps(
                    {
                        "id": args.id,
                        "raw_product": obs.get("raw_product"),
                        "shop": obs.get("shop"),
                        "shop_type_name": obs.get("shop_type_name"),
                        "budget": obs.get("budget"),
                        "code_reference": obs.get("code"),
                        "coicop_code": decision.coicop_code,
                        "libelle": decision.libelle,
                        "explication": decision.explication,
                        "confiance": decision.confiance,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        else:
            print_result(obs, decision)
        return

    asyncio.run(
        run_batch(
            df=df,
            nomenclature=nomenclature,
            model=args.model,
            concurrency=args.concurrency,
            output_file=args.output_file,
            full_nomen=args.full_nomenclature,
        )
    )


def cmd_serve(args: argparse.Namespace) -> None:
    """Start the FastAPI prediction server."""
    import uvicorn

    os.environ["COICOP_MODEL_PATH"] = args.model
    uvicorn.run(
        "src.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


def cmd_extract_ddc(args: argparse.Namespace) -> None:
    """Extract DDC data from S3."""
    from src.data.extract_ddc import extract_ddc

    extract_ddc(
        annee=args.annee,
        mois=args.mois,
        output_s3_path=args.output,
        famille_circana_path=args.famille,
        memory_limit=args.memory,
        dry_run=args.dry_run,
        encrypt=args.encrypt,
        encryption_key=args.encryption_key,
    )


def cmd_build_training_data(args: argparse.Namespace) -> None:
    """Build a balanced training dataset from DDC and synthetic data."""
    from src.data.build_training_data import build_training_data

    build_training_data(
        ddc_path=args.ddc,
        ddc_text_column=args.ddc_text_column,
        ddc_code_column=args.ddc_code_column,
        output_path=args.output,
        synthetic_path=args.synthetic,
        max_per_code=args.max_per_code,
        seed=args.seed,
        encryption_key=args.encryption_key,
        preprocess=args.preprocess,
    )


def cmd_evaluate_report(args: argparse.Namespace) -> None:
    """Generate a comprehensive evaluation report on annotated data."""
    from src.evaluation.evaluation_report import (
        run_evaluation,
        format_report,
        log_metrics_to_mlflow,
    )

    metrics = run_evaluation(
        model_path=args.model,
        data_dir=args.data_dir,
        top_k=args.top_k,
        text_column=args.text_column,
        amount_threshold=args.amount_threshold,
    )
    report = format_report(metrics)
    print(report)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(report, encoding="utf-8")
        logger.info(f"Report saved to {args.output}")

    if args.mlflow_run_id or args.mlflow_experiment:
        log_metrics_to_mlflow(
            metrics,
            run_id=args.mlflow_run_id,
            experiment_name=args.mlflow_experiment,
        )


def cmd_evaluate_predictions(args: argparse.Namespace) -> None:
    """Evaluate a prediction file by COICOP level."""
    from src.evaluation.evaluate_predictions import run_evaluate_predictions

    _, report = run_evaluate_predictions(
        prediction_path=args.predictions,
        code_column=args.code_column,
        text_column=args.text_column,
        categorical_column=args.category_column,
        max_k=args.max_k,
    )
    print(report)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(report, encoding="utf-8")
        logger.info(f"Report saved to {args.output}")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="COICOP BDF Classifier - Hierarchical text classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train-hierarchical command
    train_hier_parser = subparsers.add_parser(
        "train-hierarchical",
        help="Train the hierarchical multi-level classifier with n-gram tokenization",
    )
    train_hier_parser.add_argument(
        "--data",
        type=str,
        default="data/data-train.parquet",
        help="Path to training data (parquet or csv)",
    )
    train_hier_parser.add_argument(
        "--output",
        type=str,
        default="checkpoints/hierarchical",
        help="Output directory for trained models",
    )
    train_hier_parser.add_argument(
        "--ngram-min",
        type=int,
        default=3,
        help="Minimum n-gram size for tokenizer",
    )
    train_hier_parser.add_argument(
        "--ngram-max",
        type=int,
        default=6,
        help="Maximum n-gram size for tokenizer",
    )
    train_hier_parser.add_argument(
        "--ngram-vocab-size",
        type=int,
        default=100000,
        help="Vocabulary size for n-gram tokenizer",
    )
    train_hier_parser.add_argument(
        "--embedding-dim",
        type=int,
        default=128,
        help="Text embedding dimension",
    )
    train_hier_parser.add_argument(
        "--parent-embedding-dim",
        type=int,
        default=32,
        help="Embedding dimension for parent code categorical feature (default: 32)",
    )
    train_hier_parser.add_argument(
        "--max-seq-length",
        type=int,
        default=64,
        help="Maximum sequence length",
    )
    train_hier_parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size",
    )
    train_hier_parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="Learning rate",
    )
    train_hier_parser.add_argument(
        "--num-epochs",
        type=int,
        default=20,
        help="Maximum number of epochs",
    )
    train_hier_parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early stopping patience",
    )
    train_hier_parser.add_argument(
        "--min-samples",
        type=int,
        default=50,
        help="Minimum samples per level",
    )
    train_hier_parser.add_argument(
        "--use-parent-features",
        action="store_true",
        default=True,
        help="Use parent predictions as categorical features (default: True)",
    )
    train_hier_parser.add_argument(
        "--no-parent-features",
        action="store_false",
        dest="use_parent_features",
        help="Disable parent prediction features",
    )
    train_hier_parser.add_argument(
        "--teacher-forcing-ratio",
        type=float,
        default=0.8,
        help="Ratio of ground truth to use during training (0.0-1.0)",
    )
    train_hier_parser.add_argument(
        "--mlflow-experiment",
        type=str,
        default=None,
        help="MLflow experiment name (optional)",
    )
    train_hier_parser.add_argument(
        "--eval-data",
        type=str,
        default=None,
        help="Path to evaluation parquet for post-training top-k accuracy",
    )
    train_hier_parser.add_argument(
        "--eval-top-k",
        type=int,
        default=5,
        help="Maximum K for top-k accuracy evaluation (default: 5)",
    )
    train_hier_parser.add_argument(
        "--eval-beam-size",
        type=int,
        default=1,
        help="Beam size for hierarchical cascade during post-training evaluation (default: 1 = greedy)",
    )
    train_hier_parser.add_argument(
        "--eval-text-column",
        type=str,
        default="text",
        help="Text column name in evaluation data (default: text)",
    )
    train_hier_parser.add_argument(
        "--eval-filter-columns",
        type=str,
        nargs="+",
        metavar="COL",
        default=None,
        help="Boolean columns in eval data to compute separate metrics for (True/False subsets)",
    )
    train_hier_parser.add_argument(
        "--eval-code-column",
        type=str,
        default="code",
        help="Name of the true code column in eval data (default: code)",
    )
    train_hier_parser.add_argument(
        "--code-column",
        type=str,
        default="code",
        help="Name of the COICOP code column in training data (default: code)",
    )
    train_hier_parser.add_argument(
        "--preprocess",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply text preprocessing to training data (default: True)",
    )
    train_hier_parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume training from a previous checkpoint (uses output directory)",
    )
    train_hier_parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="HuggingFace tokenizer name (e.g. 'camembert-base'). Default: n-gram tokenizer",
    )
    train_hier_parser.add_argument(
        "--encryption-key",
        type=str,
        default=None,
        help="Parquet encryption key (hex, 32 chars) for reading/writing encrypted files",
    )
    train_hier_parser.add_argument(
        "--max-level",
        type=int,
        default=5,
        choices=range(1, 6),
        metavar="{1,2,3,4,5}",
        help="Maximum COICOP hierarchy depth to train (1-5, default: 5)",
    )
    train_hier_parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers (0 = main process only, safest on Windows; increase on Linux)",
    )
    train_hier_parser.add_argument(
        "--pin-memory",
        action="store_true",
        default=True,
        help="Pin memory for faster CPU→GPU transfer (default: True)",
    )
    train_hier_parser.add_argument(
        "--no-pin-memory",
        action="store_false",
        dest="pin_memory",
        help="Disable pinned memory",
    )
    train_hier_parser.set_defaults(func=cmd_train_hierarchical)

    # Fine-tune-hierarchical command
    ft_hier_parser = subparsers.add_parser(
        "fine-tune-hierarchical",
        help="Fine-tune a pre-trained hierarchical classifier on new data",
    )
    ft_hier_parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the pre-trained hierarchical model",
    )
    ft_hier_parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to new training data (parquet or csv)",
    )
    ft_hier_parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for the fine-tuned model",
    )
    ft_hier_parser.add_argument(
        "--levels",
        type=str,
        default=None,
        help="Comma-separated list of levels to fine-tune (e.g. level3,level4). Default: all",
    )
    ft_hier_parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (default: original lr / 10)",
    )
    ft_hier_parser.add_argument(
        "--num-epochs",
        type=int,
        default=None,
        help="Number of epochs (default: 5)",
    )
    ft_hier_parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Training batch size (default: same as original)",
    )
    ft_hier_parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Early stopping patience (default: 3)",
    )
    ft_hier_parser.add_argument(
        "--teacher-forcing-ratio",
        type=float,
        default=None,
        help="Teacher forcing ratio (default: same as original)",
    )
    ft_hier_parser.add_argument(
        "--mlflow-experiment",
        type=str,
        default=None,
        help="MLflow experiment name (optional)",
    )
    ft_hier_parser.add_argument(
        "--eval-data",
        type=str,
        default=None,
        help="Path to evaluation data for post-training top-k accuracy",
    )
    ft_hier_parser.add_argument(
        "--eval-top-k",
        type=int,
        default=5,
        help="Maximum K for top-k accuracy evaluation (default: 5)",
    )
    ft_hier_parser.add_argument(
        "--eval-beam-size",
        type=int,
        default=1,
        help="Beam size for hierarchical cascade during post-training evaluation (default: 1 = greedy)",
    )
    ft_hier_parser.add_argument(
        "--eval-text-column",
        type=str,
        default="text",
        help="Text column name in evaluation data (default: text)",
    )
    ft_hier_parser.add_argument(
        "--eval-filter-columns",
        type=str,
        nargs="+",
        metavar="COL",
        default=None,
        help="Boolean columns in eval data to compute separate metrics for (True/False subsets)",
    )
    ft_hier_parser.add_argument(
        "--eval-code-column",
        type=str,
        default="code",
        help="Name of the true code column in eval data (default: code)",
    )
    ft_hier_parser.add_argument(
        "--code-column",
        type=str,
        default="code",
        help="Name of the COICOP code column in training data (default: code)",
    )
    ft_hier_parser.add_argument(
        "--preprocess",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply text preprocessing to training data (default: True)",
    )
    ft_hier_parser.add_argument(
        "--encryption-key",
        type=str,
        default=None,
        help="Parquet encryption key (hex, 32 chars) for reading/writing encrypted files",
    )
    ft_hier_parser.add_argument(
        "--max-level",
        type=int,
        default=None,
        choices=range(1, 6),
        metavar="{1,2,3,4,5}",
        help="Maximum COICOP hierarchy depth (1-5, default: use model's setting)",
    )
    ft_hier_parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="DataLoader workers (default: use model's setting)",
    )
    ft_hier_parser.add_argument(
        "--pin-memory",
        action="store_true",
        default=None,
        help="Pin memory for faster CPU→GPU transfer",
    )
    ft_hier_parser.add_argument(
        "--no-pin-memory",
        action="store_false",
        dest="pin_memory",
        help="Disable pinned memory",
    )
    ft_hier_parser.set_defaults(func=cmd_fine_tune_hierarchical)

    # Train-basic command
    train_basic_parser = subparsers.add_parser(
        "train-basic",
        help="Train the basic flat classifier with n-gram tokenization",
    )
    train_basic_parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to training parquet (from build-training-data)",
    )
    train_basic_parser.add_argument(
        "--code-column",
        type=str,
        required=False,
        default="code8",
        help="Code column in data",
    )
    train_basic_parser.add_argument(
        "--text-column",
        type=str,
        required=False,
        default="product",
        help="Text column in data",
    )
    train_basic_parser.add_argument(
        "--output",
        type=str,
        default="checkpoints/basic",
        help="Output directory for trained model",
    )
    train_basic_parser.add_argument(
        "--ngram-min",
        type=int,
        default=3,
        help="Minimum n-gram size for tokenizer",
    )
    train_basic_parser.add_argument(
        "--ngram-max",
        type=int,
        default=6,
        help="Maximum n-gram size for tokenizer",
    )
    train_basic_parser.add_argument(
        "--ngram-vocab-size",
        type=int,
        default=100000,
        help="N-gram vocabulary size",
    )
    train_basic_parser.add_argument(
        "--embedding-dim",
        type=int,
        default=128,
        help="Embedding dimension",
    )
    train_basic_parser.add_argument(
        "--max-seq-length",
        type=int,
        default=64,
        help="Maximum sequence length",
    )
    train_basic_parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size",
    )
    train_basic_parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
        help="Learning rate",
    )
    train_basic_parser.add_argument(
        "--num-epochs",
        type=int,
        default=20,
        help="Maximum number of epochs",
    )
    train_basic_parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early stopping patience",
    )
    train_basic_parser.add_argument(
        "--mlflow-experiment",
        type=str,
        default=None,
        help="MLflow experiment name (enables MLflow + pyfunc logging)",
    )
    train_basic_parser.add_argument(
        "--eval-data",
        type=str,
        default=None,
        help="Path to evaluation parquet for post-training top-k accuracy",
    )
    train_basic_parser.add_argument(
        "--eval-top-k",
        type=int,
        default=5,
        help="Maximum K for top-k accuracy evaluation (default: 5)",
    )
    train_basic_parser.add_argument(
        "--eval-text-column",
        type=str,
        default="product",
        help="Text column name in evaluation data (default: product)",
    )
    train_basic_parser.add_argument(
        "--eval-filter-columns",
        type=str,
        nargs="+",
        metavar="COL",
        default=None,
        help="Boolean columns in eval data to compute separate metrics for (True/False subsets)",
    )
    train_basic_parser.add_argument(
        "--eval-code-column",
        type=str,
        default="code",
        help="Name of the true code column in eval data (default: code)",
    )
    train_basic_parser.add_argument(
        "--preprocess",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply text preprocessing to training data (default: False)",
    )
    train_basic_parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="HuggingFace tokenizer name (e.g. 'camembert-base'). Default: n-gram tokenizer",
    )
    train_basic_parser.add_argument(
        "--encryption-key",
        type=str,
        default=None,
        help="Parquet encryption key (hex, 32 chars) for reading/writing encrypted files",
    )
    train_basic_parser.set_defaults(func=cmd_train_basic)

    # Fine-tune-basic command
    ft_basic_parser = subparsers.add_parser(
        "fine-tune-basic",
        help="Fine-tune a pre-trained basic classifier on new data",
    )
    ft_basic_parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the pre-trained basic model",
    )
    ft_basic_parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to new training data (parquet)",
    )
    ft_basic_parser.add_argument(
        "--code-column",
        type=str,
        required=False,
        default="code8",
        help="Code column in data",
    )
    ft_basic_parser.add_argument(
        "--text-column",
        type=str,
        required=False,
        default="product",
        help="Text column in data",
    )
    ft_basic_parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for the fine-tuned model",
    )
    ft_basic_parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (default: original lr / 10)",
    )
    ft_basic_parser.add_argument(
        "--num-epochs",
        type=int,
        default=None,
        help="Number of epochs (default: 5)",
    )
    ft_basic_parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Training batch size (default: same as original)",
    )
    ft_basic_parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Early stopping patience (default: 3)",
    )
    ft_basic_parser.add_argument(
        "--mlflow-experiment",
        type=str,
        default=None,
        help="MLflow experiment name (optional)",
    )
    ft_basic_parser.add_argument(
        "--eval-data",
        type=str,
        default=None,
        help="Path to evaluation data for post-training top-k accuracy",
    )
    ft_basic_parser.add_argument(
        "--eval-top-k",
        type=int,
        default=5,
        help="Maximum K for top-k accuracy evaluation (default: 5)",
    )
    ft_basic_parser.add_argument(
        "--eval-text-column",
        type=str,
        default="product",
        help="Text column name in evaluation data (default: product)",
    )
    ft_basic_parser.add_argument(
        "--eval-filter-columns",
        type=str,
        nargs="+",
        metavar="COL",
        default=None,
        help="Boolean columns in eval data to compute separate metrics for (True/False subsets)",
    )
    ft_basic_parser.add_argument(
        "--eval-code-column",
        type=str,
        default="code",
        help="Name of the true code column in eval data (default: code)",
    )
    ft_basic_parser.add_argument(
        "--preprocess",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply text preprocessing to training data (default: False)",
    )
    ft_basic_parser.add_argument(
        "--encryption-key",
        type=str,
        default=None,
        help="Parquet encryption key (hex, 32 chars) for reading encrypted files",
    )
    ft_basic_parser.set_defaults(func=cmd_fine_tune_basic)

    # Predict-basic command
    predict_basic_parser = subparsers.add_parser(
        "predict-basic",
        help="Predict COICOP codes using basic flat classifier",
    )
    predict_basic_parser.add_argument(
        "--model",
        type=str,
        default="checkpoints/basic/basic_model",
        help=(
            "Path to saved basic model, or MLflow URI "
            "(runs:/<run_id>/…, models:/<name>/<version>, mlflow-artifacts:/…)"
        ),
    )
    predict_basic_parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Input file for batch prediction (local path or s3:// URI, CSV or parquet)",
    )
    predict_basic_parser.add_argument(
        "--output",
        type=str,
        default="predictions_basic.csv",
        help="Output file for batch prediction (local path or s3:// URI, CSV or parquet)",
    )
    predict_basic_parser.add_argument(
        "--text-column",
        type=str,
        default="product",
        help="Name of text column in input file",
    )
    predict_basic_parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for prediction",
    )
    predict_basic_parser.add_argument(
        "--top-k",
        type=int,
        default=1,
        help="Number of top predictions (default: 1)",
    )
    predict_basic_parser.add_argument(
        "texts",
        nargs="*",
        help="Text(s) to classify (if not using --file)",
    )
    predict_basic_parser.set_defaults(func=cmd_predict_basic)

    # Predict-hierarchical command
    predict_hier_parser = subparsers.add_parser(
        "predict-hierarchical",
        help="Predict COICOP codes using hierarchical classifier",
    )
    predict_hier_parser.add_argument(
        "--model",
        type=str,
        default="checkpoints/hierarchical/hierarchical_model",
        help="Path to saved hierarchical model or MLflow URI (runs:/..., models:/..., mlflow-artifacts:/...)",
    )
    predict_hier_parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Single text to classify",
    )
    predict_hier_parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Input file for batch prediction",
    )
    predict_hier_parser.add_argument(
        "--output",
        type=str,
        default="predictions_hierarchical.csv",
        help="Output file for batch prediction",
    )
    predict_hier_parser.add_argument(
        "--text-column",
        type=str,
        default="product",
        help="Name of text column in input file",
    )
    predict_hier_parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for prediction",
    )
    predict_hier_parser.add_argument(
        "--top-k",
        type=int,
        default=1,
        help="Number of top predictions per level (default: 1)",
    )
    predict_hier_parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=None,
        help="Minimum confidence per level; stop at the deepest level meeting this threshold",
    )
    predict_hier_parser.add_argument(
        "--beam-size",
        type=int,
        default=1,
        help="Beam size for hierarchical cascade (default: 1 = greedy). Higher values improve accuracy at the cost of speed.",
    )
    predict_hier_parser.add_argument(
        "texts",
        nargs="*",
        help="Text(s) to classify (if not using --file or --input)",
    )
    predict_hier_parser.set_defaults(func=cmd_predict_hierarchical)

    # Train-multihead command
    train_mh_parser = subparsers.add_parser(
        "train-multihead",
        help="Train the multi-head classifier with shared backbone",
    )
    train_mh_parser.add_argument(
        "--data",
        type=str,
        default="data/data-train.parquet",
        help="Path to training data (parquet or csv)",
    )
    train_mh_parser.add_argument(
        "--output",
        type=str,
        default="checkpoints/multihead",
        help="Output directory for trained models",
    )
    train_mh_parser.add_argument(
        "--ngram-min",
        type=int,
        default=3,
        help="Minimum n-gram size for tokenizer",
    )
    train_mh_parser.add_argument(
        "--ngram-max",
        type=int,
        default=6,
        help="Maximum n-gram size for tokenizer",
    )
    train_mh_parser.add_argument(
        "--ngram-vocab-size",
        type=int,
        default=100000,
        help="Vocabulary size for n-gram tokenizer",
    )
    train_mh_parser.add_argument(
        "--embedding-dim",
        type=int,
        default=128,
        help="Text embedding dimension",
    )
    train_mh_parser.add_argument(
        "--max-seq-length",
        type=int,
        default=64,
        help="Maximum sequence length",
    )
    train_mh_parser.add_argument(
        "--n-attention-layers",
        type=int,
        default=2,
        help="Number of transformer blocks in shared backbone",
    )
    train_mh_parser.add_argument(
        "--n-attention-heads",
        type=int,
        default=4,
        help="Attention heads per self-attention layer",
    )
    train_mh_parser.add_argument(
        "--n-kv-heads",
        type=int,
        default=4,
        help="KV heads (GQA if < n-attention-heads)",
    )
    train_mh_parser.add_argument(
        "--n-label-attention-heads",
        type=int,
        default=4,
        help="Attention heads in each LabelAttentionClassifier head",
    )
    train_mh_parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size",
    )
    train_mh_parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="Learning rate",
    )
    train_mh_parser.add_argument(
        "--num-epochs",
        type=int,
        default=20,
        help="Maximum number of epochs",
    )
    train_mh_parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early stopping patience",
    )
    train_mh_parser.add_argument(
        "--min-samples",
        type=int,
        default=50,
        help="Minimum samples per level",
    )
    train_mh_parser.add_argument(
        "--max-level",
        type=int,
        default=4,
        choices=range(1, 6),
        metavar="{1,2,3,4,5}",
        help="Maximum COICOP hierarchy depth to train (1-5, default: 4)",
    )
    train_mh_parser.add_argument(
        "--loss-weights",
        type=str,
        default=None,
        help="Comma-separated per-level loss weights (e.g. 1.0,1.0,1.0,1.0)",
    )
    train_mh_parser.add_argument(
        "--mlflow-experiment",
        type=str,
        default=None,
        help="MLflow experiment name (optional)",
    )
    train_mh_parser.add_argument(
        "--eval-data",
        type=str,
        default=None,
        help="Path to evaluation parquet for post-training top-k accuracy",
    )
    train_mh_parser.add_argument(
        "--eval-top-k",
        type=int,
        default=5,
        help="Maximum K for top-k accuracy evaluation (default: 5)",
    )
    train_mh_parser.add_argument(
        "--eval-text-column",
        type=str,
        default="text",
        help="Text column name in evaluation data (default: text)",
    )
    train_mh_parser.add_argument(
        "--eval-filter-columns",
        type=str,
        nargs="+",
        metavar="COL",
        default=None,
        help="Boolean columns in eval data to compute separate metrics for (True/False subsets)",
    )
    train_mh_parser.add_argument(
        "--eval-code-column",
        type=str,
        default="code",
        help="Name of the true code column in eval data (default: code)",
    )
    train_mh_parser.add_argument(
        "--code-column",
        type=str,
        default="code",
        help="Name of the COICOP code column in training data (default: code)",
    )
    train_mh_parser.add_argument(
        "--preprocess",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply text preprocessing to training data (default: True)",
    )
    train_mh_parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="HuggingFace tokenizer name (e.g. 'camembert-base'). Default: n-gram tokenizer",
    )
    train_mh_parser.add_argument(
        "--encryption-key",
        type=str,
        default=None,
        help="Parquet encryption key (hex, 32 chars) for reading encrypted files",
    )
    train_mh_parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers (0 = main process only)",
    )
    train_mh_parser.add_argument(
        "--pin-memory",
        action="store_true",
        default=True,
        help="Pin memory for faster CPU->GPU transfer (default: True)",
    )
    train_mh_parser.add_argument(
        "--no-pin-memory",
        action="store_false",
        dest="pin_memory",
        help="Disable pinned memory",
    )
    train_mh_parser.set_defaults(func=cmd_train_multihead)

    # Predict-multihead command
    predict_mh_parser = subparsers.add_parser(
        "predict-multihead",
        help="Predict COICOP codes using multi-head classifier",
    )
    predict_mh_parser.add_argument(
        "--model",
        type=str,
        default="checkpoints/multihead/multihead_model",
        help="Path to saved multi-head model or MLflow URI",
    )
    predict_mh_parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Single text to classify",
    )
    predict_mh_parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Input file for batch prediction",
    )
    predict_mh_parser.add_argument(
        "--output",
        type=str,
        default="predictions_multihead.csv",
        help="Output file for batch prediction",
    )
    predict_mh_parser.add_argument(
        "--text-column",
        type=str,
        default="product",
        help="Name of text column in input file",
    )
    predict_mh_parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for prediction",
    )
    predict_mh_parser.add_argument(
        "--top-k",
        type=int,
        default=1,
        help="Number of top predictions per level (default: 1)",
    )
    predict_mh_parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=None,
        help="Minimum confidence per level; stop at deepest level meeting this threshold",
    )
    predict_mh_parser.add_argument(
        "texts",
        nargs="*",
        help="Text(s) to classify (if not using --file or --input)",
    )
    predict_mh_parser.set_defaults(func=cmd_predict_multihead)

    # Evaluate-report command
    eval_report_parser = subparsers.add_parser(
        "evaluate-report",
        help="Generate comprehensive evaluation report on annotated data",
    )
    eval_report_parser.add_argument(
        "--model",
        type=str,
        default="checkpoints/basic/basic_model",
        help="Path to trained model (local dir, runs:/..., or models:/...)",
    )
    eval_report_parser.add_argument(
        "--data-dir",
        type=str,
        default="data/annotated",
        help="Directory with annotated CSVs",
    )
    eval_report_parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Max K for top-k accuracy (default: 5)",
    )
    eval_report_parser.add_argument(
        "--text-column",
        type=str,
        default="product",
        help="Text column after normalization (default: product)",
    )
    eval_report_parser.add_argument(
        "--mlflow-run-id",
        type=str,
        default=None,
        help="Existing MLflow run ID to log metrics to",
    )
    eval_report_parser.add_argument(
        "--mlflow-experiment",
        type=str,
        default=None,
        help="MLflow experiment name (creates new run if no run-id)",
    )
    eval_report_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save report to file (txt)",
    )
    eval_report_parser.add_argument(
        "--amount-threshold",
        type=float,
        default=200,
        help="Spending threshold in euros (default: 200)",
    )
    eval_report_parser.set_defaults(func=cmd_evaluate_report)

    # Evaluate-predictions command
    eval_pred_parser = subparsers.add_parser(
        "evaluate-predictions",
        help=(
            "Evaluate a prediction file (output from predict commands)"
            " by COICOP level"
        ),
    )
    eval_pred_parser.add_argument(
        "predictions",
        type=str,
        help="Path to prediction file (local parquet/CSV or s3:// URI)",
    )
    eval_pred_parser.add_argument(
        "--code-column",
        type=str,
        default="code",
        help="Column with the ground-truth COICOP code (default: code)",
    )
    eval_pred_parser.add_argument(
        "--text-column",
        type=str,
        default="product",
        help="Column with the product text (default: product)",
    )
    eval_pred_parser.add_argument(
        "--category-column",
        type=str,
        default=None,
        help=(
            "Column to group results by"
            " (e.g. source, store_type, receips_from_app)"
        ),
    )
    eval_pred_parser.add_argument(
        "--max-k",
        type=int,
        default=5,
        help="Max K for top-K accuracy (default: 5)",
    )
    eval_pred_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save text report to file",
    )
    eval_pred_parser.set_defaults(func=cmd_evaluate_predictions)

    # Build-training-data command
    build_data_parser = subparsers.add_parser(
        "build-training-data",
        help="Build a balanced training dataset from DDC extraction and synthetic data",
    )
    build_data_parser.add_argument(
        "--ddc",
        type=str,
        required=True,
        help="Path to DDC parquet file (local, S3, or HTTP URL)",
    )
    build_data_parser.add_argument(
        "--ddc-code-column",
        type=str,
        default="coicop_code",
        help="Code column in ddc",
    )
    build_data_parser.add_argument(
        "--ddc-text-column",
        type=str,
        default="description_ean",
        help="Text column in ddc",
    )
    build_data_parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output parquet file path",
    )
    build_data_parser.add_argument(
        "--synthetic",
        type=str,
        default="data/synthetic_data.csv",
        help="Path to synthetic data CSV (default: data/synthetic_data.csv)",
    )
    build_data_parser.add_argument(
        "--max-per-code",
        type=int,
        default=1000,
        help="Max DDC texts per level-4 code (default: 1000)",
    )
    build_data_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling (default: 42)",
    )
    build_data_parser.add_argument(
        "--encryption-key",
        type=str,
        default=None,
        help="Parquet encryption key (hex, 32 chars) for reading/writing encrypted files",
    )
    build_data_parser.add_argument(
        "--preprocess",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply text preprocessing (default: True)",
    )
    build_data_parser.set_defaults(func=cmd_build_training_data)

    # Extract-ddc command
    extract_ddc_parser = subparsers.add_parser(
        "extract-ddc",
        help="Extract DDC data from S3 and apply COICOP code mapping",
    )
    extract_ddc_parser.add_argument(
        "--annee",
        type=int,
        nargs="+",
        required=True,
        help="Year(s) to extract",
    )
    extract_ddc_parser.add_argument(
        "--mois",
        type=int,
        nargs="*",
        default=None,
        help="Month(s) to extract (default: all months)",
    )
    extract_ddc_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Override S3 output path",
    )
    extract_ddc_parser.add_argument(
        "--famille",
        type=str,
        default="data/famille_circana.csv",
        help="Path to famille_circana mapping CSV",
    )
    extract_ddc_parser.add_argument(
        "--memory",
        type=str,
        default="6GB",
        help="DuckDB memory limit (default: 6GB)",
    )
    extract_ddc_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the SQL without executing",
    )
    extract_ddc_parser.add_argument(
        "--encrypt",
        action="store_true",
        help="Encrypt the output parquet file (AES-GCM 256 bits)",
    )
    extract_ddc_parser.add_argument(
        "--encryption-key",
        type=str,
        default=None,
        help="Parquet encryption key (hex, 32 chars). Implies --encrypt",
    )
    extract_ddc_parser.set_defaults(func=cmd_extract_ddc)

    # Classify-llm command
    classify_llm_parser = subparsers.add_parser(
        "classify-llm",
        help="Classify a file into COICOP codes using an OpenAI-compatible LLM",
    )
    classify_llm_parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input file path (CSV or parquet, local or S3 URL)",
    )
    classify_llm_parser.add_argument(
        "--coicop",
        type=str,
        default="data/coicop_et_codes_techniques.csv",
        help="Path to COICOP taxonomy CSV (default: data/coicop_et_codes_techniques.csv)",
    )
    classify_llm_parser.add_argument(
        "--output",
        type=str,
        default="predictions_llm.csv",
        help="Output file path (CSV or parquet, local or S3; default: predictions_llm.csv)",
    )
    classify_llm_parser.add_argument(
        "--text-column",
        type=str,
        default="product",
        help="Column name containing text to classify (default: product)",
    )
    classify_llm_parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Number of texts per LLM call (default: 20)",
    )
    classify_llm_parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Max concurrent API requests (default: 10)",
    )
    classify_llm_parser.add_argument(
        "--price-column",
        type=str,
        default=None,
        help="Column name for price in euros (adds price context to LLM prompt)",
    )
    classify_llm_parser.add_argument(
        "--shop-column",
        type=str,
        default=None,
        help="Column name for shop/retailer name (adds shop context to LLM prompt)",
    )
    classify_llm_parser.add_argument(
        "--shop-category-column",
        type=str,
        default=None,
        help="Column name for shop category (adds category context to LLM prompt)",
    )
    classify_llm_parser.set_defaults(func=cmd_classify_llm)

    # Decide-coicop command
    decide_coicop_parser = subparsers.add_parser(
        "decide-coicop",
        help="Use an LLM as judge to pick the best COICOP code from multiple model predictions",
    )
    decide_coicop_parser.add_argument(
        "--lcs-file",
        type=str,
        default="predictions_lcs.parquet",
        help="Parquet file with LCS model predictions (default: predictions_lcs.parquet)",
    )
    decide_coicop_parser.add_argument(
        "--rag-file",
        type=str,
        default="predictions_rag.parquet",
        help="Parquet file with RAG model predictions (default: predictions_rag.parquet)",
    )
    decide_coicop_parser.add_argument(
        "--ttc-file",
        type=str,
        default="predictions_ttc.parquet",
        help="Parquet file with TTC deep-learning model predictions (default: predictions_ttc.parquet)",
    )
    decide_coicop_parser.add_argument(
        "--nomenclature",
        type=str,
        default="data/coicop_et_codes_techniques.csv",
        help="Path to COICOP nomenclature CSV (default: data/coicop_et_codes_techniques.csv)",
    )
    decide_coicop_parser.add_argument(
        "--id",
        default=None,
        help="Process a single observation by ID and print result. Omit for full batch mode.",
    )
    decide_coicop_parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="OpenAI-compatible model name (default: gpt-4o)",
    )
    decide_coicop_parser.add_argument(
        "--output",
        choices=["text", "json"],
        default="text",
        help="Output format for single-observation mode (default: text)",
    )
    decide_coicop_parser.add_argument(
        "--output-file",
        type=str,
        default="predictions_llm_decision.parquet",
        help=(
            "Output parquet file for batch mode (default: predictions_llm_decision.parquet). "
            "Supports automatic resume if the file already exists."
        ),
    )
    decide_coicop_parser.add_argument(
        "--concurrency",
        type=int,
        default=20,
        help="Number of parallel API calls in batch mode (default: 20)",
    )
    decide_coicop_parser.add_argument(
        "--full-nomenclature",
        action="store_true",
        help=(
            "Send the full ~700-code nomenclature with every request instead of "
            "filtering to predicted sections. Slower and more expensive."
        ),
    )
    decide_coicop_parser.set_defaults(func=cmd_decide_coicop)

    # Serve command
    serve_parser = subparsers.add_parser(
        "serve", help="Start the FastAPI prediction API server"
    )
    serve_parser.add_argument(
        "--model",
        type=str,
        default="checkpoints/hierarchical/hierarchical_model",
        help="Path to saved hierarchical model or MLflow URI (runs:/..., models:/..., mlflow-artifacts:/...)",
    )
    serve_parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the server to",
    )
    serve_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind the server to",
    )
    serve_parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )
    serve_parser.set_defaults(func=cmd_serve)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
