"""CLI du module decide-coicop.

Arbitre LLM (« LLM-as-judge ») qui choisit le code COICOP final à partir des
prédictions des quatre classifiers (codif-lcs, run-rag, run-rag-annotations,
run-ttc). La logique réutilisable est dans ``src/decide_coicop.py``.
"""

from __future__ import annotations

import argparse
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


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
    df = load_all_observations(
        args.lcs_file,
        args.rag_file,
        args.ttc_file,
        rag_annotations_path=args.rag_annotations_file,
        mapping_path=args.mapping_file,
    )

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

    if args.extra_columns_file is not None:
        from src.decide_coicop import _read_parquet, _write_parquet

        result = _read_parquet(str(args.output_file))
        extra = _read_parquet(args.extra_columns_file)
        new_cols = [c for c in extra.columns if c not in result.columns and c != "id"]
        if new_cols:
            result = result.merge(extra[["id"] + new_cols], on="id", how="left")
            _write_parquet(result, str(args.output_file))
            logger.info("Extra columns joined to output: %s", new_cols)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="decide-coicop",
        description=(
            "Arbitre LLM du code COICOP final à partir des prédictions des "
            "classifiers (codif-lcs, run-rag, run-rag-annotations, run-ttc)."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

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
        "--rag-annotations-file",
        type=str,
        default=None,
        help=(
            "Optional parquet file with annotation-RAG predictions "
            "(columns: id, code_predict, confidence, codable). Joined on 'id' when provided."
        ),
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
        "--mapping-file",
        type=str,
        default=None,
        help=(
            "Optional parquet mapping table (mapping_lvl4.parquet from the `prune` "
            "step). When provided, LCS and TTC predicted codes are truncated to "
            "level 4 and linearly pruned so the consensus and LLM arbitration "
            "compare codes on the same pruned space."
        ),
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
    decide_coicop_parser.add_argument(
        "--extra-columns-file",
        type=str,
        default=None,
        help=(
            "Parquet file containing additional columns to join (on 'id') into the output "
            "after batch processing. Columns already present in the output are skipped."
        ),
    )
    decide_coicop_parser.set_defaults(func=cmd_decide_coicop)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
