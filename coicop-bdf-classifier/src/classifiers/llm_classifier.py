"""Classify texts into COICOP v2 taxonomy using an OpenAI-compatible LLM API."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path

import duckdb
import openai
import pandas as pd
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# COICOP taxonomy helpers
# ---------------------------------------------------------------------------


def load_coicop_taxonomy(coicop_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load COICOP taxonomy.

    Returns:
        (level4_df, all_df) — level4_df filtered to 4-part codes (3 dots),
        all_df contains all levels for building parent hierarchy descriptions.
    """
    df = pd.read_csv(coicop_path, sep=";", encoding="utf-8")
    df.columns = [c.strip().strip('"') for c in df.columns]
    mask_tech = ~df["Code"].str.startswith(("98", "99"))
    all_df = df[mask_tech].copy()
    level4_df = all_df[all_df["Code"].str.count(r"\.") == 3].copy()
    logger.info(f"Loaded {len(level4_df)} COICOP level-4 codes")
    return level4_df, all_df


def build_system_prompt(
    level4_df: pd.DataFrame,
    all_df: pd.DataFrame,
    context_labels: list[str] | None = None,
) -> str:
    """Build the fixed system prompt with all valid COICOP codes + hierarchy context."""
    code_to_libelle: dict[str, str] = dict(zip(all_df["Code"], all_df["Libelle"]))

    def parent_path(code: str) -> str:
        parts = code.split(".")
        ancestors = []
        for depth in range(1, len(parts)):
            ancestor_code = ".".join(parts[:depth])
            label = code_to_libelle.get(ancestor_code, "")
            if label:
                ancestors.append(label)
        return " > ".join(ancestors)

    lines = []
    for _, row in level4_df.iterrows():
        code = row["Code"]
        libelle = row["Libelle"]
        path = parent_path(code)
        if path:
            lines.append(f"{code} - {libelle}  [{path}]")
        else:
            lines.append(f"{code} - {libelle}")

    coicop_list = "\n".join(lines)
    context_instruction = ""
    if context_labels:
        fields = ", ".join(context_labels)
        context_instruction = (
            f"\nChaque produit est fourni avec des informations contextuelles "
            f"({fields}). Utilise ces informations pour affiner ta classification.\n"
        )
    return (
        "Tu es un expert en classification COICOP. "
        "Classe chaque produit dans la catégorie COICOP la plus appropriée parmi la liste suivante:\n\n"
        f"{coicop_list}\n\n"
        f"{context_instruction}"
        "Réponds UNIQUEMENT avec un tableau JSON de codes (dans le même ordre que les produits fournis). "
        'Exemple: ["01.1.1.1", "07.1.1.1", ...]'
    )


# ---------------------------------------------------------------------------
# I/O helpers (local + S3, CSV + parquet)
# ---------------------------------------------------------------------------


def _configure_s3(con: duckdb.DuckDBPyConnection) -> None:
    """Configure DuckDB S3 secret from environment variables."""
    con.execute(f"""
        CREATE SECRET secret_ls3 (
            TYPE S3,
            KEY_ID '{os.environ["AWS_ACCESS_KEY_ID"]}',
            SECRET '{os.environ["AWS_SECRET_ACCESS_KEY"]}',
            ENDPOINT '{os.environ["AWS_S3_ENDPOINT"]}',
            SESSION_TOKEN '{os.environ["AWS_SESSION_TOKEN"]}',
            REGION 'us-east-1',
            URL_STYLE 'path',
            SCOPE 's3://'
        );
    """)


def _read_file(path: str) -> pd.DataFrame:
    """Read CSV or parquet from local path or S3 URL."""
    if path.startswith("s3://"):
        con = duckdb.connect()
        _configure_s3(con)
        return con.execute(f"SELECT * FROM '{path}'").df()
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path, sep=";")


def _write_file(df: pd.DataFrame, path: str) -> None:
    """Write DataFrame to CSV or parquet, local or S3."""
    if path.startswith("s3://"):
        con = duckdb.connect()
        _configure_s3(con)
        con.register("__output", df)
        fmt = "PARQUET" if path.endswith(".parquet") else "CSV"
        con.execute(f"COPY __output TO '{path}' (FORMAT {fmt})")
        return
    if path.endswith(".parquet"):
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def _count_output_rows(path: str) -> int:
    """Count existing rows in the output file (for checkpoint/resume)."""
    p = Path(path)
    if not p.exists():
        return 0
    try:
        if path.endswith(".parquet"):
            return len(pd.read_parquet(path))
        return len(pd.read_csv(path))
    except Exception as e:
        logger.warning(f"Could not read output file for checkpoint: {e}")
        return 0


# ---------------------------------------------------------------------------
# OpenAI client
# ---------------------------------------------------------------------------


def _build_client() -> openai.AsyncOpenAI:
    """Create AsyncOpenAI client from environment variables."""
    base_url = os.environ.get("OPENAI_BASE_URL")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return openai.AsyncOpenAI(base_url=base_url)
    return openai.AsyncOpenAI(api_key=api_key, base_url=base_url)


# ---------------------------------------------------------------------------
# Batch classification
# ---------------------------------------------------------------------------


async def _classify_batch(
    client: openai.AsyncOpenAI,
    model: str,
    system_prompt: str,
    texts: list[str] | list[dict],
    valid_codes: set[str],
    batch_id: int = 0,
) -> list[str | None]:
    """Call the LLM for a batch of texts, return list of COICOP codes (or None)."""
    user_message = json.dumps(texts, ensure_ascii=False)

    async def _call() -> list[str | None]:
        t0 = time.time()
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=0,
            ),
            timeout=60,
        )
        elapsed = time.time() - t0
        logger.debug(f"Batch {batch_id} got response in {elapsed:.2f}s")
        if response is None or not response.choices:
            raise ValueError(f"Empty response from API (response={response!r})")
        content = response.choices[0].message.content
        if content is None:
            raise ValueError("API returned message with None content")
        content = content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()
        codes = json.loads(content)
        if not isinstance(codes, list):
            raise ValueError(f"Expected list, got {type(codes)}")
        result: list[str | None] = []
        for i in range(len(texts)):
            code = codes[i] if i < len(codes) else None
            result.append(str(code) if code in valid_codes else code)
        return result

    logger.debug(f"Batch {batch_id} sending API request ({len(texts)} texts)")
    try:
        return await _call()
    except asyncio.TimeoutError:
        logger.error(f"Batch {batch_id} timed out after 60s")
        return [None] * len(texts)
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        logger.warning(f"Parse error on batch {batch_id}, retrying once: {e}")
        try:
            return await _call()
        except asyncio.TimeoutError:
            logger.error(f"Batch {batch_id} timed out after 60s on retry")
            return [None] * len(texts)
        except Exception as e2:
            logger.error(f"Batch {batch_id} failed after retry: {e2}")
            return [None] * len(texts)
    except Exception as e:
        logger.error(f"API error on batch {batch_id}: {e}")
        return [None] * len(texts)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

WRITE_EVERY = 500


async def classify_llm(
    input_path: str,
    coicop_path: str,
    output_path: str,
    text_column: str = "product",
    batch_size: int = 20,
    concurrency: int = 10,
    context_columns: dict[str, str] | None = None,
) -> None:
    """Classify texts from a file into COICOP codes using an LLM.

    Reads the full input file, sends batches to an OpenAI-compatible API
    concurrently, and writes results (original columns + coicop_code +
    coicop_libelle) to the output file.  Supports checkpoint/resume by
    skipping rows already present in the output.
    """
    # Load taxonomy
    coicop_df, all_coicop_df = load_coicop_taxonomy(coicop_path)
    valid_codes = set(coicop_df["Code"].tolist())
    system_prompt = build_system_prompt(
        coicop_df,
        all_coicop_df,
        context_labels=list(context_columns.keys()) if context_columns else None,
    )
    code_to_libelle: dict[str, str] = dict(
        zip(all_coicop_df["Code"], all_coicop_df["Libelle"])
    )

    model = (
        os.environ.get("OPENAI_MODEL")
        or os.environ.get("OPENAI_DEFAULT_MODEL")
        or "gpt-4o-mini"
    )

    # Read input
    df_input = _read_file(input_path)
    if text_column not in df_input.columns:
        raise ValueError(
            f"Column '{text_column}' not found in input file. "
            f"Available columns: {list(df_input.columns)}"
        )

    # Checkpoint: skip already-processed rows
    already_done = _count_output_rows(output_path)
    if already_done > 0:
        logger.info(f"Checkpoint: {already_done} rows already in output, skipping")
        df_input = df_input.iloc[already_done:]

    if df_input.empty:
        logger.info("Nothing to classify (all rows already processed)")
        return

    total_rows = len(df_input)
    client = _build_client()
    total_written = [0]
    start_time = time.time()

    logger.info(
        f"Starting classification — model={model!r}, batch={batch_size}, "
        f"concurrency={concurrency}, rows={total_rows}, "
        f"base_url={os.environ.get('OPENAI_BASE_URL')!r}"
    )

    # Build list of (index_in_df, text) so we can match results back
    records = list(df_input.itertuples(index=False))
    if context_columns:
        missing = [
            col for col in context_columns.values() if col not in df_input.columns
        ]
        if missing:
            raise ValueError(
                f"Context columns not found in input: {missing}. "
                f"Available: {list(df_input.columns)}"
            )
        texts: list[str] | list[dict] = []
        for r in records:
            entry: dict[str, str] = {"produit": str(getattr(r, text_column))}
            for label, col_name in context_columns.items():
                val = getattr(r, col_name, None)
                if pd.notna(val):
                    entry[label] = str(val)
            texts.append(entry)
    else:
        texts = [str(getattr(r, text_column)) for r in records]

    # Split into batches
    batches: list[tuple[int, list[int]]] = []  # (batch_id, row indices)
    for i in range(0, len(texts), batch_size):
        batches.append((len(batches), list(range(i, min(i + batch_size, len(texts))))))

    with logging_redirect_tqdm():
        pbar = tqdm(
            total=total_rows, unit="row", desc="Classifying", dynamic_ncols=True
        )
        try:
            batch_queue: asyncio.Queue = asyncio.Queue(maxsize=concurrency * 3)
            result_queue: asyncio.Queue = asyncio.Queue()

            async def producer() -> None:
                for batch_id, indices in batches:
                    batch_texts = [texts[i] for i in indices]
                    await batch_queue.put((batch_id, indices, batch_texts))
                for _ in range(concurrency):
                    await batch_queue.put(None)

            async def worker() -> None:
                while True:
                    item = await batch_queue.get()
                    if item is None:
                        await result_queue.put(None)
                        return
                    batch_id, indices, batch_texts = item
                    codes = await _classify_batch(
                        client, model, system_prompt, batch_texts, valid_codes, batch_id
                    )
                    await result_queue.put((indices, codes))

            async def writer() -> None:
                sentinels = 0
                buffer_indices: list[int] = []
                buffer_codes: list[str | None] = []
                log_counter = 0

                def flush() -> None:
                    if not buffer_indices:
                        return
                    rows = df_input.iloc[buffer_indices].copy()
                    rows["coicop_code"] = buffer_codes
                    rows["coicop_libelle"] = [
                        code_to_libelle.get(c, "") if c else "" for c in buffer_codes
                    ]
                    write_header = not Path(output_path).exists() or (
                        already_done == 0 and total_written[0] == 0
                    )
                    if output_path.endswith(".parquet"):
                        # For parquet, append by reading existing + concat
                        if Path(output_path).exists() and not write_header:
                            existing = pd.read_parquet(output_path)
                            rows = pd.concat([existing, rows], ignore_index=True)
                        rows.to_parquet(output_path, index=False)
                    else:
                        rows.to_csv(
                            output_path,
                            mode="a",
                            header=write_header,
                            index=False,
                        )
                    total_written[0] += len(buffer_indices)
                    logger.info(f"Written {total_written[0]:,} rows so far")
                    buffer_indices.clear()
                    buffer_codes.clear()

                while sentinels < concurrency:
                    item = await result_queue.get()
                    if item is None:
                        sentinels += 1
                        if sentinels == concurrency:
                            await asyncio.to_thread(flush)
                        continue
                    indices, codes = item
                    buffer_indices.extend(indices)
                    buffer_codes.extend(codes)
                    pbar.update(len(indices))
                    if len(buffer_indices) >= WRITE_EVERY:
                        await asyncio.to_thread(flush)
                        log_counter += len(buffer_indices)
                        if log_counter >= 10_000:
                            elapsed = time.time() - start_time
                            rate = total_written[0] / elapsed if elapsed > 0 else 0
                            logger.info(
                                f"Progress: {total_written[0]:,} written | "
                                f"{rate:.0f} rows/s"
                            )
                            log_counter = 0

            producer_task = asyncio.create_task(producer())
            worker_tasks = [asyncio.create_task(worker()) for _ in range(concurrency)]
            writer_task = asyncio.create_task(writer())
            await asyncio.gather(producer_task, *worker_tasks, writer_task)
        finally:
            pbar.close()

    elapsed = time.time() - start_time
    if elapsed > 0:
        logger.info(
            f"Done. {total_written[0]:,} rows written in {elapsed:.1f}s "
            f"({total_written[0] / elapsed:.0f} rows/s)"
        )
    else:
        logger.info(f"Done. {total_written[0]:,} rows written.")
