"""
Concurrent LLM generation with retry (mirror of coicop_rag.generation_tools).

Forces a JSON-schema response: {codable: bool, code_predict: str|None, confidence: float}.
"""
import asyncio
import logging
from typing import Any, Optional

from openai import AsyncOpenAI, APIConnectionError, APIStatusError, RateLimitError
from pydantic import BaseModel
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tqdm.asyncio import tqdm_asyncio

logger = logging.getLogger(__name__)


async def _count_tokens(client: AsyncOpenAI, model: str, messages: list[list]) -> list[int]:
    """Count prompt tokens per message, falling back to char/4 if unavailable."""
    counts = []
    for message in messages:
        try:
            resp = await client.responses.input_tokens.count(model=model, input=message)
            counts.append(resp.input_tokens)
        except Exception:
            prompt_text = "\n".join(m["content"] for m in message)
            counts.append(len(prompt_text) // 4)
    return counts


def _log_token_stats(counts: list[int]) -> None:
    n = len(counts)
    total = sum(counts)
    logger.info(
        "Prompt token stats (%d messages) — min: %d  max: %d  mean: %.0f  total: %d",
        n, min(counts), max(counts), total / n, total,
    )


class ReponseFormat(BaseModel):
    codable: bool
    code_predict: Optional[str] = None
    confidence: float


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, RateLimitError):
        return True
    if isinstance(exc, APIStatusError):
        return exc.status_code in {408, 429, 500, 502, 503, 504}
    if isinstance(exc, APIConnectionError):
        return True
    return False


@retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(6),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
async def _call_with_retry(client: AsyncOpenAI, config: dict, message: list) -> Any:
    try:
        return await client.chat.completions.create(
            model=config["llm"]["model_name"],
            messages=message,
            temperature=config["llm"]["temperature"],
            max_tokens=config["llm"]["max_tokens"],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "ReponseFormat",
                    "schema": ReponseFormat.model_json_schema(),
                    "strict": True,
                },
            },
        )
    except Exception as exc:
        if not _is_retryable(exc):
            raise
        logger.warning("Retryable error: %s", exc)
        raise


async def _worker(
    worker_id: int,
    queue: asyncio.Queue,
    results: list,
    client: AsyncOpenAI,
    config: dict,
    error_policy: str,
    semaphore: asyncio.Semaphore,
    pbar: tqdm_asyncio,
) -> None:
    while True:
        item = await queue.get()
        if item is None:
            queue.task_done()
            break
        idx, message = item
        try:
            async with semaphore:
                response = await _call_with_retry(client, config, message)
            results[idx] = response
        except Exception as exc:
            logger.error("Worker %d – request %d failed permanently: %s", worker_id, idx, exc)
            if error_policy == "raise":
                queue.task_done()
                raise
            elif error_policy == "store_exception":
                results[idx] = exc
            else:
                results[idx] = None
        finally:
            pbar.update(1)
            queue.task_done()


async def generate_llm_responses_async(
    messages: list[list],
    client_gen: AsyncOpenAI,
    config: dict,
    *,
    concurrency: int = 32,
    error_policy: str = "store_none",
) -> list:
    """Generate LLM responses in parallel with automatic retry."""
    logger.info("=" * 80)
    logger.info("LLM GENERATION (async, concurrency=%d)", concurrency)
    logger.info("=" * 80)

    n = len(messages)
    results: list = [None] * n

    token_counts = await _count_tokens(client_gen, config["llm"]["model_name"], messages)
    _log_token_stats(token_counts)

    semaphore = asyncio.Semaphore(concurrency)
    queue: asyncio.Queue = asyncio.Queue()
    for idx, msg in enumerate(messages):
        await queue.put((idx, msg))
    for _ in range(concurrency):
        await queue.put(None)

    with tqdm_asyncio(total=n, desc="LLM generation") as pbar:
        workers = [
            asyncio.create_task(
                _worker(i, queue, results, client_gen, config, error_policy, semaphore, pbar)
            )
            for i in range(concurrency)
        ]
        await asyncio.gather(*workers)

    failed = sum(1 for r in results if r is None or isinstance(r, Exception))
    logger.info("✓ Responses: %d ok, %d failed (policy=%s)", n - failed, failed, error_policy)
    return results


def generate_llm_responses(
    messages: list[list],
    client_gen,
    config: dict,
    *,
    concurrency: int = 32,
    error_policy: str = "store_none",
) -> list:
    """Synchronous wrapper around `generate_llm_responses_async`."""
    from openai import OpenAI

    if isinstance(client_gen, OpenAI):
        async_client = AsyncOpenAI(
            api_key=client_gen.api_key,
            base_url=str(client_gen.base_url),
            timeout=client_gen.timeout,
        )
    else:
        async_client = client_gen

    return asyncio.run(
        generate_llm_responses_async(
            messages, async_client, config,
            concurrency=concurrency, error_policy=error_policy,
        )
    )
