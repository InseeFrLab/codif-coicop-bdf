import asyncio
import logging
import time
from typing import Any, Optional
from openai import AsyncOpenAI, APIConnectionError, APIStatusError, RateLimitError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)
from tqdm.asyncio import tqdm_asyncio
from pydantic import BaseModel

logger = logging.getLogger(__name__)


async def _count_tokens(client: AsyncOpenAI, model: str, messages: list[list]) -> list[int]:
    """
    Count prompt tokens for each message using the OpenAI token-counting API
    (POST /v1/responses/input_tokens).
    Falls back to char/4 approximation if the endpoint is unavailable (e.g. vLLM).
    """
    counts = []
    for message in messages:
        try:
            resp = await client.responses.input_tokens.count(
                model=model,
                input=message,
            )
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

# ---------------------------------------------------------------------------
# Retry predicate – does not retry on 4xx "business" errors (bad request,
# refused content, etc.) but does retry on timeouts, 429, 5xx.
# ---------------------------------------------------------------------------
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
    """Single API call with exponential retry."""
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
            raise  # 4xx error → do not retry, raise immediately
        logger.warning("Retryable error: %s", exc)
        raise

# ---------------------------------------------------------------------------
# Worker: pulls tasks from a queue and writes results
# ---------------------------------------------------------------------------
async def _worker(
    worker_id: int,
    queue: asyncio.Queue,
    results: list,
    client: AsyncOpenAI,
    config: dict,
    error_policy: str,          # "raise" | "store_none" | "store_exception"
    semaphore: asyncio.Semaphore,
    pbar: tqdm_asyncio,
) -> None:
    while True:
        item = await queue.get()
        if item is None:          # poison pill
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
            else:                 # "store_none"
                results[idx] = None
        finally:
            pbar.update(1)  # incrémente exactement quand une réponse arrive
            queue.task_done()

# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
async def generate_llm_responses_async(
    messages: list[list],
    client_gen: AsyncOpenAI,
    config: dict,
    *,
    concurrency: int = 32,
    error_policy: str = "store_none",
) -> list:
    """
    Generates LLM responses in parallel with automatic retry.

    Args:
        messages:       List of conversations (each element = list of messages).
        client_gen:     AsyncOpenAI client (pointing to vllm or OpenAI).
        config:         Configuration dictionary.
        concurrency:    Max number of concurrent requests (adjust according to server).
        error_policy:   "raise"            → raises the exception on the first fatal error
                        "store_none"       → stores None for failed requests (default)
                        "store_exception"  → stores the exception for inspection

    Returns:
        List of responses (same order as messages).
    """
    logger.info("=" * 80)
    logger.info("STEP 4: LLM GENERATION (async, concurrency=%d)", concurrency)
    logger.info("=" * 80)

    n = len(messages)
    results: list = [None] * n

    # Count tokens before sending
    token_counts = await _count_tokens(client_gen, config["llm"]["model_name"], messages)
    _log_token_stats(token_counts)

    semaphore = asyncio.Semaphore(concurrency)
    queue: asyncio.Queue = asyncio.Queue()

    # Fill the queue
    for idx, msg in enumerate(messages):
        await queue.put((idx, msg))

    # Poison pills to stop workers cleanly
    for _ in range(concurrency):
        await queue.put(None)

    # La barre est créée ici et passée aux workers
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


# ---------------------------------------------------------------------------
# Synchronous wrapper (drop-in for existing code)
# ---------------------------------------------------------------------------
def generate_llm_responses(
    messages: list[list],
    client_gen,                # OpenAI *sync* OR AsyncOpenAI
    config: dict,
    *,
    concurrency: int = 32,
    error_policy: str = "store_none",
) -> list:
    """
    Synchronous wrapper – drop-in replacement for the old function.
    Automatically converts a sync client to async if needed.
    """
    from openai import OpenAI

    # Create an async client from the sync client config if needed
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
            messages,
            async_client,
            config,
            concurrency=concurrency,
            error_policy=error_policy,
        )
    )