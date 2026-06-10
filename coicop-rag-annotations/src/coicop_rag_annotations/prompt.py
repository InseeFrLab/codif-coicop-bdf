"""
Prompt loading for the annotation-based RAG.

Two sources are supported:
  - **local** (default for a first version): a markdown file split into a SYSTEM
    and a USER section, with `{placeholder}` fields filled at compile time.
  - **langfuse**: a chat prompt managed in Langfuse (same mechanism as coicop-rag).

Both return an object exposing `.compile(**kwargs) -> list[{"role", "content"}]`,
so the calling code is identical regardless of the source.
"""
import logging
import re
from collections import defaultdict
from typing import List

logger = logging.getLogger(__name__)

# Markers must be alone on their own line; this lets the file's documentation
# preamble mention them inline (in backticks) without breaking the split.
_SYSTEM_RE = re.compile(r"(?m)^<<<SYSTEM>>>[ \t]*$")
_USER_RE = re.compile(r"(?m)^<<<USER>>>[ \t]*$")


class LocalPrompt:
    """A two-message (system + user) chat prompt loaded from a local file."""

    def __init__(self, system_template: str, user_template: str):
        self.system_template = system_template
        self.user_template = user_template

    def compile(self, **kwargs) -> List[dict]:
        # None -> "" so missing optional blocks don't render as "None".
        safe = defaultdict(str, {k: ("" if v is None else v) for k, v in kwargs.items()})
        return [
            {"role": "system", "content": self.system_template.format_map(safe).strip()},
            {"role": "user", "content": self.user_template.format_map(safe).strip()},
        ]


def _load_local_prompt(path: str) -> LocalPrompt:
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    system_split = _SYSTEM_RE.split(content)
    if len(system_split) < 2:
        raise ValueError(f"Prompt file '{path}' must contain a '<<<SYSTEM>>>' marker on its own line.")
    after_system = system_split[-1]

    user_split = _USER_RE.split(after_system)
    if len(user_split) < 2:
        raise ValueError(f"Prompt file '{path}' must contain a '<<<USER>>>' marker on its own line.")

    system_part = user_split[0]
    user_part = user_split[-1]
    return LocalPrompt(system_part, user_part)


def load_prompt(config: dict):
    """
    Load the prompt according to `config['llm']`.

    - `use_langfuse: true`  -> fetch `prompt_name` (version `prompt_version`) from Langfuse.
    - otherwise              -> load the local file `prompt_file`.
    """
    llm_cfg = config["llm"]
    if llm_cfg.get("use_langfuse"):
        from langfuse import Langfuse

        logger.info("Loading prompt '%s' v%s from Langfuse...",
                    llm_cfg["prompt_name"], llm_cfg["prompt_version"])
        return Langfuse().get_prompt(
            llm_cfg["prompt_name"], version=int(llm_cfg["prompt_version"])
        )

    logger.info("Loading local prompt from %s", llm_cfg["prompt_file"])
    return _load_local_prompt(llm_cfg["prompt_file"])
