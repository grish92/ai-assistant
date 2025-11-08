import logging
from pathlib import Path
from typing import Dict, Optional

import yaml

from app.services.langfuse_client.langfuse_client import LangfuseClient

LOGGER = logging.getLogger(__name__)

_PROMPT_FILE = Path(__file__).resolve().parents[1] / "prompts" / "prompts.yaml"


class PromptNotFoundError(KeyError):
    """Raised when a requested prompt key is missing from configuration."""


def _load_prompt_definitions() -> Dict[str, Dict[str, str]]:
    if not _PROMPT_FILE.exists():
        raise FileNotFoundError(f"Prompt file not found at {_PROMPT_FILE}")

    with _PROMPT_FILE.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    return data


def get_prompt_template(prompt_key: str, *, prefer_langfuse: bool = True) -> str:
    """
    Resolve a prompt template by key.

    The function first attempts to fetch the prompt from Langfuse if a `langfuse_prompt`
    mapping is provided. If retrieval fails (network error, prompt missing, etc.),
    the local YAML template is returned as a fallback.
    """

    prompts = _load_prompt_definitions()
    config: Optional[Dict[str, str]] = prompts.get(prompt_key)

    if config is None:
        LOGGER.error("Prompt '%s' not defined in %s", prompt_key, _PROMPT_FILE)
        raise PromptNotFoundError(f"Prompt '{prompt_key}' not defined in {_PROMPT_FILE}")

    template = config.get("template")
    if template is None:
        LOGGER.error("Prompt '%s' is missing a fallback template", prompt_key)
        raise ValueError(f"Prompt '{prompt_key}' does not define a 'template' value")

    langfuse_name = config.get("langfuse_prompt")
    if prefer_langfuse and langfuse_name:
        try:
            LOGGER.debug(
                "Fetching Langfuse prompt '%s' for key '%s'",
                langfuse_name,
                prompt_key,
            )
            prompt_client = LangfuseClient.get_client().get_prompt(langfuse_name)
            template = prompt_client.get_langchain_prompt()
        except Exception as exc:  # pragma: no cover - defensive against SDK errors
            LOGGER.warning(
                "Failed to fetch Langfuse prompt '%s': %s. Falling back to YAML template.",
                langfuse_name,
                exc,
            )

    return template


def list_available_prompts() -> Dict[str, Dict[str, str]]:
    """Expose loaded prompt metadata (useful for diagnostics and debugging)."""

    prompts = _load_prompt_definitions().copy()
    LOGGER.debug("Loaded %d prompt definitions", len(prompts))
    return prompts

