import logging
from typing import Any, Dict, List, Optional

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langfuse import Langfuse

from app.core.config import settings


class LangfuseClient:
    """Minimal singleton wrapper around the Langfuse SDK."""

    _client: Optional[Langfuse] = None

    @classmethod
    def get_client(cls) -> Langfuse:
        if cls._client is None:
            cls._client = Langfuse(
                secret_key=settings.LANGFUSE_SECRET_KEY,
                public_key=settings.LANGFUSE_PUBLIC_KEY,
                host=settings.LANGFUSE_HOST,
                enabled=settings.LANGFUSE_TRACING_ENABLED,
            )
        return cls._client

    @classmethod
    def get_prompt(cls, prompt_name: str, **kwargs: Any):
        logging.debug("Fetching prompt from Langfuse: %s", prompt_name)
        return cls.get_client().get_prompt(prompt_name, **kwargs)


class LangfuseCallbackHandler(BaseCallbackHandler):
    """
    Simple Langfuse callback handler.

    Creates a single trace per run (based on `trace_name` or the first model name)
    and records an LLM span for each run_id. Chain callbacks are no-ops, keeping
    the handler focused on LLM usage tracing.
    """

    def __init__(
        self,
        session_id: Optional[str] = None,
        trace_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        enabled: bool = True,
    ) -> None:
        super().__init__()

        self.enabled = enabled and settings.LANGFUSE_TRACING_ENABLED
        self.session_id = session_id
        self.trace_name = trace_name
        self.metadata = metadata or {}

        self.client: Optional[Langfuse] = LangfuseClient.get_client() if self.enabled else None
        if not self.client:
            logging.debug("LangfuseCallbackHandler disabled; no traces will be emitted.")

        self._trace = None
        self._llm_spans: Dict[str, Any] = {}


    def on_llm_start(
        self,
        serialized: Optional[Dict[str, Any]],
        prompts: List[str],
        *,
        run_id: str,
        parent_run_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        if not self.client:
            return

        model_name = "unknown"
        if serialized:
            model_id = serialized.get("id")
            if isinstance(model_id, list) and model_id:
                model_name = model_id[-1]
            elif isinstance(model_id, str):
                model_name = model_id

        trace_title = self.trace_name or model_name or "llm-run"
        if self._trace is None:
            self._trace = self.client.trace(
                name=trace_title,
                input={"prompts": prompts},
                metadata=self.metadata,
                session_id=self.session_id,
            )

        span_metadata = {
            "model": model_name,
            **(metadata or {}),
        }
        self._llm_spans[run_id] = self._trace.span(
            name=f"llm-{run_id}",
            input=prompts,
            metadata=span_metadata,
        )

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: str,
        parent_run_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        if not self.client:
            return

        span = self._llm_spans.pop(run_id, None)
        if span is None:
            return

        outputs: List[str] = []
        if response.generations:
            for gen_list in response.generations:
                for gen in gen_list:
                    outputs.append(getattr(gen, "text", str(gen)))

        try:
            span.end(output=outputs[0] if len(outputs) == 1 else outputs)
        except Exception:
            span.end()

        if self._trace:
            try:
                self._trace.update(output=outputs[0] if outputs else None)
            except Exception:
                pass

    def on_llm_error(
        self,
        error: Exception,
        *,
        run_id: str,
        parent_run_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        if not self.client:
            return

        span = self._llm_spans.pop(run_id, None)
        if span is None:
            return

        try:
            span.end(level="ERROR", status_message=str(error))
        except Exception:
            span.end()

        if self._trace:
            try:
                self._trace.update(metadata={**self.metadata, "error": str(error)})
            except Exception:
                pass

    def flush(self) -> None:
        if self.client and hasattr(self.client, "flush"):
            try:
                self.client.flush()
            except Exception as exc:  # pragma: no cover
                logging.debug("Failed to flush Langfuse client: %s", exc)