import logging
import logging.config
from pathlib import Path
from typing import Any, Dict

DEFAULT_LOG_LEVEL = "INFO"


def _default_logging_dict() -> Dict[str, Any]:
    log_format = (
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s "
        "| trace=%(langfuse_trace_id)s session=%(langfuse_session_id)s"
    )

    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": log_format,
                "datefmt": "%Y-%m-%d %H:%M:%S",
                "class": "logging.Formatter",
            },
        },
        "filters": {
            "langfuse_context": {
                "()": "app.core.logging_filters.LangfuseContextFilter",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "standard",
                "filters": ["langfuse_context"],
            },
        },
        "root": {
            "level": DEFAULT_LOG_LEVEL,
            "handlers": ["console"],
        },
    }


def setup_logging(config_path: Path | None = None) -> None:
    """
    Configure Python logging for the entire project.

    Looks for an optional logging YAML/JSON config path; otherwise applies
    the default project configuration.
    """

    if config_path and config_path.exists():
        logging.config.fileConfig(config_path, disable_existing_loggers=False)
        return

    logging.config.dictConfig(_default_logging_dict())


