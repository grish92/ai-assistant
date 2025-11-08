import logging

from app.core.context_vars.context_vars import FlowContextManager


class LangfuseContextFilter(logging.Filter):
    """
    Attaches message-level Langfuse trace/session identifiers if available.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            record.langfuse_trace_id = FlowContextManager.get_conversation_id() or "-"
        except Exception:
            record.langfuse_trace_id = "-"

        try:
            record.langfuse_session_id = FlowContextManager.get_user_message_id() or "-"
        except Exception:
            record.langfuse_session_id = "-"

        return True

