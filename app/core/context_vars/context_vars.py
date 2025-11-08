from __future__ import annotations
import uuid
from contextvars import ContextVar
from app.schema.common import Languages


class FlowContextManager:
    """
    Class wrapper around contextvars so you can call:
    FlowContextManager.init_for_connection(...)
    FlowContextManager.init_for_message()
    FlowContextManager.get_conversation_id()
    """

    _conversation_id: ContextVar[str] = ContextVar("conversation_id", default="")
    _user_message_id: ContextVar[str] = ContextVar("user_message_id", default="")
    _log_id: ContextVar[str] = ContextVar("log_id", default="")
    _trace_id: ContextVar[str] = ContextVar("trace_id", default="")
    _flow_name: ContextVar[str] = ContextVar("flow_name", default="default")
    _locale: ContextVar[str] = ContextVar("locale", default=Languages.EN.value)

    @classmethod
    def init_for_connection(
        cls,
        *,
        conversation_id: str | None,
        flow_name: str = "websocket-chat",
        locale: str = Languages.EN.value,
    ) -> None:
        conv_id = conversation_id or str(uuid.uuid4())
        cls._conversation_id.set(conv_id)
        cls._flow_name.set(flow_name)
        cls._locale.set(locale)

    @classmethod
    def init_for_message(cls) -> None:
        cls._user_message_id.set(str(uuid.uuid4()))
        cls._log_id.set(str(uuid.uuid4()))
        cls._trace_id.set(str(uuid.uuid4()))

    # getters
    @classmethod
    def get_conversation_id(cls) -> str:
        return cls._conversation_id.get()

    @classmethod
    def get_user_message_id(cls) -> str:
        return cls._user_message_id.get()

    @classmethod
    def get_locale(cls) -> str:
        return cls._locale.get()
