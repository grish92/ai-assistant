from __future__ import annotations
from contextvars import ContextVar
from typing import Any


class ContextVarValue:
    """
    Descriptor that holds a ContextVar under a given key.
    """

    def __init__(self, key: str, *, log: bool = True, default: Any = None):
        self.key = key
        self.log = log
        if default is not None:
            self._ctx = ContextVar(key, default=default)
        else:
            self._ctx = ContextVar(key)

    def get(self) -> Any:
        return self._ctx.get()

    def set(self, value: Any) -> None:
        self._ctx.set(value)


