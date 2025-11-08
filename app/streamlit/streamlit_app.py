from __future__ import annotations

import asyncio
import json
import sys
import uuid
from pathlib import Path

import requests
import streamlit as st
import websockets

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from app.core.config import settings



def _build_ws_url(conversation_id: str, locale: str) -> str:
    return f"{settings.WS_URL}/chat/conversation?conversation_id={conversation_id}&locale={locale}"


def _init_state() -> None:
    st.session_state.setdefault("conversation_id", uuid.uuid4().hex)
    st.session_state.setdefault("locale", settings.DEFAULT_LOCALE)
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("ingest_url", "")
    st.session_state.setdefault("ingest_result", "")


async def _ws_send_recv(message: str) -> str:
    ws_url = _build_ws_url(
        conversation_id=st.session_state.conversation_id,
        locale=st.session_state.locale,
    )

    ping_interval = getattr(settings, "WS_PING_INTERVAL", 20.0)
    ping_timeout = getattr(settings, "WS_PING_TIMEOUT", 60.0)

    async with websockets.connect(
        ws_url,
        ping_interval=ping_interval,
        ping_timeout=ping_timeout,
    ) as ws:
        await ws.send(message)
        payload = await ws.recv()

    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return str(payload)
    else:
        return json.dumps(parsed, indent=2)


def _send_via_ws(message: str) -> str:
    try:
        return asyncio.run(_ws_send_recv(message))
    except Exception as e:
        return f"Unexpected websocket error: {e}"


def _call_ingest_items(source_url: str) -> str:
    """
    POST to /ingest-items on your backend with {"url": "<source>"}.
    """
    http_base = settings.BASE_HTTP_URL
    endpoint = f"{http_base}/chat/ingest-items"
    try:
        timeout = getattr(settings, "INGEST_HTTP_TIMEOUT", 60.0)
        resp = requests.post(endpoint, json={"url": source_url}, timeout=timeout)
        resp.raise_for_status()
        try:
            return json.dumps(resp.json(), indent=2)
        except ValueError:
            return resp.text
    except requests.RequestException as exc:
        return f"HTTP error calling /ingest-items: {exc}"


def _render_sidebar() -> None:
    with st.sidebar:
        st.markdown("### Session")
        st.text_input("Conversation ID", value=st.session_state.conversation_id, disabled=True)
        st.text_input("Locale", key="locale", value=st.session_state.locale)

        st.markdown("---")
        st.markdown("### Ingest source (POST /ingest-items)")
        st.text_input("Source URL", key="ingest_url")
        if st.button("Ingest now"):
            res = _call_ingest_items(st.session_state.ingest_url)
            st.session_state.ingest_result = res

        if st.session_state.ingest_result:
            st.code(st.session_state.ingest_result, language="json")


def main() -> None:
    st.set_page_config(page_title="AI Assistant", layout="wide")
    _init_state()

    st.title("AI Assistant Streamlit Client")

    _render_sidebar()

    st.divider()

    for message in st.session_state.messages:
        role = message["role"]
        with st.chat_message(role):
            st.markdown(message["content"])

    if prompt := st.chat_input("Type your message"):
        st.session_state.messages.append(
            {"role": "user", "content": prompt}
        )

        with st.chat_message("user"):
            st.markdown(prompt)

        response_text = _send_via_ws(prompt)

        st.session_state.messages.append(
            {"role": "assistant", "content": response_text}
        )

        with st.chat_message("assistant"):
            st.markdown(response_text)


if __name__ == "__main__":
    main()