from typing import Dict

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from langchain_core.documents import Document

from app.core.config import settings
from app.helpers.document_store import ensure_vectorstore, store_docs_parallel, store_docs_serial, \
     extract_html
from app.services.chat_service import ChatService
from app.core.context_vars.context_vars import FlowContextManager
from app.services.vector_reader import VectorStoreReader

router = APIRouter()

_reader = VectorStoreReader()
_chat_flow = ChatService(_reader)


@router.get("/")
async def ping():
    return {"message": "chat root works"}


@router.post("/ingest-items")
async def ingest_tems(payload: Dict[str, str]):
    url = payload.get("url", "https://newsapi.org/v2/everything?q=bitcoin")

    try:
        res = extract_html(url)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to extract articles: {e}")

    try:
        docs = await _chat_flow.generate_news_summaries(data=res.get("html", ''))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process data via chat flow: {e}")

    items = getattr(docs, "items", None)
    if not items:
        return {
            "success": True,
            "message": "Request completed, but no news items were generated from the source.",
            "stored": 0,
            "parallel": False,
        }

    lc_docs = [
        Document(
            page_content=item.summary,
            metadata={
                "source": getattr(item, "source", None),
                "title": getattr(item, "title", None),
            },
        )
        for item in items
    ]

    try:
        vs = ensure_vectorstore()
        use_parallel = len(lc_docs) > 10

        if use_parallel:
            store_docs_parallel(vs, lc_docs)
        else:
            store_docs_serial(vs, lc_docs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to store documents in vector DB: {e}")

    return {
        "success": True,
        "message": f"Ingestion completed. Stored {len(lc_docs)} document(s).",
        "stored": len(lc_docs),
        "parallel": len(lc_docs) > 10,
    }


@router.websocket("/conversation")
async def chat_ws(websocket: WebSocket):
    params = websocket.query_params
    incoming_conv_id = params.get("conversation_id")
    incoming_locale = params.get("locale", "en")
    FlowContextManager.init_for_connection(
        conversation_id=incoming_conv_id,
        flow_name="websocket-chat",
        locale=incoming_locale,
    )
    await websocket.accept(
        headers=None,
        subprotocol=None,
    )
    try:
        while True:
            text = await websocket.receive_text()
            response = await _chat_flow.handle_message(text)

            await websocket.send_json(response)
    except WebSocketDisconnect:
        pass
