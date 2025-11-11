from fastapi import APIRouter
from .chat import router as chat_router
from .video import router as video_router

router = APIRouter()
router.include_router(chat_router, prefix="/chat", tags=["chat"])
router.include_router(video_router, prefix="/video", tags=["video"])