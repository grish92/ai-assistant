import os
from pathlib import Path

from fastapi import FastAPI

from app.core.logging_config import setup_logging
from app.routers import router as api_router

_logging_config_path = os.getenv("APP_LOGGING_CONFIG")
setup_logging(Path(_logging_config_path)) if _logging_config_path else setup_logging()

app = FastAPI(title="AI Service")
app.include_router(api_router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )