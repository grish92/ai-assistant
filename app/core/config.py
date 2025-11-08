from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    PROJECT_NAME: str
    OPENAI_API_KEY: str
    LLM_MODEL: str
    LLM_TEMPERATURE: float

    LANGFUSE_PUBLIC_KEY: str
    LANGFUSE_SECRET_KEY: str
    LANGFUSE_HOST: str
    LANGFUSE_TRACING_ENABLED: bool

    QDRANT_URL: str
    QDRANT_API_KEY: str
    QDRANT_COLLECTION: str
    EMBEDDING_DIM: int
    APP_HOST: str
    APP_PORT: int
    APP_ENV: str = "local"

    WS_URL: str
    BASE_HTTP_URL: str
    DEFAULT_LOCALE: str

    WS_PING_INTERVAL: float = 20.0
    WS_PING_TIMEOUT: float = 60.0
    INGEST_HTTP_TIMEOUT: float = 60.0

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()
