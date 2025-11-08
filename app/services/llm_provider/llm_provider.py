from langchain_openai import ChatOpenAI
from app.core.config import settings


class LLMProvider:
    _llm_instance: ChatOpenAI | None = None

    @classmethod
    def get(cls) -> ChatOpenAI:
        if cls._llm_instance is None:
            cls._llm_instance = ChatOpenAI(
                model=settings.LLM_MODEL,
                temperature=settings.LLM_TEMPERATURE,
                api_key=settings.OPENAI_API_KEY,
            )
        return cls._llm_instance