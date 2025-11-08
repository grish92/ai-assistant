from enum import StrEnum
from pydantic import BaseModel, Field
from typing import List, Optional


class Languages(StrEnum):
    EN = "en"
    RU = "ru"
    ARM = "arm"

class StructuredOutputModes:
    OPENAI_JSON_SCHEMA: str = "openai_json_schema"



class NewsItem(BaseModel):
    title: str = Field(...,description="Article headline or title.")
    summary: str = Field(...,description="3–5 sentence factual summary capturing key details and context.")
    source: Optional[str] = Field(None,description="News source name or null if unavailable.")


class NewsList(BaseModel):
    items: List[NewsItem] = Field(...,description="A list of summarized news articles.")

class NewsIntentResponse(BaseModel):
    is_news: bool=Field(...,description="Whether the user query is news related or no.")