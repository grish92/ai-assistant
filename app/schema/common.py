from enum import StrEnum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, HttpUrl


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


class VideoGenerationRequest(BaseModel):
    business_type: str = Field(..., min_length=2, description="Type of business commissioning the video (e.g. 'luxury travel agency').")
    product_description: str = Field(..., min_length=5, description="Concise description of the product or service being promoted.")
    product_image_urls: List[HttpUrl] = Field(
        default_factory=list,
        description="Optional publicly accessible URLs for reference imagery supplied by the user.",
    )
    aspect_ratio: Optional[str] = Field(
        None,
        description="Optional aspect ratio hint (e.g. '16:9', '9:16', '1:1').",
    )
    duration_seconds: Optional[int] = Field(
        None,
        ge=1,
        le=120,
        description="Desired duration in seconds. Leave unset to let the model decide.",
    )
    creative_direction: Optional[str] = Field(
        None,
        description="Optional creative direction or campaign theme supplied by the user.",
    )
    negative_prompt: Optional[str] = Field(
        None,
        description="Elements or themes to avoid in the generated video.",
    )
    extra_instructions: Optional[str] = Field(
        None,
        description="Additional guidance or scene details for the model.",
    )


class VideoGenerationResponse(BaseModel):
    prompt: str = Field(..., description="Final in-house prompt sent to the model.")
    model: str = Field(..., description="Model used for generation, typically 'veo-3.1'.")
    video_uri: Optional[str] = Field(
        None,
        description="URI pointing to the generated video asset (if available).",
    )
    raw_response: Optional[Dict[str, Any]] = Field(
        None,
        description="Full raw response payload returned by Google GenAI.",
    )