from __future__ import annotations

import asyncio
import base64
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional

from google import genai
from google.genai import types

from app.core.config import settings
from app.schema.common import VideoGenerationRequest, VideoGenerationResponse

LOGGER = logging.getLogger(__name__)

BASE_MARKETING_PROMPT = (
    "You are a professional marketing specialist and cinematic video creator tasked with crafting a high-converting promotional spot. "
    "Develop a visually compelling narrative for the business described below. Maintain a polished, aspirational tone suitable for premium campaigns. "
    "Open with a striking hook, highlight differentiators, showcase lifestyle benefits, and end with a confident call to action. "
    "Ensure the video feels on-brand, modern, and tailored to the target market."
)


class VideoGenerationError(RuntimeError):
    """Raised when the Google GenAI video generation call fails."""


@dataclass(slots=True)
class _IncomingImage:
    filename: str
    content: bytes
    mime_type: Optional[str]


@dataclass(slots=True)
class _PreparedImage:
    display_name: str | None
    mime_type: str
    raw_bytes: bytes


class VideoGenerationService:
    """Orchestrates Google GenAI calls to create marketing-ready videos."""

    _configured: bool = False
    _client: genai.Client | None = None

    def __init__(self, model_name: str = "veo-3.1-generate-preview") -> None:
        self.model_name = model_name
        self._ensure_client()

    @classmethod
    def _ensure_client(cls) -> None:
        if cls._configured:
            return

        api_key = settings.GOOGLE_GENAI_API_KEY
        if not api_key:
            raise VideoGenerationError(
                "GOOGLE_GENAI_API_KEY is not configured. Set it before calling the video generation endpoint."
            )

        cls._client = genai.Client(api_key=api_key)
        cls._configured = True

    async def generate_video(
        self,
        payload: VideoGenerationRequest,
        images: Iterable[_IncomingImage],
    ) -> VideoGenerationResponse:
        images = list(images)
        if not images:
            raise VideoGenerationError("Please provide at least one product image to accompany the request.")

        prepared_images = [self._prepare_image(image) for image in images]

        try:
            result = await asyncio.to_thread(self._generate_sync, payload, prepared_images)
        except Exception as exc:  # pragma: no cover - defensive for SDK errors
            LOGGER.exception("Video generation failed: %s", exc)
            raise VideoGenerationError(str(exc)) from exc

        return result

    def _generate_sync(
        self,
        payload: VideoGenerationRequest,
        prepared_images: List[_PreparedImage],
    ) -> VideoGenerationResponse:
        final_prompt = self._compose_prompt(payload, prepared_images)

        image_part = types.Part.from_bytes(
            data=prepared_images[0].raw_bytes,
            mime_type=prepared_images[0].mime_type,
        )
        if hasattr(image_part, "as_image"):
            image_input = image_part.as_image()
        else:
            image_input = {
                "mime_type": prepared_images[0].mime_type,
                "bytes_base64_encoded": base64.b64encode(prepared_images[0].raw_bytes).decode("ascii"),
            }

        client = self._get_client()
        operation = client.models.generate_videos(
            model=self.model_name,
            prompt=final_prompt,
            image=image_input,
        )

        operation = self._await_operation(client, operation)
        if not getattr(operation, "done", False):
            raise VideoGenerationError("Video generation did not complete within the allotted time.")

        response_payload = getattr(operation, "response", None)
        generated_videos = getattr(response_payload, "generated_videos", []) if response_payload else []
        if not generated_videos:
            raise VideoGenerationError("Video generation completed but no video assets were returned.")

        video_asset = generated_videos[0]
        video_file = getattr(video_asset, "video", None)
        video_uri = getattr(video_file, "uri", None) or getattr(video_file, "file_uri", None)

        response_dict = None
        if hasattr(operation, "to_dict"):
            response_dict = operation.to_dict()
        elif hasattr(response_payload, "to_dict"):
            response_dict = response_payload.to_dict()
        elif hasattr(response_payload, "model_dump"):
            response_dict = response_payload.model_dump()
        elif isinstance(response_payload, dict):
            response_dict = response_payload

        return VideoGenerationResponse(
            prompt=final_prompt,
            model=self.model_name,
            video_uri=video_uri,
            raw_response=response_dict,
        )

    @staticmethod
    def _compose_prompt(payload: VideoGenerationRequest, prepared_images: List[_PreparedImage]) -> str:
        uploaded_image_lines = "\n".join(
            f"- {img.display_name or Path('uploaded-image').stem}"
            for img in prepared_images
        )
        reference_url_lines = "\n".join(f"- {url}" for url in payload.product_image_urls or [])
        creative_direction = payload.creative_direction.strip() if payload.creative_direction else ""

        prompt_sections = [
            BASE_MARKETING_PROMPT,
            f"Business type: {payload.business_type}.",
            f"Product focus: {payload.product_description}.",
            "Uploaded product imagery to reference:",
            uploaded_image_lines or "- No imagery provided.",
        ]

        if reference_url_lines:
            prompt_sections.extend(
                [
                    "Additional reference URLs supplied by the user:",
                    reference_url_lines,
                ]
            )

        if payload.aspect_ratio:
            prompt_sections.append(f"Target aspect ratio: {payload.aspect_ratio}.")
        if payload.duration_seconds:
            prompt_sections.append(f"Desired runtime: {payload.duration_seconds} seconds.")

        prompt_sections.extend(
            [
                "Creative direction:",
                creative_direction or "- Follow best practices for premium marketing content.",
            ]
        )

        if payload.extra_instructions:
            prompt_sections.append(f"Additional guidance from the requester: {payload.extra_instructions}")
        if payload.negative_prompt:
            prompt_sections.append(f"Elements to avoid: {payload.negative_prompt}")

        return "\n".join(section for section in prompt_sections if section)

    @staticmethod
    def _prepare_image(image: _IncomingImage) -> _PreparedImage:
        if not image.content:
            raise VideoGenerationError(f"Image '{image.filename}' is empty.")

        mime_type = image.mime_type or _guess_mime_type(image.filename)
        if not mime_type:
            raise VideoGenerationError(f"Unable to determine MIME type for '{image.filename}'.")

        return _PreparedImage(
            display_name=image.filename,
            mime_type=mime_type,
            raw_bytes=image.content,
        )

    @classmethod
    def _get_client(cls) -> genai.Client:
        cls._ensure_client()
        assert cls._client is not None
        return cls._client

    def _await_operation(
        self,
        client: genai.Client,
        operation: Any,
        *,
        timeout: int = 180,
        interval: int = 10,
    ) -> Any:
        """Poll the long-running video generation operation until completion or timeout."""
        waited = 0
        current = operation
        while not getattr(current, "done", False) and waited < timeout:
            LOGGER.info("Waiting for video generation to complete... (%ss)", waited)
            time.sleep(interval)
            waited += interval
            current = client.operations.get(current)
        if not getattr(current, "done", False):
            raise VideoGenerationError("Video generation did not complete within the allotted time.")
        return current


def _guess_mime_type(filename: str) -> Optional[str]:
    suffix = Path(filename).suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"
    if suffix == ".gif":
        return "image/gif"
    return None

