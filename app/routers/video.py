import logging
from typing import List

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import ValidationError

from app.schema.common import VideoGenerationRequest, VideoGenerationResponse
from app.services.video_generation.video_service import (
    VideoGenerationError,
    VideoGenerationService,
    _IncomingImage,
)

router = APIRouter()
LOGGER = logging.getLogger(__name__)

_video_service: VideoGenerationService | None = None


def _get_service() -> VideoGenerationService:
    global _video_service

    if _video_service is None:
        _video_service = VideoGenerationService()
    return _video_service


@router.post(
    "/generate",
    response_model=VideoGenerationResponse,
    status_code=201,
    summary="Generate a marketing video with Google GenAI Veo 3.1",
)
async def generate_video(
    payload: str = Form(..., description="JSON string matching VideoGenerationRequest."),
    product_images: List[UploadFile] = File(..., description="One or more product images."),
) -> VideoGenerationResponse:
    """Produce a video by combining business details and uploaded imagery."""
    try:
        parsed_payload = VideoGenerationRequest.model_validate_json(payload)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc

    image_inputs: List[_IncomingImage] = []
    for upload in product_images:
        content = await upload.read()
        image_inputs.append(
            _IncomingImage(
                filename=upload.filename or "product-image",
                content=content,
                mime_type=upload.content_type,
            )
        )

    service = _get_service()
    try:
        response = await service.generate_video(parsed_payload, image_inputs)
    except VideoGenerationError as exc:
        LOGGER.warning("Video generation failed: %s", exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    if response.video_uri is None:
        raise HTTPException(
            status_code=502,
            detail="Video generation completed but no video URI was returned by Google GenAI.",
        )

    return response

