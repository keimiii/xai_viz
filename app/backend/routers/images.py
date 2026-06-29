"""Image serving endpoints."""

from __future__ import annotations

from io import BytesIO
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from app.backend.config import AVAILABLE_MODELS, STYLE_NAMES
from app.backend.schemas import (
    BoundingBoxSchema,
    ImageAnnotationSchema,
    ImageDetailSchema,
    ImageListItem,
)
from app.backend.services.image_service import image_service

router = APIRouter(prefix="/images", tags=["images"])


@router.get("", response_model=list[ImageListItem])
async def list_images(
    style: Annotated[str | None, Query(description="Filter by style name")] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 139,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[ImageListItem]:
    """List all annotated images with pagination.

    Returns summary info for each image including thumbnail URL.
    """
    image_ids = image_service.list_image_ids()

    # Single pass: filter by style (if specified) and build items together
    items = []
    for image_id in image_ids:
        annotation = image_service.get_annotation(image_id)
        if not annotation:
            continue

        style_names = image_service.get_style_names(list(annotation.styles))

        if style and style not in style_names:
            continue

        items.append(
            ImageListItem(
                image_id=image_id,
                thumbnail_url=f"/api/images/{image_id}/thumbnail",
                styles=list(annotation.styles),
                style_names=style_names,
                num_bboxes=annotation.num_bboxes,
            )
        )

    # Apply pagination
    return items[offset : offset + limit]


@router.get("/styles", response_model=list[str])
async def list_styles() -> list[str]:
    """List all available architectural styles."""
    return list(STYLE_NAMES)


@router.get("/{image_id}", response_model=ImageDetailSchema)
async def get_image_detail(image_id: str) -> ImageDetailSchema:
    """Get detailed information about an image including annotations."""
    annotation = image_service.get_annotation(image_id)
    if not annotation:
        raise HTTPException(status_code=404, detail=f"Image not found: {image_id}")

    if not image_service.image_exists(image_id):
        raise HTTPException(status_code=404, detail=f"Image file missing: {image_id}")

    style_names = image_service.get_style_names(list(annotation.styles))

    # Build bbox list with feature names
    bboxes = []
    for bbox in annotation.bboxes:
        bboxes.append(
            BoundingBoxSchema(
                left=bbox.left,
                top=bbox.top,
                width=bbox.width,
                height=bbox.height,
                label=bbox.label,
                label_name=image_service.get_feature_name(bbox.label),
            )
        )

    annotation_schema = ImageAnnotationSchema(
        image_id=image_id,
        styles=list(annotation.styles),
        style_names=style_names,
        num_bboxes=annotation.num_bboxes,
        bboxes=bboxes,
    )

    return ImageDetailSchema(
        image_id=image_id,
        image_url=f"/api/images/{image_id}/file",
        thumbnail_url=f"/api/images/{image_id}/thumbnail",
        annotation=annotation_schema,
        available_models=AVAILABLE_MODELS,
    )


@router.get("/{image_id}/file")
async def get_image_file(
    image_id: str,
    size: Annotated[int | None, Query(description="Resize to size x size")] = None,
) -> StreamingResponse:
    """Serve the original image file."""
    if not image_service.image_exists(image_id):
        raise HTTPException(status_code=404, detail=f"Image not found: {image_id}")

    try:
        resize_size = (size, size) if size else None
        img = image_service.load_image(image_id, size=resize_size)

        # Convert to bytes
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=90)
        buf.seek(0)

        return StreamingResponse(
            buf,
            media_type="image/jpeg",
            headers={"Cache-Control": "max-age=3600"},
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Image not found: {image_id}") from None


@router.get("/{image_id}/thumbnail")
async def get_thumbnail(image_id: str) -> StreamingResponse:
    """Serve a thumbnail version of the image."""
    if not image_service.image_exists(image_id):
        raise HTTPException(status_code=404, detail=f"Image not found: {image_id}")

    try:
        img = image_service.load_thumbnail(image_id)

        buf = BytesIO()
        img.save(buf, format="JPEG", quality=85)
        buf.seek(0)

        return StreamingResponse(
            buf,
            media_type="image/jpeg",
            headers={"Cache-Control": "max-age=86400"},  # 24 hours
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Image not found: {image_id}") from None


@router.get("/{image_id}/with_bboxes")
async def get_image_with_bboxes(image_id: str) -> StreamingResponse:
    """Serve original image with bounding boxes drawn."""
    bbox_path = image_service.get_original_with_bbox_path(image_id)
    if not bbox_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Bbox image not found. Run generate_heatmap_images.py first.",
        )

    with open(bbox_path, "rb") as f:
        return StreamingResponse(
            BytesIO(f.read()),
            media_type="image/png",
            headers={"Cache-Control": "max-age=86400"},
        )
