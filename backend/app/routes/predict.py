import io
import time
import logging
from collections import defaultdict, deque
from typing import Dict, Deque, List, Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Request, status
from pydantic import BaseModel, Field
from PIL import Image

from app.services.inference import (
    predict,
    predict_batch_tensors,
    preprocess_image_bytes,
    is_model_loaded,
    get_models_info
)
from app.utils.security import validate_image_magic_bytes

logger = logging.getLogger("app_logger")

router = APIRouter()

VALID_MODELS = {"efficientnet", "forensicnet"}


class PredictionResponse(BaseModel):
    label: str = Field(..., description="Predicted category: 'AI Generated' or 'Real'")
    confidence: float = Field(..., description="Probability of predicted class (0.0 to 1.0)")
    inference_time_ms: float = Field(..., description="Wall-clock time of forward pass in milliseconds")
    model_used: str = Field(..., description="Identifier of the model used for prediction")
    model_display_name: str = Field(..., description="Human-readable display name of the model used")


class BatchPredictionItemResponse(BaseModel):
    filename: str = Field(..., description="Name of the uploaded file")
    success: bool = Field(..., description="Whether prediction was successful for this image")
    label: Optional[str] = Field(None, description="Predicted category: 'AI Generated' or 'Real'")
    confidence: Optional[float] = Field(None, description="Probability of predicted class (0.0 to 1.0)")
    inference_time_ms: Optional[float] = Field(None, description="Wall-clock time of forward pass in milliseconds")
    model_used: Optional[str] = Field(None, description="Identifier of the model used for prediction")
    model_display_name: Optional[str] = Field(None, description="Human-readable display name of the model used")
    error: Optional[str] = Field(None, description="Error message if success is false")


class ModelInfoResponse(BaseModel):
    id: str = Field(..., description="Unique model identifier")
    display_name: str = Field(..., description="Human-readable model name")
    description: str = Field(..., description="Brief description of architecture and capabilities")
    test_accuracy: float = Field(..., description="Test set accuracy (0.0 to 1.0)")
    auc_roc: float = Field(..., description="Area Under ROC Curve (0.0 to 1.0)")
    params: str = Field(..., description="Formatted total parameter count")
    avg_inference_ms_gpu: float = Field(..., description="Average GPU inference latency per image in ms")
    available: bool = Field(..., description="Whether model is loaded and ready for inference")


class InMemoryRateLimiter:
    """In-memory sliding window rate limiter for client IP addresses."""
    def __init__(self, requests_per_minute: int = 30):
        self.rate_limit = requests_per_minute
        self.window_seconds = 60
        self.requests: Dict[str, Deque[float]] = defaultdict(deque)

    def is_rate_limited(self, ip: str) -> bool:
        now = time.time()
        client_requests = self.requests[ip]

        while client_requests and now - client_requests[0] > self.window_seconds:
            client_requests.popleft()

        if len(client_requests) >= self.rate_limit:
            return True

        client_requests.append(now)
        return False


rate_limiter = InMemoryRateLimiter(requests_per_minute=30)
batch_rate_limiter = InMemoryRateLimiter(requests_per_minute=5)


def get_client_ip(request: Request) -> str:
    """Extracts client IP safely, accounting for X-Forwarded-For when behind a reverse proxy."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        # First IP in comma-separated list is the original client IP
        client_ip = forwarded.split(",")[0].strip()
        if client_ip:
            return client_ip
    if request.client and request.client.host:
        return request.client.host
    return "127.0.0.1"

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB in bytes


def validate_file_metadata(image: UploadFile):
    """Helper to validate content type and file extensions."""
    content_type = image.content_type.lower() if image.content_type else ""
    filename = image.filename.lower() if image.filename else ""
    ext = f".{filename.split('.')[-1]}" if "." in filename else ""

    if content_type not in ALLOWED_CONTENT_TYPES and ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Unsupported Media Type. Only JPEG, PNG, and WEBP formats are accepted."
        )


def validate_file_binary_integrity(contents: bytes):
    """Helper to validate file size, magic bytes signature, and PIL integrity."""
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Payload Too Large. Maximum allowed file size is 10MB."
        )

    if not validate_image_magic_bytes(contents):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Unsupported or invalid image file signature."
        )

    try:
        img_verify = Image.open(io.BytesIO(contents))
        img_verify.verify()
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Unprocessable Entity. Invalid or corrupted image data."
        )


def resolve_model_choice(model_form: Optional[str], request: Request) -> str:
    """Helper to extract and validate the requested model identifier."""
    query_model = request.query_params.get("model")
    chosen = query_model if query_model is not None else model_form
    if not chosen:
        chosen = "efficientnet"

    chosen_normalized = chosen.lower().strip()
    if chosen_normalized not in VALID_MODELS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid model specified: '{chosen}'. Valid options are 'efficientnet' or 'forensicnet'."
        )
    return chosen_normalized


@router.get(
    "/models",
    response_model=List[ModelInfoResponse],
    summary="Retrieve metadata and metrics for available AI detection models",
    status_code=status.HTTP_200_OK
)
async def get_models():
    """Returns metadata for all available models dynamically populated from evaluation results."""
    return get_models_info()


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Classify image as AI Generated or Real",
    status_code=status.HTTP_200_OK
)
async def predict_image(
    request: Request,
    image: UploadFile = File(...),
    model: Optional[str] = Form("efficientnet")
):
    selected_model = resolve_model_choice(model, request)

    if not is_model_loaded(selected_model):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model '{selected_model}' is not loaded or unavailable."
        )

    client_ip = get_client_ip(request)
    if rate_limiter.is_rate_limited(client_ip):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Maximum 30 requests per minute allowed."
        )

    validate_file_metadata(image)

    try:
        contents = await image.read()
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not read uploaded file."
        )

    validate_file_binary_integrity(contents)

    try:
        result = predict(contents, model_id=selected_model)
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(ve)
        )
    except RuntimeError as re:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(re)
        )
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while processing the image."
        )

    return PredictionResponse(**result)


@router.post(
    "/predict/batch",
    response_model=List[BatchPredictionItemResponse],
    summary="Classify multiple images in a batch as AI Generated or Real",
    status_code=status.HTTP_200_OK
)
async def predict_batch(
    request: Request,
    images: List[UploadFile] = File(...),
    model: Optional[str] = Form("efficientnet")
):
    selected_model = resolve_model_choice(model, request)

    if not is_model_loaded(selected_model):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model '{selected_model}' is not loaded or unavailable."
        )

    client_ip = get_client_ip(request)
    if batch_rate_limiter.is_rate_limited(client_ip):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Maximum 5 batch requests per minute allowed."
        )

    if len(images) > 10:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Maximum 10 images per batch request."
        )

    start_time = time.perf_counter()
    results: List[Optional[BatchPredictionItemResponse]] = [None] * len(images)
    success_count = 0
    failure_count = 0

    valid_indices: List[int] = []
    valid_tensors = []
    valid_filenames: List[str] = []

    # Step 1: Validate metadata, verify magic bytes, and preprocess image tensors
    for idx, image in enumerate(images):
        filename = image.filename or "unknown"
        try:
            validate_file_metadata(image)

            try:
                contents = await image.read()
            except Exception:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Could not read uploaded file."
                )

            validate_file_binary_integrity(contents)

            try:
                tensor = preprocess_image_bytes(contents)
            except ValueError as ve:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=str(ve)
                )

            valid_indices.append(idx)
            valid_tensors.append(tensor)
            valid_filenames.append(filename)

        except HTTPException as he:
            results[idx] = BatchPredictionItemResponse(
                filename=filename,
                success=False,
                error=he.detail
            )
            failure_count += 1
        except Exception as e:
            logger.error(f"Unexpected error processing batch image '{filename}': {e}", exc_info=True)
            results[idx] = BatchPredictionItemResponse(
                filename=filename,
                success=False,
                error="An error occurred while processing the image."
            )
            failure_count += 1

    # Step 2: Perform single batched GPU/CPU forward pass for all valid images
    if valid_tensors:
        try:
            batch_predictions = predict_batch_tensors(valid_tensors, model_id=selected_model)
            for idx, filename, pred_result in zip(valid_indices, valid_filenames, batch_predictions):
                results[idx] = BatchPredictionItemResponse(
                    filename=filename,
                    success=True,
                    label=pred_result["label"],
                    confidence=pred_result["confidence"],
                    inference_time_ms=pred_result["inference_time_ms"],
                    model_used=pred_result["model_used"],
                    model_display_name=pred_result["model_display_name"]
                )
                success_count += 1
        except Exception as batch_err:
            logger.error(f"Batched forward pass failed: {batch_err}", exc_info=True)
            for idx, filename in zip(valid_indices, valid_filenames):
                results[idx] = BatchPredictionItemResponse(
                    filename=filename,
                    success=False,
                    error="An error occurred while processing the image."
                )
                failure_count += 1

    duration_ms = (time.perf_counter() - start_time) * 1000.0
    logger.info(
        f"Batch prediction complete [{selected_model}]: total_images={len(images)}, "
        f"success_count={success_count}, failure_count={failure_count}, "
        f"total_processing_time_ms={duration_ms:.2f}"
    )

    return results
