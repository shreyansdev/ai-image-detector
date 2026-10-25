import os
import time
import logging
from logging.handlers import RotatingFileHandler
from contextlib import asynccontextmanager
from typing import Dict, Optional
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from app.services.inference import initialize_models, is_model_loaded, get_device, get_model_status
from app.routes.predict import router as predict_router

# Load environment variables from .env file
load_dotenv()


class HealthResponse(BaseModel):
    status: str = Field(..., description="Operational status of the backend service")
    model_loaded: bool = Field(..., description="Whether at least one ML model is loaded and ready")
    models: Optional[Dict[str, bool]] = Field(None, description="Load status for each individual model")
    device: Optional[str] = Field(None, description="Hardware compute device (cuda or cpu)")


# --- Strict CORS Configuration ---
ALLOWED_ORIGINS_RAW = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000")
ALLOWED_ORIGINS = [
    origin.strip() for origin in ALLOWED_ORIGINS_RAW.split(",")
    if origin.strip() and origin.strip() != "*"
]
if not ALLOWED_ORIGINS:
    ALLOWED_ORIGINS = ["http://localhost:3000"]


# --- Structured Logging & Rotating File Handler ---
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # backend
log_dir = os.path.join(base_dir, "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "app.log")

logger = logging.getLogger("app_logger")
logger.setLevel(logging.INFO)

if not logger.handlers:
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = RotatingFileHandler(log_file, maxBytes=10 * 1024 * 1024, backupCount=5)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager handling application startup and shutdown events."""
    logger.info("Initializing ML model services on application startup...")
    try:
        initialize_models()
    except Exception as e:
        logger.error(f"Failed to initialize models during startup: {e}")
    yield
    logger.info("Shutting down application...")


app = FastAPI(
    title="AI Image Detector API",
    description="Secure backend REST API for detecting AI-generated images with multi-model support.",
    version="1.1.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response


MAX_REQUEST_SIZE = 10 * 1024 * 1024  # 10MB in bytes
MAX_BATCH_REQUEST_SIZE = 110 * 1024 * 1024  # 110MB in bytes

@app.middleware("http")
async def enforce_max_request_size(request: Request, call_next):
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            is_batch = request.url.path.rstrip('/') == "/api/v1/predict/batch"
            limit = MAX_BATCH_REQUEST_SIZE if is_batch else MAX_REQUEST_SIZE
            if int(content_length) > limit:
                logger.warning(f"Rejected oversized request from {request.client.host if request.client else 'unknown'}: {content_length} bytes")
                detail_msg = (
                    "Payload Too Large. Maximum allowed request size for batch is 110MB."
                    if is_batch else
                    "Payload Too Large. Maximum allowed request size is 10MB."
                )
                return JSONResponse(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    content={"detail": detail_msg}
                )
        except ValueError:
            pass
    return await call_next(request)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start_time) * 1000.0
    client_ip = request.client.host if request.client else "unknown"
    logger.info(
        f"Client: {client_ip} | HTTP {request.method} {request.url.path} | Status: {response.status_code} | Duration: {duration_ms:.2f}ms"
    )
    return response


@app.exception_handler(HTTPException)
async def custom_http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )


@app.exception_handler(Exception)
async def global_unhandled_exception_handler(request: Request, exc: Exception):
    client_ip = request.client.host if request.client else "unknown"
    logger.error(
        f"Unhandled Exception on {request.method} {request.url.path} from client {client_ip}: {exc}",
        exc_info=True
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal server error occurred."}
    )


app.include_router(predict_router, prefix="/api/v1", tags=["Prediction"])


@app.get("/health", response_model=HealthResponse, summary="Retrieve backend health status")
async def health_check():
    """Health check endpoint returning backend status and ML model operational status."""
    return HealthResponse(
        status="operational",
        model_loaded=is_model_loaded(),
        models=get_model_status(),
        device=str(get_device())
    )


# --- Static Files Serving ---
# Mount frontend build directory to serve frontend index.html and assets
frontend_dist_dir = os.path.abspath(
    os.path.join(base_dir, "..", "frontend", "dist")
)

if os.path.exists(frontend_dist_dir):
    from fastapi.responses import FileResponse

    # Serve files from frontend/dist
    @app.get("/{file_name:path}")
    async def serve_static_or_spa(file_name: str):
        # Prevent path traversal security issue
        if file_name.startswith("api/") or file_name == "health":
            raise HTTPException(status_code=404, detail="Not Found")

        file_path = os.path.abspath(os.path.join(frontend_dist_dir, file_name))
        if not file_path.startswith(frontend_dist_dir):
            raise HTTPException(status_code=403, detail="Access Denied")

        if os.path.isfile(file_path):
            headers = {}
            if "/assets/" in file_path or file_name.startswith("assets/"):
                headers["Cache-Control"] = "public, max-age=31536000, immutable"
            elif file_name.endswith(".html"):
                headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            return FileResponse(file_path, headers=headers)

        index_path = os.path.join(frontend_dist_dir, "index.html")
        if os.path.exists(index_path):
            return FileResponse(
                index_path,
                headers={"Cache-Control": "no-cache, no-store, must-revalidate"}
            )

        raise HTTPException(status_code=404, detail="Not Found")
else:
    logger.warning(f"Frontend dist directory not found at {frontend_dist_dir}. Frontend will not be served.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
