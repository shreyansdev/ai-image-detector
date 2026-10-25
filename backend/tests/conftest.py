import os
import sys
import io
import pytest
from PIL import Image

# Ensure backend root directory is in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture(scope="module")
def client():
    """FastAPI TestClient instance fixture."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def valid_jpeg_bytes() -> bytes:
    """Generates synthetic valid JPEG image bytes using PIL."""
    img = Image.new("RGB", (100, 100), color="blue")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture(scope="session")
def valid_png_bytes() -> bytes:
    """Generates synthetic valid PNG image bytes using PIL."""
    img = Image.new("RGB", (100, 100), color="green")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(scope="session")
def valid_webp_bytes() -> bytes:
    """Generates synthetic valid WEBP image bytes using PIL."""
    img = Image.new("RGB", (100, 100), color="red")
    buf = io.BytesIO()
    img.save(buf, format="WEBP")
    return buf.getvalue()


@pytest.fixture(scope="session")
def invalid_txt_bytes() -> bytes:
    """Text file bytes disguised with image extension."""
    return b"This is plain text content pretending to be an image file."


@pytest.fixture(scope="session")
def corrupted_image_bytes() -> bytes:
    """Corrupted random bytes with partial JPEG header."""
    return b"\xff\xd8\xff\xe0" + os.urandom(200)


@pytest.fixture(scope="session")
def oversized_file_bytes() -> bytes:
    """File bytes exceeding 10MB limit."""
    return b"\xff\xd8\xff" + b"0" * (10 * 1024 * 1024 + 500)
