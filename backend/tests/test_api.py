import os
import torch
import pytest
from app.utils.security import validate_image_magic_bytes
from app.services.inference import predict, predict_batch_tensors, find_model_weights_path, is_model_loaded


def test_health_endpoint(client):
    """1. Test GET /health returns 200 and valid JSON status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "operational"
    assert "model_loaded" in data
    assert isinstance(data["model_loaded"], bool)
    assert "models" in data
    assert isinstance(data["models"], dict)


def test_predict_valid_jpeg(client, valid_jpeg_bytes):
    """2. Test POST /api/v1/predict with valid JPEG returns 200 and expected fields."""
    response = client.post(
        "/api/v1/predict",
        files={"image": ("sample.jpg", valid_jpeg_bytes, "image/jpeg")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["label"] in ["AI Generated", "Real"]
    assert isinstance(data["confidence"], float)
    assert 0.0 <= data["confidence"] <= 1.0
    assert isinstance(data["inference_time_ms"], float)
    assert data["inference_time_ms"] >= 0.0
    assert data["model_used"] == "efficientnet"
    assert data["model_display_name"] == "EfficientNet-B0"


def test_predict_valid_png(client, valid_png_bytes):
    """3. Test POST /api/v1/predict with valid PNG returns 200."""
    response = client.post(
        "/api/v1/predict",
        files={"image": ("sample.png", valid_png_bytes, "image/png")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["label"] in ["AI Generated", "Real"]
    assert 0.0 <= data["confidence"] <= 1.0
    assert data["model_used"] == "efficientnet"
    assert data["model_display_name"] == "EfficientNet-B0"


def test_predict_rejects_non_image(client, invalid_txt_bytes):
    """4. Test POST with a text file disguised as .jpg returns 415 or 422."""
    response = client.post(
        "/api/v1/predict",
        files={"image": ("disguised.jpg", invalid_txt_bytes, "image/jpeg")}
    )
    assert response.status_code in [415, 422]


def test_predict_rejects_oversized_file(client, oversized_file_bytes):
    """5. Test POST with a file larger than 10MB returns HTTP 413."""
    response = client.post(
        "/api/v1/predict",
        files={"image": ("large.jpg", oversized_file_bytes, "image/jpeg")}
    )
    assert response.status_code == 413


def test_predict_rejects_corrupted_image(client, corrupted_image_bytes):
    """6. Test POST with corrupted image bytes returns HTTP 415 or 422."""
    response = client.post(
        "/api/v1/predict",
        files={"image": ("corrupt.jpg", corrupted_image_bytes, "image/jpeg")}
    )
    assert response.status_code in [415, 422]


def test_magic_bytes_validation(valid_jpeg_bytes, valid_png_bytes, valid_webp_bytes, invalid_txt_bytes):
    """7. Unit test for validate_image_magic_bytes() function."""
    assert validate_image_magic_bytes(valid_jpeg_bytes) is True
    assert validate_image_magic_bytes(valid_png_bytes) is True
    assert validate_image_magic_bytes(valid_webp_bytes) is True
    assert validate_image_magic_bytes(invalid_txt_bytes) is False
    assert validate_image_magic_bytes(b"Short") is False


def test_inference_service_output_shape(valid_jpeg_bytes):
    """8. Unit test for inference.predict() return dict shape and value types."""
    res = predict(valid_jpeg_bytes)
    assert isinstance(res, dict)
    assert "label" in res and res["label"] in ["AI Generated", "Real"]
    assert "confidence" in res and isinstance(res["confidence"], float)
    assert 0.0 <= res["confidence"] <= 1.0
    assert "inference_time_ms" in res and isinstance(res["inference_time_ms"], float)
    assert "model_used" in res and res["model_used"] == "efficientnet"
    assert "model_display_name" in res and res["model_display_name"] == "EfficientNet-B0"


def test_get_models_endpoint(client):
    """9. Test GET /api/v1/models returns metadata for both available models."""
    response = client.get("/api/v1/models")
    assert response.status_code == 200
    models = response.json()
    assert isinstance(models, list)
    assert len(models) == 2

    model_ids = {m["id"]: m for m in models}
    assert "efficientnet" in model_ids
    assert "forensicnet" in model_ids

    eff = model_ids["efficientnet"]
    assert eff["display_name"] == "EfficientNet-B0"
    assert eff["test_accuracy"] == 0.9782
    assert eff["auc_roc"] == 0.9979
    assert eff["params"] == "4.66M"
    assert eff["avg_inference_ms_gpu"] == 3.78
    assert eff["available"] is True

    fn = model_ids["forensicnet"]
    assert fn["display_name"] == "ForensicNet"
    assert fn["test_accuracy"] == 0.9483
    assert fn["auc_roc"] == 0.9903
    assert fn["params"] == "1.05M"
    assert fn["avg_inference_ms_gpu"] == 1.78
    assert fn["available"] is True


def test_predict_with_efficientnet_model(client, valid_jpeg_bytes):
    """10. Test POST /api/v1/predict with explicit model='efficientnet'."""
    response = client.post(
        "/api/v1/predict",
        data={"model": "efficientnet"},
        files={"image": ("sample.jpg", valid_jpeg_bytes, "image/jpeg")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["model_used"] == "efficientnet"
    assert data["model_display_name"] == "EfficientNet-B0"
    assert data["label"] in ["AI Generated", "Real"]


def test_predict_with_forensicnet_model(client, valid_jpeg_bytes):
    """11. Test POST /api/v1/predict with explicit model='forensicnet'."""
    response = client.post(
        "/api/v1/predict",
        data={"model": "forensicnet"},
        files={"image": ("sample.jpg", valid_jpeg_bytes, "image/jpeg")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["model_used"] == "forensicnet"
    assert data["model_display_name"] == "ForensicNet"
    assert data["label"] in ["AI Generated", "Real"]


def test_predict_with_invalid_model(client, valid_jpeg_bytes):
    """12. Test POST /api/v1/predict with an invalid model name returns HTTP 400."""
    response = client.post(
        "/api/v1/predict",
        data={"model": "invalid_supernet_3000"},
        files={"image": ("sample.jpg", valid_jpeg_bytes, "image/jpeg")}
    )
    assert response.status_code == 400
    data = response.json()
    assert "Invalid model specified" in data["detail"]
    assert "efficientnet" in data["detail"]
    assert "forensicnet" in data["detail"]


def test_predict_batch_valid(client, valid_jpeg_bytes, valid_png_bytes, valid_webp_bytes):
    """13. Test POST /api/v1/predict/batch with a valid batch of 3 images."""
    files = [
        ("images", ("image1.jpg", valid_jpeg_bytes, "image/jpeg")),
        ("images", ("image2.png", valid_png_bytes, "image/png")),
        ("images", ("image3.webp", valid_webp_bytes, "image/webp")),
    ]
    response = client.post("/api/v1/predict/batch", files=files)
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 3

    assert data[0]["filename"] == "image1.jpg"
    assert data[0]["success"] is True
    assert data[0]["label"] in ["AI Generated", "Real"]
    assert data[0]["model_used"] == "efficientnet"
    assert data[0]["model_display_name"] == "EfficientNet-B0"

    assert data[1]["filename"] == "image2.png"
    assert data[1]["success"] is True
    assert data[1]["label"] in ["AI Generated", "Real"]

    assert data[2]["filename"] == "image3.webp"
    assert data[2]["success"] is True
    assert data[2]["label"] in ["AI Generated", "Real"]


def test_predict_batch_with_forensicnet(client, valid_jpeg_bytes, valid_png_bytes):
    """14. Test POST /api/v1/predict/batch with model='forensicnet'."""
    files = [
        ("images", ("image1.jpg", valid_jpeg_bytes, "image/jpeg")),
        ("images", ("image2.png", valid_png_bytes, "image/png")),
    ]
    response = client.post(
        "/api/v1/predict/batch",
        data={"model": "forensicnet"},
        files=files
    )
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 2
    assert data[0]["model_used"] == "forensicnet"
    assert data[0]["model_display_name"] == "ForensicNet"
    assert data[1]["model_used"] == "forensicnet"
    assert data[1]["model_display_name"] == "ForensicNet"


def test_predict_batch_exceeds_max(client, valid_jpeg_bytes):
    """15. Test POST /api/v1/predict/batch with 11 images returns HTTP 400."""
    files = [
        ("images", (f"image{i}.jpg", valid_jpeg_bytes, "image/jpeg"))
        for i in range(11)
    ]
    response = client.post("/api/v1/predict/batch", files=files)
    assert response.status_code == 400
    data = response.json()
    assert data["detail"] == "Maximum 10 images per batch request."


def test_predict_batch_partial_success(client, valid_jpeg_bytes, corrupted_image_bytes, valid_png_bytes):
    """16. Test POST /api/v1/predict/batch with one corrupted image returns partial success."""
    files = [
        ("images", ("valid1.jpg", valid_jpeg_bytes, "image/jpeg")),
        ("images", ("corrupted.jpg", corrupted_image_bytes, "image/jpeg")),
        ("images", ("valid2.png", valid_png_bytes, "image/png")),
    ]
    response = client.post("/api/v1/predict/batch", files=files)
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 3

    assert data[0]["filename"] == "valid1.jpg"
    assert data[0]["success"] is True
    assert data[0]["label"] in ["AI Generated", "Real"]

    assert data[1]["filename"] == "corrupted.jpg"
    assert data[1]["success"] is False
    assert "error" in data[1]
    assert data[1]["error"] is not None

    assert data[2]["filename"] == "valid2.png"
    assert data[2]["success"] is True
    assert data[2]["label"] in ["AI Generated", "Real"]


def test_unavailable_model_returns_503(client, valid_jpeg_bytes):
    """17. Test requesting an unavailable model returns HTTP 503."""
    from app.services import inference
    orig_model = inference._models["forensicnet"]
    orig_status = inference._model_statuses["forensicnet"]
    try:
        inference._models["forensicnet"] = None
        inference._model_statuses["forensicnet"] = False

        response = client.post(
            "/api/v1/predict",
            data={"model": "forensicnet"},
            files={"image": ("sample.jpg", valid_jpeg_bytes, "image/jpeg")}
        )
        assert response.status_code == 503
        data = response.json()
        assert "Model 'forensicnet' is not loaded or unavailable." in data["detail"]

        models_resp = client.get("/api/v1/models")
        assert models_resp.status_code == 200
        models_data = {m["id"]: m for m in models_resp.json()}
        assert models_data["forensicnet"]["available"] is False
        assert models_data["efficientnet"]["available"] is True

    finally:
        inference._models["forensicnet"] = orig_model
        inference._model_statuses["forensicnet"] = orig_status


def test_security_headers_include_hsts(client):
    """18. Test security headers include HSTS, nosniff, and DENY."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.headers.get("Strict-Transport-Security") == "max-age=31536000; includeSubDomains"
    assert response.headers.get("X-Content-Type-Options") == "nosniff"
    assert response.headers.get("X-Frame-Options") == "DENY"


def test_rate_limiter_uses_forwarded_for(client, valid_jpeg_bytes):
    """19. Test rate limiter distinguishes clients by X-Forwarded-For header."""
    for i in range(5):
        resp = client.post(
            "/api/v1/predict",
            headers={"X-Forwarded-For": f"198.51.100.{i}"},
            files={"image": ("sample.jpg", valid_jpeg_bytes, "image/jpeg")}
        )
        assert resp.status_code == 200


def test_batched_tensor_forward_pass():
    """20. Test predict_batch_tensors with stacked tensors runs in a single forward pass."""
    dummy_tensors = [torch.zeros(1, 3, 224, 224) for _ in range(4)]
    
    # Test EfficientNet batched forward pass
    eff_results = predict_batch_tensors(dummy_tensors, model_id="efficientnet")
    assert len(eff_results) == 4
    for item in eff_results:
        assert item["label"] in ["AI Generated", "Real"]
        assert item["model_used"] == "efficientnet"
        assert item["model_display_name"] == "EfficientNet-B0"
        assert item["inference_time_ms"] >= 0.0

    # Test ForensicNet batched forward pass
    fn_results = predict_batch_tensors(dummy_tensors, model_id="forensicnet")
    assert len(fn_results) == 4
    for item in fn_results:
        assert item["label"] in ["AI Generated", "Real"]
        assert item["model_used"] == "forensicnet"
        assert item["model_display_name"] == "ForensicNet"
        assert item["inference_time_ms"] >= 0.0


def test_find_model_weights_path_env_override(tmp_path, monkeypatch):
    """21. Test MODEL_WEIGHTS_PATH env var override correctly finds weight file."""
    fake_weight_file = tmp_path / "custom_effnet.pth"
    fake_weight_file.write_text("fake_weights")

    monkeypatch.setenv("MODEL_WEIGHTS_PATH", str(fake_weight_file))
    found = find_model_weights_path("efficientnet")
    assert found == str(fake_weight_file)


def test_static_asset_cache_control_headers(client):
    """22. Test static assets have proper Cache-Control headers."""
    # Test index.html / SPA route has no-cache
    resp_spa = client.get("/")
    if resp_spa.status_code == 200:
        assert "no-cache" in resp_spa.headers.get("Cache-Control", "")
    
    # Test hashed JS asset has immutable caching
    assets_dir = os.path.join(os.path.dirname(__file__), "..", "..", "frontend", "dist", "assets")
    if os.path.exists(assets_dir):
        files = os.listdir(assets_dir)
        if files:
            asset_file = files[0]
            resp_asset = client.get(f"/assets/{asset_file}")
            if resp_asset.status_code == 200:
                assert "max-age=31536000" in resp_asset.headers.get("Cache-Control", "")
                assert "immutable" in resp_asset.headers.get("Cache-Control", "")


