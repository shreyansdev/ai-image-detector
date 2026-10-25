#!/usr/bin/env python3
"""
End-to-End Integration Verification Checklist Script.
Automates check execution against the running FastAPI service at http://localhost:8000.
"""

import sys
import io
import json
import urllib.request
import urllib.parse
import urllib.error
from PIL import Image

BASE_URL = "http://localhost:8000"


def create_synthetic_image_bytes() -> bytes:
    """Generates a 224x224 synthetic PNG image."""
    img = Image.new("RGB", (224, 224), color="blue")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def create_multipart_form_data(field_name: str, filename: str, content: bytes, content_type: str):
    """Encodes multipart form-data payload using standard library."""
    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    body = io.BytesIO()
    body.write(f"--{boundary}\r\n".encode("utf-8"))
    body.write(f'Content-Disposition: form-data; name="{field_name}"; filename="{filename}"\r\n'.encode("utf-8"))
    body.write(f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"))
    body.write(content)
    body.write(f"\r\n--{boundary}--\r\n".encode("utf-8"))
    return body.getvalue(), f"multipart/form-data; boundary={boundary}"


def run_verification():
    print("=" * 65)
    print("      END-TO-END INTEGRATION VERIFICATION CHECKLIST")
    print("=" * 65)

    passed_checks = 0
    total_checks = 6

    # 1. Check server health
    check1_pass = False
    health_data = {}
    try:
        req = urllib.request.Request(f"{BASE_URL}/health", headers={"Origin": "http://localhost:3000"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status == 200:
                health_data = json.loads(resp.read().decode())
                check1_pass = True
                print(f"[Check 1] Server Health Check          : PASS ✓ (HTTP 200 OK)")
            else:
                print(f"[Check 1] Server Health Check          : FAIL ✗ (HTTP {resp.status})")
    except Exception as e:
        print(f"[Check 1] Server Health Check          : FAIL ✗ (Could not connect to {BASE_URL}: {e})")
        print("\n[ERROR] Server is not running. Please start the backend service:")
        print("        uvicorn app.main:app --reload\n")
        sys.exit(1)

    if check1_pass:
        passed_checks += 1

    # 2. Verify model_loaded is true
    if check1_pass and health_data.get("model_loaded") is True:
        print(f"[Check 2] Model Loaded Status          : PASS ✓ (model_loaded=True)")
        passed_checks += 1
    else:
        print(f"[Check 2] Model Loaded Status          : FAIL ✗ (model_loaded={health_data.get('model_loaded')})")

    # 3. Send synthetic image to /api/v1/predict
    img_bytes = create_synthetic_image_bytes()
    body_data, content_type_header = create_multipart_form_data("image", "synthetic.png", img_bytes, "image/png")

    try:
        req = urllib.request.Request(
            f"{BASE_URL}/api/v1/predict",
            data=body_data,
            headers={"Content-Type": content_type_header, "Origin": "http://localhost:3000"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            resp_data = json.loads(resp.read().decode())
            status_ok = (resp.status == 200)
            has_keys = all(k in resp_data for k in ["label", "confidence", "inference_time_ms"])
            valid_label = resp_data.get("label") in ["AI Generated", "Real"]
            valid_conf = isinstance(resp_data.get("confidence"), (float, int)) and 0.0 <= resp_data.get("confidence") <= 1.0
            valid_time = isinstance(resp_data.get("inference_time_ms"), (float, int)) and resp_data.get("inference_time_ms") > 0

            if status_ok and has_keys and valid_label and valid_conf and valid_time:
                print(f"[Check 3] Prediction Inference Pipeline: PASS ✓ (Label: {resp_data['label']}, Conf: {resp_data['confidence']})")
                passed_checks += 1
            else:
                print(f"[Check 3] Prediction Inference Pipeline: FAIL ✗ (Response data invalid: {resp_data})")
    except Exception as e:
        print(f"[Check 3] Prediction Inference Pipeline: FAIL ✗ (Request failed: {e})")

    # 4. Test CORS headers
    try:
        req = urllib.request.Request(f"{BASE_URL}/health", headers={"Origin": "http://localhost:3000"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            cors_header = resp.headers.get("Access-Control-Allow-Origin")
            if cors_header == "http://localhost:3000":
                print(f"[Check 4] CORS Headers Verification    : PASS ✓ (Access-Control-Allow-Origin: {cors_header})")
                passed_checks += 1
            else:
                print(f"[Check 4] CORS Headers Verification    : FAIL ✗ (Header missing or unexpected: {cors_header})")
    except Exception as e:
        print(f"[Check 4] CORS Headers Verification    : FAIL ✗ ({e})")

    # 5. Test Security headers
    try:
        req = urllib.request.Request(f"{BASE_URL}/health")
        with urllib.request.urlopen(req, timeout=5) as resp:
            nosniff = resp.headers.get("X-Content-Type-Options") == "nosniff"
            frame_deny = resp.headers.get("X-Frame-Options") == "DENY"
            if nosniff and frame_deny:
                print(f"[Check 5] Security Headers Presence   : PASS ✓ (nosniff & DENY confirmed)")
                passed_checks += 1
            else:
                print(f"[Check 5] Security Headers Presence   : FAIL ✗ (nosniff={nosniff}, DENY={frame_deny})")
    except Exception as e:
        print(f"[Check 5] Security Headers Presence   : FAIL ✗ ({e})")

    # 6. Test Rate Limiting
    rate_limit_hit = False
    for i in range(35):
        try:
            req = urllib.request.Request(
                f"{BASE_URL}/api/v1/predict",
                data=body_data,
                headers={"Content-Type": content_type_header, "Origin": "http://localhost:3000"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                pass
        except urllib.error.HTTPError as he:
            if he.code == 429:
                rate_limit_hit = True
                break
        except Exception:
            pass

    if rate_limit_hit:
        print(f"[Check 6] Rate Limiting Enforcement    : PASS ✓ (HTTP 429 Too Many Requests triggered)")
        passed_checks += 1
    else:
        print(f"[Check 6] Rate Limiting Enforcement    : FAIL ✗ (HTTP 429 was not triggered)")

    print("=" * 65)
    print(f" FINAL SUMMARY: {passed_checks}/{total_checks} Checks Passed.")
    print("=" * 65)

    print("\n📋 MANUAL QA CHECKLIST FOR DEVELOPERS:")
    print("  [ ] 1. Open the React frontend (http://localhost:3000) and upload a real photograph.")
    print("         -> Verify result displays 'AUTHENTIC IMAGE' / 'Real' in bold GREEN.")
    print("  [ ] 2. Upload an AI-generated image (e.g. Midjourney or DALL-E).")
    print("         -> Verify result displays 'AI GENERATED' in bold RED.")
    print("  [ ] 3. Try uploading a PDF or non-image document.")
    print("         -> Confirm it is rejected with an error banner: 'Invalid file type.'")
    print("  [ ] 4. Try uploading a 15MB image file.")
    print("         -> Confirm it is rejected with an error banner: 'File too large. Maximum size is 10MB.'")
    print()


if __name__ == "__main__":
    run_verification()
