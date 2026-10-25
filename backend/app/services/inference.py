import os
import io
import json
import time
import logging
from typing import Dict, Any, List, Optional
from PIL import Image, UnidentifiedImageError

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

try:
    from ml.custom_model import ForensicNet
except ImportError:
    from backend.ml.custom_model import ForensicNet

# Configure logger for inference service
logger = logging.getLogger("inference_service")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

# Singleton module variables
_models: Dict[str, Optional[nn.Module]] = {
    "efficientnet": None,
    "forensicnet": None
}
_model_statuses: Dict[str, bool] = {
    "efficientnet": False,
    "forensicnet": False
}
_model_params: Dict[str, Optional[int]] = {
    "efficientnet": None,
    "forensicnet": None
}
_device: Optional[torch.device] = None

# Validation transformation pipeline
_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

LABEL_MAPPING = {
    0: "AI Generated",
    1: "Real"
}

MODEL_DISPLAY_NAMES = {
    "efficientnet": "EfficientNet-B0",
    "forensicnet": "ForensicNet"
}


def get_device() -> torch.device:
    """Returns CUDA device if available, else CPU."""
    global _device
    if _device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _device


def find_model_weights_path(model_id: str = "efficientnet") -> Optional[str]:
    """Helper to locate candidate model weights paths for a given model."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    backend_dir = os.path.abspath(os.path.join(base_dir, "..", ".."))

    # Check environment variable overrides first
    env_vars = []
    if model_id == "efficientnet":
        env_vars = ["MODEL_WEIGHTS_PATH", "EFFICIENTNET_WEIGHTS_PATH"]
    elif model_id == "forensicnet":
        env_vars = ["FORENSICNET_WEIGHTS_PATH"]

    for var in env_vars:
        custom_path = os.getenv(var)
        if custom_path:
            candidates = [
                custom_path,
                os.path.join(backend_dir, custom_path),
                os.path.abspath(custom_path),
            ]
            for p in candidates:
                if os.path.exists(p) and os.path.isfile(p):
                    return os.path.abspath(p)

    weights_dir = os.getenv("MODEL_WEIGHTS_DIR")

    if model_id == "forensicnet":
        candidates = [
            os.path.join(backend_dir, "ml", "model_weights", "best_forensicnet.pth"),
            os.path.join(backend_dir, "ml", "model_weights", "forensicnet.pth"),
        ]
        if weights_dir:
            candidates.insert(0, os.path.join(weights_dir, "best_forensicnet.pth"))
            candidates.insert(1, os.path.join(weights_dir, "forensicnet.pth"))
    else:  # efficientnet
        candidates = [
            os.path.join(backend_dir, "ml", "model_weights", "best_efficientnet_b0.pth"),
            os.path.join(backend_dir, "ml", "model_weights", "best_model.pth"),
            os.path.join(backend_dir, "ml", "model_weights", "resnet50_finetuned.pth"),
        ]
        if weights_dir:
            candidates.insert(0, os.path.join(weights_dir, "best_efficientnet_b0.pth"))
            candidates.insert(1, os.path.join(weights_dir, "best_model.pth"))

    for cand in candidates:
        if os.path.exists(cand) and os.path.isfile(cand):
            return cand
    return None


def build_model_architecture(model_id: str) -> nn.Module:
    """Instantiates model architecture for a given model identifier."""
    if model_id == "forensicnet":
        return ForensicNet(num_classes=2)
    elif model_id == "efficientnet":
        model_eff = models.efficientnet_b0(weights=None)
        model_eff.classifier = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 2)
        )
        return model_eff
    else:
        raise ValueError(f"Unknown model_id: '{model_id}'")


def format_params_count(count: int) -> str:
    """Helper to format raw parameter counts into human-readable strings (e.g. 4.66M or 1.05M)."""
    if count >= 1_000_000:
        val = count / 1_000_000.0
        return f"{val:.2f}M"
    elif count >= 1_000:
        return f"{count / 1_000.0:.1f}K"
    return str(count)


def initialize_models() -> Dict[str, bool]:
    """
    Loads EfficientNet-B0 and ForensicNet as separate singleton model instances at application startup.
    Computes and stores exact parameter counts dynamically.
    Returns a dictionary mapping model_id to boolean indicating whether it loaded successfully.
    """
    global _models, _device, _model_statuses, _model_params

    device = get_device()

    # 1. Initialize EfficientNet-B0
    if _models["efficientnet"] is None:
        try:
            eff_path = find_model_weights_path("efficientnet")
            model_eff = build_model_architecture("efficientnet")
            if eff_path and os.path.exists(eff_path):
                state_dict = torch.load(eff_path, map_location=device)
                model_eff.load_state_dict(state_dict)
                logger.info(f"Loaded EfficientNet-B0 weights from '{eff_path}'")
            else:
                logger.warning("No EfficientNet-B0 weights found. Running with uninitialized weights.")

            model_eff.to(device)
            model_eff.eval()
            _models["efficientnet"] = model_eff
            _model_statuses["efficientnet"] = True
            
            param_count = sum(p.numel() for p in model_eff.parameters())
            _model_params["efficientnet"] = param_count
            formatted_params = format_params_count(param_count)
            logger.info(f"Loaded EfficientNet-B0 ({formatted_params} params)")

            # Warmup pass to eliminate initial cold-start latency spike
            try:
                with torch.inference_mode():
                    dummy_eff = torch.zeros(1, 3, 224, 224, device=device)
                    _ = model_eff(dummy_eff)
                logger.info(f"Warmed up EfficientNet-B0 on {device.type.upper()}")
            except Exception as we:
                logger.warning(f"EfficientNet-B0 warmup pass failed: {we}")
        except Exception as e:
            logger.error(f"Failed to load EfficientNet-B0 model: {e}")
            _models["efficientnet"] = None
            _model_statuses["efficientnet"] = False

    # 2. Initialize ForensicNet
    if _models["forensicnet"] is None:
        try:
            fn_path = find_model_weights_path("forensicnet")
            model_fn = build_model_architecture("forensicnet")
            if fn_path and os.path.exists(fn_path):
                state_dict = torch.load(fn_path, map_location=device)
                model_fn.load_state_dict(state_dict)
                logger.info(f"Loaded ForensicNet weights from '{fn_path}'")
            else:
                logger.warning("No ForensicNet weights found. Running with uninitialized weights.")

            model_fn.to(device)
            model_fn.eval()
            _models["forensicnet"] = model_fn
            _model_statuses["forensicnet"] = True

            param_count = sum(p.numel() for p in model_fn.parameters())
            _model_params["forensicnet"] = param_count
            formatted_params = format_params_count(param_count)
            logger.info(f"Loaded ForensicNet ({formatted_params} params)")

            # Warmup pass to eliminate initial cold-start latency spike
            try:
                with torch.inference_mode():
                    dummy_fn = torch.zeros(1, 3, 224, 224, device=device)
                    _ = model_fn(dummy_fn)
                logger.info(f"Warmed up ForensicNet on {device.type.upper()}")
            except Exception as we:
                logger.warning(f"ForensicNet warmup pass failed: {we}")
        except Exception as e:
            logger.error(f"Failed to load ForensicNet model: {e}")
            _models["forensicnet"] = None
            _model_statuses["forensicnet"] = False

    # Log overall startup status confirmation
    loaded_names = [MODEL_DISPLAY_NAMES[m] for m, loaded in _model_statuses.items() if loaded]
    if len(loaded_names) == 2:
        logger.info("Confirmation: Both models loaded successfully.")
    elif len(loaded_names) == 1:
        logger.warning(f"Confirmation: Only {loaded_names[0]} loaded successfully.")
    else:
        logger.error("Confirmation: Neither model could be loaded.")

    # Calculate and report combined memory usage
    total_bytes = 0
    for m in _models.values():
        if m is not None:
            total_bytes += sum(p.numel() * p.element_size() for p in m.parameters())
            total_bytes += sum(b.numel() * b.element_size() for b in m.buffers())
    total_mem_mb = total_bytes / (1024.0 * 1024.0)
    mem_type = "VRAM" if device.type == "cuda" else "RAM"
    logger.info(f"Total combined model memory footprint ({mem_type}): {total_mem_mb:.2f} MB")

    return _model_statuses


def initialize_model(model_path: Optional[str] = None) -> nn.Module:
    """Backward-compatible wrapper for initializing models."""
    initialize_models()
    return _models.get("efficientnet")


def is_model_loaded(model_id: Optional[str] = None) -> bool:
    """Helper function to check if a specific model or at least one model is loaded."""
    global _models, _model_statuses
    if model_id is not None:
        normalized_id = model_id.lower().strip()
        return _models.get(normalized_id) is not None and _model_statuses.get(normalized_id, False)
    return any(m is not None for m in _models.values())


def get_model_status() -> Dict[str, bool]:
    """Returns status of all registered models."""
    return dict(_model_statuses)


def get_model_param_count(model_id: str) -> int:
    """Returns exact integer parameter count for a model, computing live if not cached."""
    global _models, _model_params
    normalized_id = model_id.lower().strip()

    if _model_params.get(normalized_id) is not None:
        return _model_params[normalized_id]

    if _models.get(normalized_id) is not None:
        cnt = sum(p.numel() for p in _models[normalized_id].parameters())
        _model_params[normalized_id] = cnt
        return cnt

    # Compute from model architecture
    arch = build_model_architecture(normalized_id)
    cnt = sum(p.numel() for p in arch.parameters())
    _model_params[normalized_id] = cnt
    return cnt


def preprocess_image_bytes(image_bytes: bytes) -> torch.Tensor:
    """Helper to decode and transform image bytes into a batch tensor."""
    if not image_bytes:
        raise ValueError("Invalid or corrupted image data.")

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = _transform(image).unsqueeze(0)
        return tensor
    except (UnidentifiedImageError, OSError, ValueError, Exception) as e:
        logger.error(f"Image decoding/transformation error: {e}")
        raise ValueError("Invalid or corrupted image data.")


def run_forward_pass(model: nn.Module, tensor: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, float]:
    """Helper to perform timed forward pass."""
    tensor = tensor.to(device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start_time = time.perf_counter()

    with torch.inference_mode():
        logits = model(tensor)

    if device.type == "cuda":
        torch.cuda.synchronize()
    end_time = time.perf_counter()

    inference_time_ms = (end_time - start_time) * 1000.0
    return logits, inference_time_ms


def predict(image_bytes: bytes, model_id: str = "efficientnet") -> Dict[str, Any]:
    """Performs inference on raw image bytes using the requested model."""
    normalized_id = (model_id or "efficientnet").lower().strip()
    VALID_MODELS = {"efficientnet", "forensicnet"}

    if normalized_id not in VALID_MODELS:
        raise ValueError(
            f"Invalid model specified: '{model_id}'. Valid options are 'efficientnet' or 'forensicnet'."
        )

    model = _models.get(normalized_id)
    if model is None:
        initialize_models()
        model = _models.get(normalized_id)
        if model is None:
            raise RuntimeError(f"Model '{normalized_id}' is not loaded or unavailable.")

    device = get_device()
    tensor = preprocess_image_bytes(image_bytes)
    logits, inference_time_ms = run_forward_pass(model, tensor, device)

    probs = F.softmax(logits, dim=1)[0]
    pred_idx = int(torch.argmax(probs).item())
    confidence = round(float(probs[pred_idx].item()), 4)
    label = LABEL_MAPPING.get(pred_idx, "Unknown")
    display_name = MODEL_DISPLAY_NAMES.get(normalized_id, normalized_id)

    logger.info(
        f"Prediction [{display_name}] | label='{label}' | confidence={confidence:.4f} | "
        f"inference_time={inference_time_ms:.2f}ms"
    )

    return {
        "label": label,
        "confidence": confidence,
        "inference_time_ms": round(inference_time_ms, 2),
        "model_used": normalized_id,
        "model_display_name": display_name
    }


def predict_batch_tensors(tensors: List[torch.Tensor], model_id: str = "efficientnet") -> List[Dict[str, Any]]:
    """
    Performs batched inference on a list of preprocessed image tensors (each shape [1, C, H, W]).
    Concatenates tensors along batch dimension to execute a single forward pass for high GPU/CPU throughput.
    """
    if not tensors:
        return []

    normalized_id = (model_id or "efficientnet").lower().strip()
    VALID_MODELS = {"efficientnet", "forensicnet"}

    if normalized_id not in VALID_MODELS:
        raise ValueError(
            f"Invalid model specified: '{model_id}'. Valid options are 'efficientnet' or 'forensicnet'."
        )

    model = _models.get(normalized_id)
    if model is None:
        initialize_models()
        model = _models.get(normalized_id)
        if model is None:
            raise RuntimeError(f"Model '{normalized_id}' is not loaded or unavailable.")

    device = get_device()
    batch_tensor = torch.cat(tensors, dim=0)
    logits, total_time_ms = run_forward_pass(model, batch_tensor, device)

    per_item_time_ms = round(total_time_ms / len(tensors), 2)
    display_name = MODEL_DISPLAY_NAMES.get(normalized_id, normalized_id)

    probs = F.softmax(logits, dim=1)
    pred_indices = torch.argmax(probs, dim=1)

    results = []
    for i in range(len(tensors)):
        pred_idx = int(pred_indices[i].item())
        confidence = round(float(probs[i][pred_idx].item()), 4)
        label = LABEL_MAPPING.get(pred_idx, "Unknown")
        results.append({
            "label": label,
            "confidence": confidence,
            "inference_time_ms": per_item_time_ms,
            "model_used": normalized_id,
            "model_display_name": display_name
        })

    logger.info(
        f"Batch forward pass [{display_name}] | batch_size={len(tensors)} | "
        f"total_time={total_time_ms:.2f}ms | per_image={per_item_time_ms:.2f}ms"
    )

    return results


def get_models_info() -> List[Dict[str, Any]]:
    """
    Returns metadata for available models dynamically.
    Parameter counts are computed live from actual model architecture parameters and formatted.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    ml_dir = os.path.abspath(os.path.join(base_dir, "..", "..", "ml"))

    configs = [
        {
            "id": "efficientnet",
            "display_name": "EfficientNet-B0",
            "description": "Transfer-learned CNN, highest accuracy",
            "eval_file": os.path.join(ml_dir, "evaluation_results.json"),
            "fallback_acc": 0.9782,
            "fallback_auc": 0.9979,
            "fallback_gpu_ms": 3.78,
        },
        {
            "id": "forensicnet",
            "display_name": "ForensicNet",
            "description": "Custom tri-stream forensic architecture (noise + frequency + spatial analysis), faster GPU inference",
            "eval_file": os.path.join(ml_dir, "evaluation_results_forensicnet.json"),
            "fallback_acc": 0.9483,
            "fallback_auc": 0.9903,
            "fallback_gpu_ms": 1.78,
        }
    ]

    result = []
    for cfg in configs:
        eval_path = cfg["eval_file"]
        eval_data = {}
        if os.path.exists(eval_path):
            try:
                with open(eval_path, "r", encoding="utf-8") as f:
                    eval_data = json.load(f)
            except Exception as e:
                logger.error(f"Failed to read evaluation results from '{eval_path}': {e}")

        test_acc = eval_data.get("overall_accuracy", cfg["fallback_acc"])
        auc_roc = eval_data.get("auc_roc_score", cfg["fallback_auc"])

        speed_data = eval_data.get("inference_speed_ms", {})
        gpu_ms = speed_data.get("gpu_per_image", cfg["fallback_gpu_ms"])

        # Compute parameter count live from model instance / architecture
        raw_params = get_model_param_count(cfg["id"])
        params_str = format_params_count(raw_params)

        is_avail = is_model_loaded(cfg["id"])

        result.append({
            "id": cfg["id"],
            "display_name": cfg["display_name"],
            "description": cfg["description"],
            "test_accuracy": test_acc,
            "auc_roc": auc_roc,
            "params": params_str,
            "avg_inference_ms_gpu": gpu_ms,
            "available": is_avail
        })

    return result
