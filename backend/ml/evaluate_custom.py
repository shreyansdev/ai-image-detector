#!/usr/bin/env python3
"""
Evaluation script for trained ForensicNet model on the test dataset.
Computes performance metrics, confusion matrix, ROC curve, and inference latency.
Saves metrics to evaluation_results_forensicnet.json matching the format of evaluation_results.json.
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)

# Ensure local imports work regardless of execution directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from custom_model import ForensicNet
    from dataset import get_dataloaders
except ImportError:
    from backend.ml.custom_model import ForensicNet
    from backend.ml.dataset import get_dataloaders


def get_device() -> torch.device:
    """Select CUDA device if available, otherwise fallback to CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_trained_forensicnet(model_path: str, device: torch.device):
    """
    Loads ForensicNet model weights from model_path or standard fallback locations.
    """
    ml_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        model_path,
        os.path.join(ml_dir, "model_weights", "best_forensicnet.pth"),
        os.path.join(ml_dir, "model_weights", "forensicnet.pth"),
    ]

    resolved_path = None
    for cand in candidates:
        if cand and os.path.exists(cand):
            resolved_path = cand
            break

    if not resolved_path:
        print(f"[ERROR] Could not find ForensicNet model weights file.")
        print(f"Checked paths: {candidates}")
        sys.exit(1)

    print(f"[INFO] Loading ForensicNet model weights from: '{resolved_path}'")
    model = ForensicNet(num_classes=2)
    state_dict = torch.load(resolved_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, resolved_path


def benchmark_inference_speed(model: nn.Module, device: torch.device, input_size=(1, 3, 224, 224), num_iters=100):
    """
    Measures average single-image prediction latency in milliseconds on CPU and GPU.
    """
    dummy_input = torch.randn(*input_size)

    # 1. CPU Latency Benchmark
    model_cpu = model.to("cpu")
    model_cpu.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model_cpu(dummy_input)

    start_cpu = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_iters):
            _ = model_cpu(dummy_input)
    end_cpu = time.perf_counter()
    cpu_ms = ((end_cpu - start_cpu) / num_iters) * 1000.0

    # 2. GPU Latency Benchmark (if available)
    gpu_ms = None
    if torch.cuda.is_available():
        model_gpu = model.to("cuda")
        model_gpu.eval()
        dummy_gpu = dummy_input.to("cuda")

        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model_gpu(dummy_gpu)
        torch.cuda.synchronize()

        start_gpu = time.perf_counter()
        with torch.no_grad():
            for _ in range(num_iters):
                _ = model_gpu(dummy_gpu)
        torch.cuda.synchronize()
        end_gpu = time.perf_counter()
        gpu_ms = ((end_gpu - start_gpu) / num_iters) * 1000.0

        # Move model back to target device
        model.to(device)

    return cpu_ms, gpu_ms


def evaluate_forensicnet(data_path: str = None, model_path: str = None, device=None):
    device = device or get_device()
    ml_dir = os.path.dirname(os.path.abspath(__file__))
    if data_path is None:
        data_path = os.path.join(ml_dir, "data")
    if model_path is None:
        model_path = os.path.join(ml_dir, "model_weights", "best_forensicnet.pth")

    print(f"=== FORENSICNET EVALUATION PIPELINE STARTED ===")
    print(f"Target Device: {device}")

    # Load test dataloader using seed 42 to ensure identical test split
    num_workers = 4 if torch.cuda.is_available() else 2
    _, _, test_loader, class_to_idx = get_dataloaders(data_path, batch_size=32, num_workers=num_workers, seed=42)
    print(f"Class Mapping: {class_to_idx}")

    # Map class indices to target labels ("AI Generated" vs "Real")
    class_names = ["AI Generated", "Real"]
    if "real" in [k.lower() for k in class_to_idx.keys()]:
        sorted_classes = sorted(class_to_idx.items(), key=lambda x: x[1])
        class_names = [
            "Real" if "real" in c[0].lower() else "AI Generated"
            for c in sorted_classes
        ]

    model, loaded_weights_path = load_trained_forensicnet(model_path, device)

    # Collect predictions
    y_true = []
    y_pred = []
    y_probs = []

    print("\n[INFO] Running evaluation inference on test dataset...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_probs.extend(probs[:, 1].cpu().numpy())  # Probability of class 1

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = np.array(y_probs)

    # Metrics computation
    acc = accuracy_score(y_true, y_pred)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    prec_per, rec_per, f1_per, _ = precision_recall_fscore_support(y_true, y_pred, average=None)
    auc_score = roc_auc_score(y_true, y_probs)

    # 1. Confusion Matrix Heatmap
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.7)
    fig.colorbar(cax)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, fontsize=11)
    ax.set_yticklabels(class_names, fontsize=11)
    ax.set_xlabel('Predicted Label', fontsize=12, labelpad=10)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('ForensicNet Confusion Matrix', fontsize=14, fontweight='bold', pad=15)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=14, fontweight='bold',
                    color='white' if cm[i, j] > cm.max() / 2.0 else 'black')

    plt.tight_layout()
    cm_path = os.path.join(ml_dir, "confusion_matrix_forensicnet.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"[INFO] ForensicNet confusion matrix heatmap saved to '{cm_path}'")

    # 2. ROC Curve Plot
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    plt.figure(figsize=(7, 5.5))
    plt.plot(fpr, tpr, color='#9333ea', linewidth=2.5, label=f'ForensicNet ROC (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1.5, label='Random Classifier')
    plt.xlabel('False Positive Rate', fontsize=11)
    plt.ylabel('True Positive Rate', fontsize=11)
    plt.title('ForensicNet ROC Curve', fontsize=13, fontweight='bold', pad=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=10, loc='lower right')
    plt.tight_layout()
    roc_path = os.path.join(ml_dir, "roc_curve_forensicnet.png")
    plt.savefig(roc_path, dpi=150)
    plt.close()
    print(f"[INFO] ForensicNet ROC curve plot saved to '{roc_path}'")

    # 3. Inference Speed Benchmark
    print("[INFO] Benchmarking ForensicNet single-image inference latency...")
    cpu_ms, gpu_ms = benchmark_inference_speed(model, device)

    # 4. JSON Results Export
    results = {
        "model_architecture": "forensicnet",
        "weights_file": os.path.basename(loaded_weights_path),
        "overall_accuracy": round(float(acc), 4),
        "auc_roc_score": round(float(auc_score), 4),
        "macro_metrics": {
            "precision": round(float(prec_macro), 4),
            "recall": round(float(rec_macro), 4),
            "f1_score": round(float(f1_macro), 4)
        },
        "per_class_metrics": {
            class_names[i]: {
                "precision": round(float(prec_per[i]), 4),
                "recall": round(float(rec_per[i]), 4),
                "f1_score": round(float(f1_per[i]), 4)
            } for i in range(len(class_names))
        },
        "inference_speed_ms": {
            "cpu_per_image": round(float(cpu_ms), 2),
            "gpu_per_image": round(float(gpu_ms), 2) if gpu_ms is not None else None
        }
    }

    json_path = os.path.join(ml_dir, "evaluation_results_forensicnet.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"[INFO] Full ForensicNet evaluation metrics saved to '{json_path}'")

    # Print Results Summary Table
    print("\n" + "=" * 60)
    print("           FORENSICNET EVALUATION METRICS SUMMARY")
    print("=" * 60)
    print(f" Model Architecture    : forensicnet")
    print(f" Overall Accuracy      : {acc * 100:.2f}% ({acc:.4f})")
    print(f" AUC-ROC Score         : {auc_score:.4f}")
    print(f" Macro Precision       : {prec_macro:.4f}")
    print(f" Macro Recall          : {rec_macro:.4f}")
    print(f" Macro F1-Score        : {f1_macro:.4f}")
    print("-" * 60)
    print(" Per-Class Metrics:")
    for i, name in enumerate(class_names):
        print(f"   * {name:15s}: Prec={prec_per[i]:.4f} | Rec={rec_per[i]:.4f} | F1={f1_per[i]:.4f}")
    print("-" * 60)
    print(" Inference Latency (per image):")
    print(f"   * CPU Speed         : {cpu_ms:.2f} ms")
    if gpu_ms is not None:
        print(f"   * GPU Speed         : {gpu_ms:.2f} ms")
    else:
        print("   * GPU Speed         : N/A (GPU not active)")
    print("=" * 60)

    # PASS / WARN / FAIL Verdict
    print()
    if acc > 0.85:
        print(f"=== EVALUATION RESULT: PASS === (Accuracy {acc*100:.2f}% > 85%)")
    elif acc >= 0.70:
        print(f"=== EVALUATION RESULT: WARN === (Accuracy {acc*100:.2f}% is between 70% and 85%)")
    else:
        print(f"=== EVALUATION RESULT: FAIL === (Accuracy {acc*100:.2f}% < 70%)")
    print()

    return results


if __name__ == "__main__":
    ml_dir = os.path.dirname(os.path.abspath(__file__))
    default_data = os.path.join(ml_dir, "data")
    default_model = os.path.join(ml_dir, "model_weights", "best_forensicnet.pth")

    parser = argparse.ArgumentParser(description="Evaluate trained ForensicNet AI Image Detector model.")
    parser.add_argument("--data_path", type=str, default=default_data, help="Path to dataset root folder.")
    parser.add_argument("--model_path", type=str, default=default_model, help="Path to trained model weights.")

    args = parser.parse_args()
    evaluate_forensicnet(args.data_path, args.model_path)
