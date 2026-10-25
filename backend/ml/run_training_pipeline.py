#!/usr/bin/env python3
"""
Automated ML Training Pipeline Runner for AI Image Detector.
Performs Environment & CUDA Setup, Dataset Validation, Dataset Preparation,
Model Training (using selected device), Visualization, and Evaluation end-to-end.
"""

import os
import sys
import time
import json
import logging
import argparse
import subprocess
import torch

# Ensure local ml directory is in sys.path
ml_dir = os.path.dirname(os.path.abspath(__file__))
if ml_dir not in sys.path:
    sys.path.append(ml_dir)

from dataset import get_dataloaders, resolve_dataset_dir, visualize_samples, prepare_datasets
from train import train_model
from plot_training import plot_training_diagnostics
from evaluate import evaluate_model


def setup_pipeline_logging():
    """Configures structured logging to both console and backend/ml/pipeline.log."""
    log_file = os.path.join(ml_dir, "pipeline.log")

    logger = logging.getLogger("training_pipeline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


logger = setup_pipeline_logging()


def step_1_environment_check_and_cuda_setup():
    logger.info("--- Step 1/7: Environment Check and CUDA Setup ---")
    try:
        # a. Check if torch.cuda.is_available() returns True
        cuda_available = torch.cuda.is_available()

        # b. If CUDA is NOT available, attempt automatic resolution
        if not cuda_available:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            cuda_available = torch.cuda.is_available()

            if not cuda_available:
                try:
                    res = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
                    nvidia_smi_success = (res.returncode == 0)
                except Exception:
                    nvidia_smi_success = False

                if not nvidia_smi_success:
                    logger.warning("No NVIDIA GPU detected. Training will run on CPU. Expect significantly longer training times.")
                    device = torch.device("cpu")
                    vram_gb = None
                else:
                    diagnosis = (
                        "\nCUDA GPU detected by system but PyTorch cannot access it.\n"
                        "Likely cause: PyTorch was installed without CUDA support.\n"
                        "Fix: Uninstall current PyTorch and reinstall the CUDA version:\n"
                        "  pip uninstall torch torchvision\n"
                        "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121\n"
                        "Then re-run this script."
                    )
                    logger.error(diagnosis)
                    print(diagnosis)
                    sys.exit(1)

        # c. If CUDA IS available (either initially or after fix)
        if cuda_available:
            device = torch.device("cuda")
            gpu_name = torch.cuda.get_device_name(0)
            total_bytes = torch.cuda.get_device_properties(0).total_memory
            vram_gb = round(total_bytes / 1e9, 1)

            logger.info(f"GPU Model: {gpu_name}")
            logger.info(f"Total VRAM Available: {vram_gb} GB")

            # Performance optimizations
            os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            logger.info("CUDA optimizations enabled (cudnn.benchmark=True)")

        # d. Final device selection display
        if device.type == "cuda":
            selection_str = f"cuda ({torch.cuda.get_device_name(0)})"
            logger.info("Device selected: cuda — GPU training enabled")
        else:
            selection_str = "cpu"
            logger.info("Device selected: cpu — GPU not available, falling back to CPU")

        # e. Verify PyTorch version and torchvision importability
        logger.info(f"PyTorch Version: {torch.__version__}")
        try:
            import torchvision
            logger.info(f"Torchvision Version: {torchvision.__version__}")
        except ImportError:
            err_msg = (
                "torchvision is not installed or import failed.\n"
                "Fix: Install torchvision using: pip install torchvision"
            )
            logger.error(err_msg)
            print(err_msg)
            sys.exit(1)

        return device, selection_str, vram_gb
    except Exception as e:
        logger.error(f"STEP 1 FAILED: Environment check and CUDA setup failed: {e}")
        sys.exit(1)


def step_2_dataset_validation(data_path: str):
    logger.info("--- Step 2/7: Dataset Validation ---")
    try:
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset root path '{data_path}' does not exist.")

        resolved_path = resolve_dataset_dir(data_path)
        logger.info(f"Resolved dataset directory: '{resolved_path}'")

        subfolders = [
            d for d in os.listdir(resolved_path)
            if os.path.isdir(os.path.join(resolved_path, d)) and not d.startswith(".")
        ]

        if len(subfolders) != 2:
            raise ValueError(
                f"Expected exactly 2 class subfolders in dataset path, but found {len(subfolders)}: {subfolders}"
            )

        logger.info(f"Detected class subfolders: {subfolders}")
        total_dataset_images = 0

        for folder in subfolders:
            folder_path = os.path.join(resolved_path, folder)
            valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
            images = [
                f for f in os.listdir(folder_path)
                if os.path.splitext(f.lower())[1] in valid_exts
            ]
            count = len(images)
            total_dataset_images += count
            logger.info(f"  * Class '{folder}': {count} images")

            if count < 1000:
                raise ValueError(
                    f"Class '{folder}' has only {count} images. Minimum required is 1000 images per class."
                )

        logger.info(f"Total dataset images validated: {total_dataset_images}")
        return resolved_path, total_dataset_images
    except Exception as e:
        logger.error(f"STEP 2 FAILED: Dataset validation failed: {e}")
        sys.exit(1)


def step_3_dataset_preparation(resolved_path: str, batch_size: int = 32, num_workers: int = 2):
    logger.info("--- Step 3/7: Dataset Preparation ---")
    try:
        train_ds, val_ds, test_ds, _ = prepare_datasets(resolved_path, seed=42)
        train_size, val_size, test_size = len(train_ds), len(val_ds), len(test_ds)
        logger.info(f"Split sizes -> Train: {train_size} | Val: {val_size} | Test: {test_size}")

        train_loader, val_loader, test_loader, class_names = get_dataloaders(resolved_path, batch_size=batch_size, num_workers=num_workers)

        sample_grid_path = os.path.join(ml_dir, "sample_grid.png")
        visualize_samples(train_loader, n=8, save_path=sample_grid_path)
        logger.info(f"Sample grid saved to {sample_grid_path}")
    except Exception as e:
        logger.error(f"STEP 3 FAILED: Dataset preparation failed: {e}")
        sys.exit(1)


def step_4_model_training(pipeline_args, device: torch.device):
    logger.info("--- Step 4/7: Model Training ---")
    try:
        logger.info("=== Starting EfficientNet-B0 Training ===")

        pipeline_args.device = device
        train_stats = train_model(pipeline_args, device=device)

        best_weights_path = os.path.join(ml_dir, "model_weights", "best_efficientnet_b0.pth")
        if not os.path.exists(best_weights_path):
            raise FileNotFoundError(f"Expected model weights file '{best_weights_path}' was not found after training.")

        logger.info(f"Training completed successfully!")
        logger.info(f"  * Best Epoch       : {train_stats['best_epoch']}")
        logger.info(f"  * Best Val Accuracy: {train_stats['best_val_acc']*100:.2f}% ({train_stats['best_val_acc']:.4f})")
        logger.info(f"  * Best Val Loss    : {train_stats['best_val_loss']:.4f}")
        logger.info(f"  * Time Taken       : {train_stats['elapsed_time_mins']:.2f} minutes")
        logger.info(f"Confirmed weights saved to '{best_weights_path}'")

        return train_stats, best_weights_path
    except Exception as e:
        logger.error(f"STEP 4 FAILED: Model training failed: {e}")
        sys.exit(1)


def step_5_training_visualization():
    logger.info("--- Step 5/7: Training Visualization ---")
    try:
        log_path = os.path.join(ml_dir, "training_log.json")
        plots_path = os.path.join(ml_dir, "training_plots.png")

        plot_training_diagnostics(log_path=log_path, save_path=plots_path, max_epochs=20)
        logger.info(f"Training plots saved to {plots_path}")
    except Exception as e:
        logger.error(f"STEP 5 FAILED: Training visualization failed: {e}")
        sys.exit(1)


def step_6_model_evaluation(data_path: str, best_weights_path: str, device: torch.device):
    logger.info("--- Step 6/7: Model Evaluation ---")
    try:
        eval_results = evaluate_model(data_path=data_path, model_path=best_weights_path, device=device)

        test_acc = eval_results["overall_accuracy"]
        auc_roc = eval_results["auc_roc_score"]
        macro_prec = eval_results["macro_metrics"]["precision"]
        macro_rec = eval_results["macro_metrics"]["recall"]
        macro_f1 = eval_results["macro_metrics"]["f1_score"]

        speed_info = eval_results["inference_speed_ms"]
        gpu_speed = speed_info.get("gpu_per_image")
        cpu_speed = speed_info.get("cpu_per_image")
        speed_str = f"{gpu_speed:.2f} ms (GPU)" if gpu_speed is not None else f"{cpu_speed:.2f} ms (CPU)"

        logger.info("\n" + "=" * 55)
        logger.info("            EVALUATION RESULTS SUMMARY TABLE")
        logger.info("=" * 55)
        logger.info(f" Overall Test Accuracy : {test_acc*100:.2f}% ({test_acc:.4f})")
        logger.info(f" Macro Precision       : {macro_prec:.4f}")
        logger.info(f" Macro Recall          : {macro_rec:.4f}")
        logger.info(f" Macro F1-Score        : {macro_f1:.4f}")
        logger.info(f" AUC-ROC Score         : {auc_roc:.4f}")
        logger.info(f" Avg Inference Latency : {speed_str}")
        logger.info("=" * 55)

        if test_acc < 0.70:
            logger.warning("WARNING: Low accuracy (< 70%). Consider adjusting hyperparameters or checking dataset quality.")

        return eval_results
    except Exception as e:
        logger.error(f"STEP 6 FAILED: Model evaluation failed: {e}")
        sys.exit(1)


def step_7_final_summary(device_selection: str, vram_gb: float, data_path: str, total_images: int, train_stats: dict, weights_path: str):
    logger.info("--- Step 7/7: Final Summary ---")
    vram_line = f"VRAM available      : {vram_gb:.1f} GB\n" if vram_gb is not None else ""
    
    # Read all evaluation metrics directly from the saved backend/ml/evaluation_results.json file
    eval_json_path = os.path.join(ml_dir, "evaluation_results.json")
    eval_results = {}
    if os.path.exists(eval_json_path):
        try:
            with open(eval_json_path, "r", encoding="utf-8") as f:
                eval_results = json.load(f)
        except Exception as e:
            logger.warning(f"Warning: Could not read evaluation results JSON file: {e}")
    else:
        logger.warning(f"Warning: Evaluation results JSON file not found at {eval_json_path}")

    # Defensive check: if any expected key is missing, print clear warning and print "N/A"
    def get_metric(d, keys, name):
        curr = d
        for k in keys:
            if isinstance(curr, dict) and k in curr:
                curr = curr[k]
            else:
                print(f"Warning: metric '{name}' not found in evaluation results, showing 'N/A' instead")
                return "N/A"
        return curr

    test_acc_raw = get_metric(eval_results, ["overall_accuracy"], "overall_accuracy")
    auc_roc_raw = get_metric(eval_results, ["auc_roc_score"], "auc_roc_score")

    if test_acc_raw != "N/A":
        test_acc_str = f"{test_acc_raw * 100:.2f}%"
    else:
        test_acc_str = "N/A"

    if auc_roc_raw != "N/A":
        auc_roc_str = f"{auc_roc_raw:.4f}"
    else:
        auc_roc_str = "N/A"

    # Best validation accuracy from train_stats or training_log.json
    best_val_acc_str = "N/A"
    if train_stats and "best_val_acc" in train_stats:
        best_val_acc_str = f"{train_stats['best_val_acc'] * 100:.2f}%"
    else:
        training_log_path = os.path.join(ml_dir, "training_log.json")
        if os.path.exists(training_log_path):
            try:
                with open(training_log_path, "r", encoding="utf-8") as f:
                    history = json.load(f)
                if history:
                    max_acc = max(epoch.get("val_acc", 0.0) for epoch in history)
                    best_val_acc_str = f"{max_acc * 100:.2f}%"
            except Exception:
                pass

    # Training time from train_stats or pipeline.log
    training_time = "N/A"
    if train_stats and "elapsed_time_mins" in train_stats:
        training_time = f"{train_stats['elapsed_time_mins']:.1f} minutes"
    else:
        pipeline_log_path = os.path.join(ml_dir, "pipeline.log")
        if os.path.exists(pipeline_log_path):
            try:
                import re
                with open(pipeline_log_path, "r", encoding="utf-8") as f:
                    content = f.read()
                matches = re.findall(r"Time Taken\s*:\s*([\d\.]+)\s*minutes", content)
                if matches:
                    training_time = f"{float(matches[-1]):.1f} minutes"
            except Exception:
                pass

    # Get model weights saved path (relative to workspace root)
    project_root = os.path.dirname(os.path.dirname(ml_dir))
    rel_weights_path = os.path.relpath(weights_path, start=project_root)
    rel_weights_path = rel_weights_path.replace(os.path.sep, "/")

    summary_box = f"""
============================================
TRAINING PIPELINE COMPLETE
============================================
Device used        : {device_selection}
{vram_line}Dataset path        : {data_path}
Total images        : {total_images}
Training time       : {training_time}
Best val accuracy   : {best_val_acc_str}
Test accuracy       : {test_acc_str}
AUC-ROC              : {auc_roc_str}
Model saved to      : {rel_weights_path}
============================================
"""
    logger.info(summary_box)


def main():
    parser = argparse.ArgumentParser(
        description="Automated ML training pipeline runner for AI Image Detector.",
        epilog="""
Usage Examples:
  Start or resume training automatically:
  python backend/ml/run_training_pipeline.py --data_path ./data --unfreeze_at 8

  Force a fresh start (ignore any saved checkpoint):
  python backend/ml/run_training_pipeline.py --data_path ./data --unfreeze_at 8 --fresh_start

  Run with worker benchmark before training:
  python backend/ml/run_training_pipeline.py --data_path ./data --unfreeze_at 8 --benchmark_workers

  Skip training and evaluation, print the final summary directly:
  python backend/ml/run_training_pipeline.py --data_path ./data --summary_only
""",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data_path", type=str, required=True, help="Path to dataset root folder.")
    parser.add_argument("--unfreeze_at", type=int, default=None, help="Epoch at which to unfreeze backbone layers.")
    parser.add_argument("--epochs", type=int, default=20, help="Maximum number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for DataLoaders.")
    parser.add_argument("--lr", type=float, default=0.0005, help="Initial learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for Adam optimizer.")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of workers for DataLoaders.")
    parser.add_argument("--fresh_start", action="store_true", help="Ignore and delete any existing checkpoint and start training from scratch.")
    parser.add_argument("--benchmark_workers", action="store_true", help="Benchmark num_workers before training.")
    parser.add_argument("--amp", type=str, default="True", help="Enable/disable Mixed Precision (AMP) training (True/False).")
    parser.add_argument("--summary_only", action="store_true", help="Skip all training and evaluation, print the final summary directly.")

    args = parser.parse_args()
    args.amp = args.amp.lower() in ('true', '1', 'yes')

    if args.summary_only:
        logger.info("Summary Only mode: skipping training and evaluation.")
        device, device_selection, vram_gb = step_1_environment_check_and_cuda_setup()
        try:
            resolved_path, total_images = step_2_dataset_validation(args.data_path)
        except Exception:
            total_images = "N/A"
        
        best_weights_path = os.path.join(ml_dir, "model_weights", "best_efficientnet_b0.pth")
        step_7_final_summary(device_selection, vram_gb, args.data_path, total_images, None, best_weights_path)
        return

    logger.info("Starting End-to-End ML Training Pipeline...")

    device, device_selection, vram_gb = step_1_environment_check_and_cuda_setup()
    resolved_path, total_images = step_2_dataset_validation(args.data_path)
    step_3_dataset_preparation(resolved_path, batch_size=args.batch_size, num_workers=args.num_workers)
    
    # Run the model training
    train_stats, best_weights_path = step_4_model_training(args, device)
    
    step_5_training_visualization()
    eval_results = step_6_model_evaluation(args.data_path, best_weights_path, device)
    step_7_final_summary(device_selection, vram_gb, args.data_path, total_images, train_stats, best_weights_path)


if __name__ == "__main__":
    main()
