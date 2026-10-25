#!/usr/bin/env python3
"""
train.py - Model training script for fine-tuning EfficientNet-B0 on AI-generated vs Real image classification.
Hardware-Optimized for NVIDIA RTX 3050 GPU, Intel i5-11400H CPU, and 16GB System RAM.
"""

import os
import sys
import time
import json
import argparse
import threading
import subprocess
import torch
import torch.nn as nn
import torch.optim as optim
import psutil
from torchvision import models
from tqdm import tqdm

# Ensure local imports work regardless of execution directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from dataset import get_dataloaders
except ImportError:
    from backend.ml.dataset import get_dataloaders


class GPUUtilMonitor:
    """Monitors GPU compute and memory utilization in a background thread via nvidia-smi."""
    def __init__(self, interval: float = 5.0):
        self.interval = interval
        self.stop_flag = threading.Event()
        self.gpu_utils = []
        self.mem_utils = []
        self.thread = None

    def start(self):
        self.stop_flag.clear()
        self.gpu_utils = []
        self.mem_utils = []
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()

    def _monitor(self):
        while not self.stop_flag.is_set():
            try:
                res = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,utilization.memory", "--format=csv,noheader,nounits"],
                    text=True, stderr=subprocess.DEVNULL
                ).strip()
                gpu_u, mem_u = [float(x) for x in res.split(',')]
                self.gpu_utils.append(gpu_u)
                self.mem_utils.append(mem_u)
            except Exception:
                pass
            time.sleep(self.interval)

    def stop(self):
        self.stop_flag.set()
        if self.thread:
            self.thread.join(timeout=2.0)
        avg_gpu = sum(self.gpu_utils) / len(self.gpu_utils) if self.gpu_utils else 0.0
        avg_mem = sum(self.mem_utils) / len(self.mem_utils) if self.mem_utils else 0.0
        return avg_gpu, avg_mem


def get_device() -> torch.device:
    """Select CUDA device if available, otherwise fallback to CPU."""
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    print("CUDA not available. Falling back to CPU for training.")
    return torch.device("cpu")


def build_efficientnet_b0(num_classes: int = 2) -> nn.Module:
    """
    Instantiates EfficientNet-B0 with pretrained weights and replaces classifier head.
    Freezes all layers except the classifier head.
    """
    try:
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        model = models.efficientnet_b0(weights=weights)
    except AttributeError:
        model = models.efficientnet_b0(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(
        nn.Linear(1280, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, num_classes)
    )

    for param in model.classifier.parameters():
        param.requires_grad = True

    return model


def benchmark_dataloader_workers(data_path: str, batch_size: int, device: torch.device) -> int:
    print("\n--- Running DataLoader num_workers Benchmark ---")
    worker_options = [0, 2, 4, 6, 8]
    pin_memory = (device.type == "cuda")
    
    try:
        from dataset import prepare_datasets
    except ImportError:
        from backend.ml.dataset import prepare_datasets

    try:
        train_ds, _, _, _ = prepare_datasets(data_path)
    except Exception as e:
        print(f"Failed to load dataset for benchmark: {e}")
        return 2

    results = {}
    for w in worker_options:
        print(f"Testing num_workers={w}...")
        loader = torch.utils.data.DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=w, pin_memory=pin_memory
        )
        
        start_t = time.time()
        count = 0
        try:
            for inputs, labels in loader:
                count += 1
                if count >= 20:
                    break
            elapsed = time.time() - start_t
            avg_time = elapsed / count if count > 0 else 0.0
            results[w] = avg_time
            print(f"  num_workers={w}: average batch load time = {avg_time:.4f}s")
        except Exception as e:
            print(f"  num_workers={w}: failed with error: {e}")
            results[w] = float('inf')

    print("\n" + "=" * 45)
    print(f"{'num_workers':<15}{'Avg Batch Load Time (s)':<25}")
    print("-" * 45)
    best_w = 2
    best_t = float('inf')
    for w in worker_options:
        t_val = results.get(w, float('inf'))
        t_str = f"{t_val:.4f}s" if t_val != float('inf') else "N/A"
        print(f"{w:<15}{t_str:<25}")
        if t_val < best_t:
            best_t = t_val
            best_w = w
    print("=" * 45)
    print(f"Optimal value recommended: num_workers={best_w} (average load time {best_t:.4f}s)")
    print("=" * 45 + "\n")
    return best_w


def check_ram_cache_feasibility(dataset_size: int):
    """
    Calculates estimated in-RAM cache size for dataset and evaluates if --cache_in_ram is safe.
    """
    sys_ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    f32_gb = (dataset_size * 224 * 224 * 3 * 4) / (1024 ** 3)
    u8_gb = (dataset_size * 224 * 224 * 3 * 1) / (1024 ** 3)
    safe_ram_gb = sys_ram_gb - 4.0

    print("\n--- In-Memory RAM Cache Feasibility Analysis ---")
    print(f"Total Dataset Images: {dataset_size:,}")
    print(f"Total System RAM:     {sys_ram_gb:.2f} GB")
    print(f"Required RAM Cache (float32 tensors): {f32_gb:.2f} GB")
    print(f"Required RAM Cache (uint8 arrays):    {u8_gb:.2f} GB")
    print(f"Safe RAM Allocation Limit (leaving 4GB for OS): {safe_ram_gb:.2f} GB")
    if u8_gb > safe_ram_gb:
        print(f"--> RAM Cache Status: SKIPPED (Requires {u8_gb:.2f} GB, exceeding safe limit of {safe_ram_gb:.2f} GB).\n")
        return False
    else:
        print(f"--> RAM Cache Status: FEASIBLE\n")
        return True


def train_model(args, device=None):
    start_time = time.time()
    device = device or getattr(args, 'device', None) or get_device()

    # Hardware optimizations: Ampere TF32 Mode & cuDNN benchmark
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        if getattr(args, 'tf32', True):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("Ampere TF32 Acceleration: ENABLED (matmul & cudnn allow_tf32 = True)")

    if getattr(args, 'benchmark_workers', False):
        recommended_workers = benchmark_dataloader_workers(args.data_path, args.batch_size, device)
        print(f"[INFO] Using optimal num_workers={recommended_workers} for training based on benchmark.")
        args.num_workers = recommended_workers

    print(f"\n--- Loading Dataset from '{args.data_path}' ---")
    train_loader, val_loader, test_loader, class_to_idx = get_dataloaders(
        data_dir=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    print(f"Class Mapping: {class_to_idx}")

    total_imgs = len(train_loader.dataset) + len(val_loader.dataset) + len(test_loader.dataset)
    check_ram_cache_feasibility(total_imgs)

    amp_enabled = getattr(args, 'amp', True) and (device.type == "cuda")
    if amp_enabled:
        print("Mixed Precision (AMP): ENABLED — using ~50% less VRAM")
    else:
        print("Mixed Precision (AMP): DISABLED")

    if device.type == "cuda":
        current_alloc = torch.cuda.memory_allocated() / (1024 ** 2)
        current_res = torch.cuda.memory_reserved() / (1024 ** 2)
        total_vram = torch.cuda.get_device_properties(device).total_memory / (1024 ** 2)
        print(f"Current GPU Memory Allocated: {current_alloc:.2f} MB")
        print(f"Current GPU Memory Reserved:  {current_res:.2f} MB")
        print(f"Total Available VRAM:         {total_vram:.2f} MB")
        torch.cuda.reset_peak_memory_stats()

    print("\n--- Initializing EfficientNet-B0 ---")
    model = build_efficientnet_b0(num_classes=2).to(device)

    if getattr(args, 'channels_last', False):
        print("Applying torch.channels_last memory format...")
        model = model.to(memory_format=torch.channels_last)

    if getattr(args, 'compile', False):
        print("Attempting model compilation via torch.compile()...")
        try:
            model = torch.compile(model)
            print("Model successfully compiled with torch.compile().")
        except Exception as e:
            print(f"torch.compile() failed or unsupported ({e}). Falling back to uncompiled eager mode.")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6
    )
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_weights")
    os.makedirs(output_dir, exist_ok=True)
    best_model_path = os.path.join(output_dir, "best_efficientnet_b0.pth")
    checkpoint_path = os.path.join(output_dir, "checkpoint_last.pth")
    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_log.json")

    if getattr(args, 'fresh_start', False) and os.path.exists(checkpoint_path):
        try:
            os.remove(checkpoint_path)
            print("Found --fresh_start flag. Deleted existing checkpoint.")
        except OSError as e:
            print(f"Warning: Could not delete checkpoint: {e}")

    best_val_acc = 0.0
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_no_improve = 0
    training_history = []
    start_epoch = 1
    backbone_frozen = True

    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            completed_epoch = checkpoint['epoch']
            print(f"Found existing checkpoint. Resuming from epoch {completed_epoch}.")
            
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            best_val_acc = checkpoint['best_val_acc']
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            best_epoch = checkpoint['best_epoch']
            epochs_no_improve = checkpoint.get('epochs_no_improve', checkpoint.get('epochs_without_improvement', 0))
            backbone_frozen = checkpoint.get('backbone_frozen', True)
            
            if not backbone_frozen:
                for param in model.parameters():
                    param.requires_grad = True
                print("[INFO] Resumed model state: backbone is UNFREEZED.")
            else:
                for param in model.parameters():
                    param.requires_grad = False
                for param in model.classifier.parameters():
                    param.requires_grad = True
                print("[INFO] Resumed model state: backbone is FROZEN.")

            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            training_history = checkpoint.get('training_history', checkpoint.get('training_log', []))
            start_epoch = completed_epoch + 1
        except Exception as e:
            print(f"Error loading checkpoint: {e}. Starting from scratch.")

    print(f"\nStarting Training (Max Epochs: {args.epochs}, Early Stopping Patience: 5)...")
    print("=" * 75)

    gpu_monitor = GPUUtilMonitor(interval=5.0)
    epoch_times = []

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            epoch_start_time = time.time()

            if device.type == "cuda":
                gpu_monitor.start()

            if args.unfreeze_at is not None and epoch == args.unfreeze_at:
                print(f"\n[INFO] Epoch {epoch}: Unfreezing all model layers for full fine-tuning.")
                for param in model.parameters():
                    param.requires_grad = True
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.1
                print(f"[INFO] Reduced learning rate to {optimizer.param_groups[0]['lr']:.6f}\n")

            current_lr = optimizer.param_groups[0]['lr']

            model.train()
            running_loss = 0.0
            running_corrects = 0
            total_train_samples = 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{args.epochs:02d} [Train]")
            for inputs, labels in pbar:
                if getattr(args, 'channels_last', False):
                    inputs = inputs.to(device, memory_format=torch.channels_last)
                else:
                    inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()
                total_train_samples += inputs.size(0)

                current_loss = running_loss / total_train_samples
                current_acc = running_corrects / total_train_samples
                pbar.set_postfix(loss=f"{current_loss:.4f}", acc=f"{current_acc:.4f}")

            epoch_train_loss = running_loss / total_train_samples
            epoch_train_acc = running_corrects / total_train_samples

            model.eval()
            val_running_loss = 0.0
            val_running_corrects = 0
            total_val_samples = 0

            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch:02d}/{args.epochs:02d} [Val]")
            with torch.no_grad():
                for inputs, labels in pbar_val:
                    if getattr(args, 'channels_last', False):
                        inputs = inputs.to(device, memory_format=torch.channels_last)
                    else:
                        inputs = inputs.to(device)
                    labels = labels.to(device)

                    with torch.cuda.amp.autocast(enabled=amp_enabled):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)
                    val_running_loss += loss.item() * inputs.size(0)
                    val_running_corrects += torch.sum(preds == labels.data).item()
                    total_val_samples += inputs.size(0)

                    current_loss = val_running_loss / total_val_samples
                    current_acc = val_running_corrects / total_val_samples
                    pbar_val.set_postfix(loss=f"{current_loss:.4f}", acc=f"{current_acc:.4f}")

            epoch_val_loss = val_running_loss / total_val_samples
            epoch_val_acc = val_running_corrects / total_val_samples

            scheduler.step(epoch_val_loss)

            avg_gpu_u, avg_mem_u = (0.0, 0.0)
            if device.type == "cuda":
                avg_gpu_u, avg_mem_u = gpu_monitor.stop()

            sys_ram_pct = psutil.virtual_memory().percent
            cpu_per_core = psutil.cpu_percent(percpu=True)

            epoch_stat = {
                "epoch": epoch,
                "train_loss": round(epoch_train_loss, 4),
                "train_acc": round(epoch_train_acc, 4),
                "val_loss": round(epoch_val_loss, 4),
                "val_acc": round(epoch_val_acc, 4),
                "lr": current_lr,
                "gpu_compute_util_pct": round(avg_gpu_u, 1),
                "sys_ram_util_pct": round(sys_ram_pct, 1)
            }
            training_history.append(epoch_stat)

            print(f"Epoch {epoch:02d}/{args.epochs:02d} | "
                  f"Train Loss: {epoch_train_loss:.4f} - Train Acc: {epoch_train_acc:.4f} | "
                  f"Val Loss: {epoch_val_loss:.4f} - Val Acc: {epoch_val_acc:.4f} | "
                  f"LR: {current_lr:.6f}")
            print(f"Hardware Stats | GPU Compute: {avg_gpu_u:.1f}% | GPU Mem Util: {avg_mem_u:.1f}% | System RAM: {sys_ram_pct:.1f}%")

            if device.type == "cuda" and avg_gpu_u < 70.0:
                print(f"[DIAGNOSTIC] GPU compute utilization low ({avg_gpu_u:.1f}%) — likely data-loading bottlenecked, not GPU-bound. Increasing batch_size alone will not help; focus on data pipeline instead.")

            if epoch == 1 and device.type == "cuda":
                peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
                total_vram_mb = torch.cuda.get_device_properties(device).total_memory / (1024 ** 2)
                vram_pct = (peak_memory / total_vram_mb) * 100
                print(f"Peak GPU VRAM usage during Epoch 1: {peak_memory:.2f} MB ({vram_pct:.1f}% of total {total_vram_mb/1024:.1f}GB VRAM)")

            epoch_duration = time.time() - epoch_start_time
            epoch_times.append(epoch_duration)
            epoch_mins, epoch_secs = divmod(epoch_duration, 60)
            print(f"Epoch duration: {int(epoch_mins)}m {int(epoch_secs):02d}s")
            
            remaining_epochs = args.epochs - epoch
            if remaining_epochs > 0:
                last_n_times = epoch_times[-3:]
                avg_epoch_time = sum(last_n_times) / len(last_n_times)
                estimated_remaining_time = avg_epoch_time * remaining_epochs
                eta_hours, eta_rem = divmod(estimated_remaining_time, 3600)
                eta_mins, _ = divmod(eta_rem, 60)
                print(f"Estimated time remaining: {int(eta_hours)} hours {int(eta_mins)} minutes")

            if epoch_val_acc > best_val_acc:
                best_val_acc = epoch_val_acc
                best_val_loss = epoch_val_loss
                best_epoch = epoch
                epochs_no_improve = 0
                torch.save(model.state_dict(), best_model_path)
                print(f"  --> Saved new best model weights (Val Acc: {best_val_acc:.4f})")
            else:
                epochs_no_improve += 1

            with open(log_path, "w") as f:
                json.dump(training_history, f, indent=4)

            backbone_frozen = not next(model.parameters()).requires_grad
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
                'best_val_loss': best_val_loss,
                'best_epoch': best_epoch,
                'epochs_no_improve': epochs_no_improve,
                'epochs_without_improvement': epochs_no_improve,
                'backbone_frozen': backbone_frozen,
                'scaler_state_dict': scaler.state_dict(),
                'training_history': training_history
            }
            torch.save(checkpoint, checkpoint_path)

            if epochs_no_improve >= 5:
                print(f"\n[INFO] Early stopping triggered! Validation loss did not improve for 5 consecutive epochs.")
                break

    except KeyboardInterrupt:
        if device.type == "cuda":
            gpu_monitor.stop()
        print("\nTraining paused. Checkpoint saved. Resume anytime by re-running the same command.")
        completed_epoch = (epoch - 1) if 'epoch' in locals() else (start_epoch - 1)
        if 'model' in locals():
            backbone_frozen = not next(model.parameters()).requires_grad
            checkpoint = {
                'epoch': completed_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
                'best_val_loss': best_val_loss,
                'best_epoch': best_epoch,
                'epochs_no_improve': epochs_no_improve,
                'epochs_without_improvement': epochs_no_improve,
                'backbone_frozen': backbone_frozen,
                'scaler_state_dict': scaler.state_dict(),
                'training_history': training_history
            }
            torch.save(checkpoint, checkpoint_path)
        sys.exit(0)

    if os.path.exists(checkpoint_path):
        try:
            os.remove(checkpoint_path)
        except OSError:
            pass

    elapsed_time = time.time() - start_time
    mins, secs = divmod(elapsed_time, 60)

    print("=" * 75)
    print("\n=== TRAINING COMPLETED ===")
    print(f"Best Epoch: {best_epoch}")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Total Training Time: {int(mins)}m {int(secs)}s")
    print(f"Best Weights Saved To: {best_model_path}")
    print(f"Training Log Saved To: {log_path}\n")

    return {
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "best_val_loss": best_val_loss,
        "elapsed_time_mins": elapsed_time / 60.0
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train EfficientNet-B0 for AI Image Detection.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data_path", type=str, default="data", help="Path to dataset root folder.")
    parser.add_argument("--epochs", type=int, default=20, help="Maximum number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for DataLoaders.")
    parser.add_argument("--lr", type=float, default=0.0005, help="Initial learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for Adam optimizer.")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of workers for DataLoaders.")
    parser.add_argument("--unfreeze_at", type=int, default=None, help="Epoch at which to unfreeze all layers for fine-tuning.")
    parser.add_argument("--fresh_start", action="store_true", help="Ignore and delete any existing checkpoint and start training from scratch.")
    parser.add_argument("--benchmark_workers", action="store_true", help="Benchmark num_workers before training.")
    parser.add_argument("--amp", type=str, default="True", help="Enable/disable Mixed Precision (AMP) training (True/False).")
    parser.add_argument("--tf32", type=str, default="True", help="Enable/disable TF32 mode on Ampere GPU (True/False).")
    parser.add_argument("--compile", action="store_true", help="Enable PyTorch 2.0+ torch.compile(model).")
    parser.add_argument("--channels_last", action="store_true", help="Apply torch.channels_last memory format.")

    args = parser.parse_args()
    args.amp = args.amp.lower() in ('true', '1', 'yes')
    args.tf32 = args.tf32.lower() in ('true', '1', 'yes')
    train_model(args)

