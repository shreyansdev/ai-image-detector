#!/usr/bin/env python3
"""
train_custom.py - Hardware-Optimized Training Pipeline for ForensicNet from scratch.

Hardware Optimizations Integrated:
-----------------------------------
- GPU Utilization Monitoring: Periodically queries nvidia-smi in a background thread.
  If GPU compute utilization drops below 70%, displays bottleneck diagnostic warnings.
- Ampere TF32 Acceleration: Enables torch.backends.cuda.matmul.allow_tf32 and cudnn.allow_tf32.
- Data Loading Pipeline: persistent_workers=True, prefetch_factor=4, pin_memory=True.
- PyTorch 2.0+ torch.compile Support: Optional --compile flag with automatic fallback.
- Channels Last Memory Format: Optional --channels_last flag.
- Batch Size Finder: Optional --find_batch_size flag to benchmark throughput across batch sizes.
- System RAM & CPU Monitoring: Logs peak RAM & per-core CPU usage via psutil.
"""

import os
import sys
import time
import json
import copy
import math
import argparse
import threading
import subprocess
import torch
import torch.nn as nn
import torch.optim as optim
import psutil
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms
from tqdm import tqdm

# Ensure local imports work regardless of execution directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from custom_model import ForensicNet, BayarConv2d
    from dataset import resolve_dataset_dir, TransformedSubset, IMAGENET_MEAN, IMAGENET_STD
except ImportError:
    from backend.ml.custom_model import ForensicNet, BayarConv2d
    from backend.ml.dataset import resolve_dataset_dir, TransformedSubset, IMAGENET_MEAN, IMAGENET_STD


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
    """Select CUDA device if available, otherwise CPU."""
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    print("CUDA not available. Falling back to CPU for training.")
    return torch.device("cpu")


def get_forensic_transforms():
    """
    Returns gentle training and validation transformation pipelines designed to
    preserve delicate forensic noise residuals and frequency domain artifacts.
    """
    train_transform = transforms.Compose([
        transforms.Resize(240, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=5, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    val_transform = transforms.Compose([
        transforms.Resize(240, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    return train_transform, val_transform


def get_forensic_dataloaders(data_dir: str, batch_size: int = 32, num_workers: int = 6, seed: int = 42):
    """
    Creates DataLoaders for train, validation, and test splits with gentle forensic transforms.
    Includes persistent_workers, prefetch_factor, and pin_memory optimizations.
    """
    resolved_dir = resolve_dataset_dir(data_dir)
    raw_dataset = datasets.ImageFolder(root=resolved_dir, transform=None)
    
    if len(raw_dataset) == 0:
        raise ValueError(f"No valid images found in dataset directory '{resolved_dir}'.")

    total_size = len(raw_dataset)
    train_size = int(0.70 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    generator = torch.Generator().manual_seed(seed)
    train_sub, val_sub, test_sub = random_split(
        raw_dataset, [train_size, val_size, test_size], generator=generator
    )

    train_transform, val_transform = get_forensic_transforms()

    train_dataset = TransformedSubset(train_sub, transform=train_transform)
    val_dataset = TransformedSubset(val_sub, transform=val_transform)
    test_dataset = TransformedSubset(test_sub, transform=val_transform)

    pin_memory = torch.cuda.is_available()
    persistent = (num_workers > 0)
    prefetch = 4 if num_workers > 0 else None

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=persistent, prefetch_factor=prefetch
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=persistent, prefetch_factor=prefetch
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=persistent, prefetch_factor=prefetch
    )

    return train_loader, val_loader, test_loader, raw_dataset.class_to_idx


def init_forensicnet_weights(model: nn.Module):
    """
    Applies He/Kaiming Normal initialization to standard Conv2d and Linear layers.
    Preserves and enforces the Bayar constraint on BayarConv2d layers.
    """
    for name, module in model.named_modules():
        if isinstance(module, BayarConv2d) or "bayar_conv" in name:
            continue
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0.0)

    if hasattr(model, 'constrain_bayar_weights'):
        model.constrain_bayar_weights()


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


def run_batch_size_finder(data_path, device):
    print("\n--- Automatic Batch Size Finder ---")
    candidates = [32, 64, 96, 128, 256]
    resolved_dir = resolve_dataset_dir(data_path)
    raw_dataset = datasets.ImageFolder(root=resolved_dir, transform=None)
    _, val_tf = get_forensic_transforms()
    train_sub, _ = random_split(raw_dataset, [min(2000, len(raw_dataset)), len(raw_dataset) - min(2000, len(raw_dataset))], generator=torch.Generator().manual_seed(42))
    dataset = TransformedSubset(train_sub, transform=val_tf)

    best_bs = 32
    max_throughput = 0.0

    for bs in candidates:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        loader = DataLoader(dataset, batch_size=bs, shuffle=False, num_workers=2, pin_memory=True)
        model = ForensicNet(num_classes=2).to(device)
        model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        try:
            start_t = time.time()
            count = 0
            for i, (inputs, labels) in enumerate(loader):
                if i >= 10: break
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                out = model(inputs)
                loss = criterion(out, labels)
                loss.backward()
                optimizer.step()
                count += inputs.size(0)
            elapsed = time.time() - start_t
            tp = count / elapsed
            vram = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            print(f"  batch_size={bs:3d}: PASS | Throughput: {tp:6.2f} img/sec | Peak VRAM: {vram:6.1f} MB")
            if tp > max_throughput:
                max_throughput = tp
                best_bs = bs
        except torch.cuda.OutOfMemoryError:
            print(f"  batch_size={bs:3d}: CUDA OOM!")
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  batch_size={bs:3d}: Failed ({e})")

    print(f"\nLargest stable batch size found: {best_bs} (throughput: {max_throughput:.2f} img/sec)")
    print(f"[RECOMMENDATION] You may pass '--batch_size {best_bs}' to train with optimal throughput.\n")
    return best_bs


def train_forensicnet(args):
    start_time = time.time()
    device = get_device()

    # Hardware optimizations: Ampere TF32 Mode & cuDNN benchmark
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        if getattr(args, 'tf32', True):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("Ampere TF32 Acceleration: ENABLED (matmul & cudnn allow_tf32 = True)")

    if getattr(args, 'find_batch_size', False):
        run_batch_size_finder(args.data_path, device)
        if getattr(args, 'exit_after_find_bs', False):
            sys.exit(0)

    print(f"\n--- Loading Forensic Dataset from '{args.data_path}' ---")
    train_loader, val_loader, test_loader, class_to_idx = get_forensic_dataloaders(
        data_dir=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    print(f"Class Mapping: {class_to_idx}")
    print(f"Splits: Train = {len(train_loader.dataset)}, Val = {len(val_loader.dataset)}, Test = {len(test_loader.dataset)}")

    # Check RAM Cache Feasibility
    total_imgs = len(train_loader.dataset) + len(val_loader.dataset) + len(test_loader.dataset)
    check_ram_cache_feasibility(total_imgs)

    # Mixed precision check and printed log
    amp_enabled = args.amp and (device.type == "cuda")
    if amp_enabled:
        print("Mixed Precision (AMP): ENABLED — using ~50% less VRAM")
    else:
        print("Mixed Precision (AMP): DISABLED")

    # VRAM stats reporting
    if device.type == "cuda":
        current_alloc = torch.cuda.memory_allocated() / (1024 ** 2)
        current_res = torch.cuda.memory_reserved() / (1024 ** 2)
        total_vram = torch.cuda.get_device_properties(device).total_memory / (1024 ** 2)
        print(f"Current GPU Memory Allocated: {current_alloc:.2f} MB")
        print(f"Current GPU Memory Reserved:  {current_res:.2f} MB")
        print(f"Total Available VRAM:         {total_vram:.2f} MB")
        torch.cuda.reset_peak_memory_stats()

    print("\n--- Initializing ForensicNet Model from Scratch ---")
    model = ForensicNet(num_classes=2).to(device)
    init_forensicnet_weights(model)

    # Optional memory format optimization
    if getattr(args, 'channels_last', False):
        print("Applying torch.channels_last memory format...")
        model = model.to(memory_format=torch.channels_last)

    # Optional torch.compile PyTorch 2.0+ optimization
    if getattr(args, 'compile', False):
        print("Attempting model compilation via torch.compile()...")
        try:
            model = torch.compile(model)
            print("Model successfully compiled with torch.compile().")
        except Exception as e:
            print(f"torch.compile() failed or unsupported ({e}). Falling back to uncompiled eager mode.")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    # Output file paths
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_weights")
    os.makedirs(output_dir, exist_ok=True)
    best_model_path = os.path.join(output_dir, "best_forensicnet.pth")
    checkpoint_path = os.path.join(output_dir, "checkpoint_forensicnet_last.pth")
    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_log_forensicnet.json")

    # Handle fresh start flag
    if args.fresh_start and os.path.exists(checkpoint_path):
        try:
            os.remove(checkpoint_path)
            print("Found --fresh_start flag. Deleted existing ForensicNet checkpoint.")
        except OSError as e:
            print(f"Warning: Could not delete checkpoint: {e}")

    best_val_acc = 0.0
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_no_improve = 0
    training_history = []
    start_epoch = 1

    last_known_good_state = copy.deepcopy(model.state_dict())
    nan_count = 0

    # Resume from existing checkpoint if present
    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            completed_epoch = checkpoint['epoch']
            print(f"Found existing ForensicNet checkpoint. Resuming from epoch {completed_epoch + 1}.")
            
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            best_val_acc = checkpoint.get('best_val_acc', 0.0)
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            best_epoch = checkpoint.get('best_epoch', 0)
            epochs_no_improve = checkpoint.get('epochs_no_improve', 0)
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            training_history = checkpoint.get('training_history', [])
            start_epoch = completed_epoch + 1
            
            if hasattr(model, 'constrain_bayar_weights'):
                model.constrain_bayar_weights()
            last_known_good_state = copy.deepcopy(model.state_dict())
        except Exception as e:
            print(f"Error loading ForensicNet checkpoint: {e}. Starting from scratch.")

    if getattr(args, 'smoke_test', False):
        args.epochs = 3
        print("\n--- Smoke Test Mode Enabled: Restricting training to 3 epochs ---")

    print(f"\nStarting ForensicNet Training (Max Epochs: {args.epochs}, Early Stopping Patience: 8)...")
    print("=" * 75)

    gpu_monitor = GPUUtilMonitor(interval=5.0)
    epoch_times = []

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            epoch_start_time = time.time()
            current_lr = optimizer.param_groups[0]['lr']

            if device.type == "cuda":
                gpu_monitor.start()

            # --- Training Phase ---
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
                if amp_enabled:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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

            # --- Validation Phase ---
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

            scheduler.step()

            # GPU & RAM monitoring metrics
            avg_gpu_u, avg_mem_u = (0.0, 0.0)
            if device.type == "cuda":
                avg_gpu_u, avg_mem_u = gpu_monitor.stop()

            sys_ram_pct = psutil.virtual_memory().percent
            cpu_per_core = psutil.cpu_percent(percpu=True)

            print(f"Epoch {epoch:02d}/{args.epochs:02d} | "
                  f"Train Loss: {epoch_train_loss:.4f} - Train Acc: {epoch_train_acc:.4f} | "
                  f"Val Loss: {epoch_val_loss:.4f} - Val Acc: {epoch_val_acc:.4f} | "
                  f"LR: {current_lr:.6f}")
            print(f"Hardware Stats | GPU Compute: {avg_gpu_u:.1f}% | GPU Mem Util: {avg_mem_u:.1f}% | System RAM: {sys_ram_pct:.1f}%")

            # --- NaN Detection and Automatic Safe Rollback ---
            is_nan_or_inf = math.isnan(epoch_train_loss) or math.isinf(epoch_train_loss) or math.isnan(epoch_val_loss) or math.isinf(epoch_val_loss)
            if is_nan_or_inf:
                nan_count += 1
                print(f"\n[WARNING] NaN detected at epoch {epoch}. Skipping checkpoint save. Consider reducing learning rate.")
                
                if nan_count >= 3:
                    print(f"\n[ERROR] NaN detected {nan_count} times total across the run. Aborting training.")
                    if getattr(args, 'smoke_test', False):
                        print("\n" + "=" * 75)
                        print("SMOKE TEST RESULT: FAIL (NaN detected during 3-epoch smoke test)")
                        print("=" * 75)
                    sys.exit(1)

                # Automatically reduce learning rate by 50%
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.5
                new_lr = optimizer.param_groups[0]['lr']
                print(f"--> Automatically reduced learning rate by 50% to {new_lr:.6f}")

                # Reload model weights from last known-good checkpoint
                if os.path.exists(checkpoint_path):
                    try:
                        ckpt = torch.load(checkpoint_path, map_location=device)
                        model.load_state_dict(ckpt['model_state_dict'])
                        print(f"--> Reloaded model weights from last known-good checkpoint (epoch {ckpt['epoch']}).")
                    except Exception:
                        if last_known_good_state is not None:
                            model.load_state_dict(last_known_good_state)
                            print("--> Reloaded model weights from last known-good in-memory state.")
                elif last_known_good_state is not None:
                    model.load_state_dict(last_known_good_state)
                    print("--> Reloaded model weights from last known-good in-memory state.")

                continue
            else:
                last_known_good_state = copy.deepcopy(model.state_dict())

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

            # GPU Utilization Diagnostic Warning
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
                avg_epoch_time = sum(epoch_times[-3:]) / len(epoch_times[-3:])
                estimated_remaining_time = avg_epoch_time * remaining_epochs
                eta_hours, eta_rem = divmod(estimated_remaining_time, 3600)
                eta_mins, _ = divmod(eta_rem, 60)
                print(f"Estimated time remaining: {int(eta_hours)} hours {int(eta_mins)} minutes")

            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                best_val_acc = epoch_val_acc
                best_epoch = epoch
                epochs_no_improve = 0
                torch.save(model.state_dict(), best_model_path)
                print(f"  --> Saved new best model weights (Val Loss: {best_val_loss:.4f}, Val Acc: {best_val_acc:.4f})")
            else:
                epochs_no_improve += 1

            with open(log_path, "w") as f:
                json.dump(training_history, f, indent=4)

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
                'best_val_loss': best_val_loss,
                'best_epoch': best_epoch,
                'epochs_no_improve': epochs_no_improve,
                'scaler_state_dict': scaler.state_dict(),
                'training_history': training_history
            }
            torch.save(checkpoint, checkpoint_path)

            if epochs_no_improve >= 8:
                print(f"\n[INFO] Early stopping triggered! Validation loss did not improve for 8 consecutive epochs.")
                break

    except KeyboardInterrupt:
        if device.type == "cuda":
            gpu_monitor.stop()
        print("\nTraining paused by user. ForensicNet checkpoint saved. Resume anytime by re-running the same command.")
        completed_epoch = (epoch - 1) if 'epoch' in locals() else (start_epoch - 1)
        if 'model' in locals():
            checkpoint = {
                'epoch': completed_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
                'best_val_loss': best_val_loss,
                'best_epoch': best_epoch,
                'epochs_no_improve': epochs_no_improve,
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
    print("┌" + "─" * 73 + "┐")
    print("│{:^73}│".format("FORENSICNET TRAINING COMPLETE"))
    print("├" + "─" * 73 + "┤")
    print("│ {:<25}: {:<45} │".format("Best Epoch", f"{best_epoch}/{args.epochs}"))
    print("│ {:<25}: {:<45} │".format("Best Validation Loss", f"{best_val_loss:.4f}"))
    print("│ {:<25}: {:<45} │".format("Best Validation Accuracy", f"{best_val_acc:.4f} ({best_val_acc * 100:.2f}%)"))
    print("│ {:<25}: {:<45} │".format("Total Training Time", f"{int(mins)}m {int(secs):02d}s"))
    print("│ {:<25}: {:<45} │".format("Best Model Path", f"{best_model_path}"))
    print("│ {:<25}: {:<45} │".format("Training Log Path", f"{log_path}"))
    print("└" + "─" * 73 + "┘")
    print("=" * 75 + "\n")

    if getattr(args, 'smoke_test', False):
        print("=" * 75)
        if nan_count == 0:
            print("SMOKE TEST RESULT: PASS (3-epoch smoke test completed with no NaN detected)")
        else:
            print("SMOKE TEST RESULT: FAIL (NaN was detected during smoke test)")
        print("=" * 75 + "\n")

    return {
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "best_val_loss": best_val_loss,
        "elapsed_time_mins": elapsed_time / 60.0
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train ForensicNet from scratch on merged dataset for AI image detection.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data_path", type=str, default="backend/ml/data/merged_dataset", help="Path to merged dataset root folder.")
    parser.add_argument("--epochs", type=int, default=40, help="Maximum number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for DataLoaders.")
    parser.add_argument("--lr", type=float, default=0.0005, help="Initial learning rate for Adam.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for Adam optimizer.")
    parser.add_argument("--num_workers", type=int, default=6, help="Number of worker processes for DataLoaders.")
    parser.add_argument("--fresh_start", action="store_true", help="Ignore and delete any existing checkpoint to start training from scratch.")
    parser.add_argument("--amp", type=str, default="True", help="Enable/disable Mixed Precision (AMP) training (True/False).")
    parser.add_argument("--no_amp", action="store_true", help="Disable Mixed Precision (AMP) training and train in full FP32.")
    parser.add_argument("--tf32", type=str, default="True", help="Enable/disable TF32 mode on Ampere GPU (True/False).")
    parser.add_argument("--compile", action="store_true", help="Enable PyTorch 2.0+ torch.compile(model).")
    parser.add_argument("--channels_last", action="store_true", help="Apply torch.channels_last memory format.")
    parser.add_argument("--find_batch_size", action="store_true", help="Run automatic batch size finder to benchmark throughput across batch sizes.")
    parser.add_argument("--smoke_test", action="store_true", help="Run a 3-epoch smoke test to verify training stability.")

    args = parser.parse_args()
    if args.no_amp:
        args.amp = False
    else:
        args.amp = args.amp.lower() in ('true', '1', 'yes')
    args.tf32 = args.tf32.lower() in ('true', '1', 'yes')
    train_forensicnet(args)

