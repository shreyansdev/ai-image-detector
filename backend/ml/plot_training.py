#!/usr/bin/env python3
"""
Diagnostic plotting script for model training metrics.
Reads training_log.json produced by train.py and generates training diagnostic charts.
"""

import os
import sys
import json
import argparse
import matplotlib.pyplot as plt


def plot_training_diagnostics(log_path: str, save_path: str, max_epochs: int = 20):
    """
    Reads training metrics from log_path and generates a 2-panel diagnostic figure.
    """
    if not os.path.exists(log_path):
        print(f"[ERROR] Training log file '{log_path}' does not exist.")
        sys.exit(1)

    with open(log_path, "r") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print(f"[ERROR] Failed to parse JSON from '{log_path}'. File may be corrupt.")
            sys.exit(1)

    if not data:
        print(f"[ERROR] Training log file '{log_path}' is empty.")
        sys.exit(1)

    epochs = [item["epoch"] for item in data]
    train_loss = [item["train_loss"] for item in data]
    val_loss = [item["val_loss"] for item in data]
    train_acc = [item["train_acc"] for item in data]
    val_acc = [item["val_acc"] for item in data]

    # Identify best epoch based on validation accuracy (tie-break with validation loss)
    best_idx = 0
    best_val_acc = -1.0
    best_val_loss = float("inf")
    for i, (acc, loss) in enumerate(zip(val_acc, val_loss)):
        if acc > best_val_acc or (acc == best_val_acc and loss < best_val_loss):
            best_val_acc = acc
            best_val_loss = loss
            best_idx = i

    best_epoch = epochs[best_idx]
    total_epochs = len(epochs)

    # Determine if early stopping triggered
    early_stopped = False
    if total_epochs < max_epochs:
        early_stopped = True
    elif total_epochs >= 5:
        # Check if the last 5 validation loss entries showed no improvement over previous best
        best_prior_loss = min(val_loss[:-5]) if len(val_loss) > 5 else float("inf")
        if min(val_loss[-5:]) >= best_prior_loss:
            early_stopped = True

    # 2-Panel Figure Generation
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel 1: Loss Chart
    ax1.plot(epochs, train_loss, label="Train Loss", color="#1f77b4", linewidth=2, marker='o', markersize=4)
    ax1.plot(epochs, val_loss, label="Validation Loss", color="#ff7f0e", linewidth=2, marker='s', markersize=4)
    ax1.axvline(x=best_epoch, color="#d62728", linestyle="--", linewidth=1.5, label=f"Best Epoch ({best_epoch})")
    ax1.set_title("Training vs Validation Loss", fontsize=13, fontweight="bold", pad=10)
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Loss", fontsize=11)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(fontsize=10, loc="upper right")

    # Panel 2: Accuracy Chart
    ax2.plot(epochs, train_acc, label="Train Accuracy", color="#1f77b4", linewidth=2, marker='o', markersize=4)
    ax2.plot(epochs, val_acc, label="Validation Accuracy", color="#2ca02c", linewidth=2, marker='s', markersize=4)
    ax2.axvline(x=best_epoch, color="#d62728", linestyle="--", linewidth=1.5, label=f"Best Epoch ({best_epoch})")
    ax2.set_title("Training vs Validation Accuracy", fontsize=13, fontweight="bold", pad=10)
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Accuracy", fontsize=11)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(fontsize=10, loc="lower right")

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[INFO] Diagnostic plot saved to '{save_path}' (150 DPI).")

    # Console Summary Table Output
    final_train_acc = train_acc[-1]
    print("\n" + "=" * 52)
    print("          TRAINING DIAGNOSTICS SUMMARY TABLE")
    print("=" * 52)
    print(f" Total Epochs Trained  : {total_epochs}")
    print(f" Best Epoch            : {best_epoch}")
    print(f" Best Val Accuracy     : {best_val_acc:.4f} ({best_val_acc * 100:.2f}%)")
    print(f" Best Val Loss         : {best_val_loss:.4f}")
    print(f" Final Train Accuracy  : {final_train_acc:.4f} ({final_train_acc * 100:.2f}%)")
    print(f" Early Stopping        : {'Triggered' if early_stopped else 'No (Reached Max Epochs)'}")
    print("=" * 52 + "\n")


if __name__ == "__main__":
    default_log = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_log.json")
    default_save = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_plots.png")

    parser = argparse.ArgumentParser(description="Plot training diagnostic charts from training_log.json.")
    parser.add_argument("--log_path", type=str, default=default_log, help="Path to training_log.json")
    parser.add_argument("--save_path", type=str, default=default_save, help="Path to save output plot PNG")
    parser.add_argument("--max_epochs", type=int, default=20, help="Maximum epochs configured for training")

    args = parser.parse_args()
    plot_training_diagnostics(args.log_path, args.save_path, args.max_epochs)
