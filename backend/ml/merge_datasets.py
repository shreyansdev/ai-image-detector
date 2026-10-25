#!/usr/bin/env python3
"""
Merge Datasets Script for AI Image Detector.

Disk space note: Since images are being copied (not moved) and duplicated across
two locations, you will need roughly double the combined size of both source
datasets in free disk space. The script will check for sufficient disk space
before starting any copying.

This script combines Dataset A (already used) and Dataset B (new Deepfake and Real Images)
into a unified dataset folder at:
  backend/ml/data/merged_dataset/
with two subfolders: 'real' and 'fake'.
Files are prefixed with their source dataset name to avoid name collisions,
and validated using PIL to filter out corrupted files.
"""

import os
import sys
import argparse
import shutil
from PIL import Image, ImageFile
from tqdm import tqdm

# Prevent PIL from crashing on truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = False
# Disable pixel limit to avoid errors on very large images
Image.MAX_IMAGE_PIXELS = None

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.tiff', '.tif'}

def get_class_label(file_path):
    """
    Traverses up the file path parts to determine if the file belongs to
    the 'real' or 'fake' class (case-insensitive).
    """
    path_parts = os.path.normpath(file_path).split(os.sep)
    for part in reversed(path_parts):
        part_lower = part.lower()
        if part_lower == "real":
            return "real"
        elif part_lower == "fake":
            return "fake"
    return None

def scan_dataset(dataset_path, prefix):
    """
    Scans the dataset path recursively, counting files and grouping them
    by class (real/fake). Returns a list of dicts with file metadata.
    """
    scanned_files = []
    
    if not os.path.isdir(dataset_path):
        print(f"[-] Error: Source directory does not exist: {dataset_path}")
        sys.exit(1)
        
    print(f"[*] Scanning {prefix} at: {dataset_path}...")
    for root, _, files in os.walk(dataset_path):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in IMAGE_EXTENSIONS:
                fpath = os.path.join(root, f)
                label = get_class_label(fpath)
                if label:
                    try:
                        size = os.path.getsize(fpath)
                        scanned_files.append({
                            'src_path': fpath,
                            'label': label,
                            'size': size
                        })
                    except OSError:
                        pass
    
    real_count = sum(1 for x in scanned_files if x['label'] == 'real')
    fake_count = sum(1 for x in scanned_files if x['label'] == 'fake')
    print(f"[+] Found {len(scanned_files)} valid images (Real: {real_count} | Fake: {fake_count})")
    return scanned_files

def check_disk_space(output_path, estimated_size_bytes):
    """
    Checks if the destination path has enough free space for the merge.
    """
    check_path = output_path
    while not os.path.exists(check_path):
        parent = os.path.dirname(check_path)
        if parent == check_path:
            break
        check_path = parent
        
    try:
        usage = shutil.disk_usage(check_path)
        return usage.free
    except Exception as e:
        print(f"[!] Warning: Could not check disk usage: {e}")
        return None

def validate_image(file_path):
    """
    Validates an image by opening it and converting to RGB.
    Returns True if valid, False otherwise.
    """
    try:
        with Image.open(file_path) as img:
            img.convert("RGB")
        return True
    except Exception:
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Merge two image datasets for Real vs Fake classification."
    )
    parser.add_argument(
        "--dataset_a_path",
        required=True,
        help="Path to Dataset A (containing 'real' and 'fake' subfolders)"
    )
    parser.add_argument(
        "--dataset_b_path",
        required=True,
        help="Path to Dataset B (containing train/valid/test with Real/Fake)"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Scan and check but do not copy any files or create directories"
    )
    
    args = parser.parse_args()
    
    # Determine the script's directory and output directory relative to it
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.abspath(os.path.join(script_dir, "data", "merged_dataset"))
    
    print("=" * 65)
    print("                 DATASET MERGING UTILITY")
    print("=" * 65)
    
    if args.dry_run:
        print("[!] DRY RUN MODE: No directories will be created and no files copied.")
        
    # Scan datasets
    files_a = scan_dataset(args.dataset_a_path, "Dataset A")
    files_b = scan_dataset(args.dataset_b_path, "Dataset B")
    
    # Calculate disk usage estimates
    total_size_bytes = sum(x['size'] for x in files_a) + sum(x['size'] for x in files_b)
    total_size_gb = total_size_bytes / (1024 ** 3)
    
    print(f"\n[*] Disk Space Check:")
    print(f"    Estimated space needed: {total_size_gb:.2f} GB")
    
    free_space = check_disk_space(output_dir, total_size_bytes)
    if free_space is not None:
        free_space_gb = free_space / (1024 ** 3)
        print(f"    Available free space:  {free_space_gb:.2f} GB")
        
        if free_space < total_size_bytes:
            print(f"\n[-] ERROR: Insufficient disk space on target volume!")
            print(f"    Required:  {total_size_gb:.2f} GB")
            print(f"    Available: {free_space_gb:.2f} GB")
            if not args.dry_run:
                print("[-] Aborting due to lack of disk space.")
                sys.exit(1)
            else:
                print("[!] Dry run proceeding despite disk space shortage...")
    else:
        print("[!] Could not determine available free space.")

    # Initialize count tracking
    real_a = sum(1 for x in files_a if x['label'] == 'real')
    fake_a = sum(1 for x in files_a if x['label'] == 'fake')
    
    real_b = sum(1 for x in files_b if x['label'] == 'real')
    fake_b = sum(1 for x in files_b if x['label'] == 'fake')
    
    corrupted_count = 0
    copied_real = 0
    copied_fake = 0
    
    if not args.dry_run:
        # Create output directories
        real_dest = os.path.join(output_dir, "real")
        fake_dest = os.path.join(output_dir, "fake")
        os.makedirs(real_dest, exist_ok=True)
        os.makedirs(fake_dest, exist_ok=True)
        
        # Combine files for tqdm progress
        all_files_to_copy = []
        for item in files_a:
            all_files_to_copy.append((item, args.dataset_a_path, "datasetA"))
        for item in files_b:
            all_files_to_copy.append((item, args.dataset_b_path, "datasetB"))
            
        print(f"\n[*] Copying and validating {len(all_files_to_copy)} images...")
        
        for item, start_dir, prefix in tqdm(all_files_to_copy, desc="Merging", unit="img"):
            # Compute destination path to avoid collisions
            rel_path = os.path.relpath(item['src_path'], start_dir)
            safe_name = rel_path.replace(os.sep, "_")
            dest_filename = f"{prefix}_{safe_name}"
            dest_path = os.path.join(output_dir, item['label'], dest_filename)
            
            # Validate image integrity before copying
            if validate_image(item['src_path']):
                try:
                    shutil.copy2(item['src_path'], dest_path)
                    if item['label'] == 'real':
                        copied_real += 1
                    else:
                        copied_fake += 1
                except Exception as e:
                    print(f"\n[!] Error copying {item['src_path']}: {e}")
                    corrupted_count += 1
            else:
                corrupted_count += 1
    else:
        # For dry run, assume all scanned images are valid
        copied_real = real_a + real_b
        copied_fake = fake_a + fake_b
        corrupted_count = 0

    total_copied = copied_real + copied_fake
    
    # 7. Print Summary
    print("\n" + "=" * 65)
    print("                         SUMMARY")
    print("=" * 65)
    print(f"Total images from Dataset A: {len(files_a)} (Real: {real_a} | Fake: {fake_a})")
    print(f"Total images from Dataset B: {len(files_b)} (Real: {real_b} | Fake: {fake_b})")
    print(f"Final merged totals:         {total_copied} (Real: {copied_real} | Fake: {copied_fake})")
    
    # Class balance calculation
    if total_copied > 0:
        real_pct = (copied_real / total_copied) * 100
        fake_pct = (copied_fake / total_copied) * 100
        print(f"Final class balance ratio:   Real: {real_pct:.1f}% | Fake: {fake_pct:.1f}%")
    else:
        real_pct = 0
        fake_pct = 0
        print(f"Final class balance ratio:   Real: 0% | Fake: 0%")
        
    if args.dry_run:
        print(f"Number of corrupted files skipped: 0 (dry run)")
    else:
        print(f"Number of corrupted files skipped: {corrupted_count}")
    print("=" * 65)
    
    # 8. Class imbalance check
    if total_copied > 0 and (real_pct > 60.0 or real_pct < 40.0):
        print("\n[!] WARNING: Class imbalance detected. Consider undersampling the majority class for better training results.")
        
    # 9. Format instructions printout
    # We display paths relative to the project root for copy-paste convenience
    rel_output_path = os.path.relpath(output_dir, os.getcwd())
    
    print(f"\nMerged dataset ready at: {rel_output_path}")
    print("Use this path for training:")
    print(f"  python backend/ml/run_training_pipeline.py --data_path {rel_output_path} --unfreeze_at 8 --fresh_start\n")

if __name__ == "__main__":
    main()
