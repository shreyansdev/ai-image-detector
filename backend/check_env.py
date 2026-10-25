#!/usr/bin/env python3
"""
Development environment verification script for model training.
Checks Python, PyTorch, CUDA, Torchvision, and System RAM,
and verifies model loading & execution with a dummy ResNet-50 forward pass.
"""

import os
import sys
import platform

def get_system_ram():
    """Returns available and total system RAM in GB."""
    try:
        import psutil
        mem = psutil.virtual_memory()
        total_gb = mem.total / (1024 ** 3)
        available_gb = mem.available / (1024 ** 3)
        return total_gb, available_gb
    except ImportError:
        pass

    # Fallback for Linux via /proc/meminfo
    if os.path.exists('/proc/meminfo'):
        try:
            mem_info = {}
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    parts = line.split(':')
                    if len(parts) == 2:
                        key = parts[0].strip()
                        val = parts[1].strip().split()[0]
                        mem_info[key] = int(val)
            total_gb = mem_info.get('MemTotal', 0) / (1024 ** 2)
            available_gb = mem_info.get('MemAvailable', mem_info.get('MemFree', 0)) / (1024 ** 2)
            return total_gb, available_gb
        except Exception:
            pass

    # Fallback using os.sysconf
    try:
        pages = os.sysconf('SC_PHYS_PAGES')
        page_size = os.sysconf('SC_PAGE_SIZE')
        total_gb = (pages * page_size) / (1024 ** 3)
        return total_gb, None
    except Exception:
        return None, None

def check_environment():
    print("=" * 60)
    print("       ENVIRONMENT & MODEL SETUP VERIFICATION")
    print("=" * 60)
    
    all_passed = True
    
    # 1. Python Version
    py_ver = sys.version.split()[0]
    print(f"[1] Python Version: {py_ver} ({platform.python_implementation()})")
    
    # 2. PyTorch & CUDA Check
    pytorch_installed = False
    cuda_available = False
    device = "cpu"
    
    try:
        import torch
        pytorch_installed = True
        torch_ver = torch.__version__
        cuda_available = torch.cuda.is_available()
        print(f"[2] PyTorch Version: {torch_ver}")
        print(f"    CUDA Available: {cuda_available}")
        
        if cuda_available:
            device = "cuda"
            # 3. CUDA Device & VRAM Info
            gpu_name = torch.cuda.get_device_name(0)
            total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            print(f"[3] CUDA Device Name: {gpu_name}")
            print(f"    Total VRAM Available: {total_vram_gb:.2f} GB")
        else:
            print("[3] CUDA Device Info: N/A (GPU not detected/available)")
            print("    [WARNING] CUDA is not available! Model training will fall back to CPU.")
            print("              Training on CPU may be significantly slower.")
    except ImportError:
        print("[2] PyTorch Version: NOT INSTALLED")
        print("    [ERROR] PyTorch is required but not installed.")
        all_passed = False

    # 4. Torchvision Check
    torchvision_installed = False
    try:
        import torchvision
        torchvision_installed = True
        print(f"[4] Torchvision Version: {torchvision.__version__}")
    except ImportError:
        print("[4] Torchvision Version: NOT INSTALLED")
        print("    [ERROR] torchvision is required but not installed.")
        all_passed = False

    # 5. System RAM Check
    total_ram, avail_ram = get_system_ram()
    if total_ram is not None:
        if avail_ram is not None:
            print(f"[5] System RAM: {avail_ram:.2f} GB available (Total: {total_ram:.2f} GB)")
        else:
            print(f"[5] System RAM: Total {total_ram:.2f} GB")
    else:
        print("[5] System RAM: Unable to determine")

    print("-" * 60)
    print("Running End-to-End Dummy Model Test...")
    
    # E2E Test: ResNet-50 forward pass
    if pytorch_installed and torchvision_installed:
        try:
            import torch
            import torchvision.models as models

            print(f" -> Target Device for Test: {device.upper()}")
            print(" -> Instantiating ResNet-50 model from torchvision...")
            try:
                model = models.resnet50(weights=None)
            except TypeError:
                model = models.resnet50(pretrained=False)
                
            model = model.to(device)
            model.eval()
            
            print(" -> Running dummy forward pass with random 224x224 tensor...")
            dummy_input = torch.randn(1, 3, 224, 224, device=device)
            
            with torch.no_grad():
                output = model(dummy_input)
                
            print(f" -> Output Tensor Shape: {output.shape}")
            if output.shape == (1, 1000):
                print(" -> Forward pass completed successfully!")
            else:
                print(f" -> [ERROR] Unexpected output shape: {output.shape}")
                all_passed = False
        except Exception as e:
            print(f" -> [ERROR] Dummy forward pass failed with exception: {e}")
            all_passed = False
    else:
        print(" -> Skipped due to missing PyTorch or Torchvision.")
        all_passed = False

    print("=" * 60)
    if all_passed:
        print("=== SUMMARY: PASS ===")
        print("Environment configuration and model test succeeded.")
        sys.exit(0)
    else:
        print("=== SUMMARY: FAIL ===")
        print("One or more environment checks or tests failed.")
        sys.exit(1)

if __name__ == "__main__":
    check_environment()
