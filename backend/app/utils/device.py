import torch

def get_device() -> torch.device:
    """
    Select CUDA device if available (optimised for NVIDIA GPUs like RTX 3050),
    otherwise fallback cleanly to CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
