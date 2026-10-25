import warnings
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = False
Image.MAX_IMAGE_PIXELS = None
warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)

import os
import torch
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torchvision import datasets, transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt

# Define standard ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class TransformedSubset(Dataset):
    """
    Wrapper around a torch.utils.data.Subset or Dataset to apply
    specific torchvision transforms dynamically upon item retrieval.
    """
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform is not None:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)


def get_transforms():
    """
    Returns training and validation/test transformation pipelines.
    """
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    return train_transform, val_transform


def resolve_dataset_dir(data_dir: str) -> str:
    """
    Resolves the dataset directory path, handling nested structures like archive/train.
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(
            f"Dataset directory '{data_dir}' does not exist."
        )
    
    candidates = [
        data_dir,
        os.path.join(data_dir, "merged_dataset"),
        os.path.join(data_dir, "archive", "train"),
        os.path.join(data_dir, "archive"),
        os.path.join(data_dir, "train")
    ]
    
    for cand in candidates:
        if os.path.exists(cand):
            try:
                subdirs = [d for d in os.listdir(cand) if os.path.isdir(os.path.join(cand, d))]
                if any(s.lower() in ['real', 'fake', 'ai_generated', 'ai'] for s in subdirs):
                    return cand
            except Exception:
                pass
                
    return data_dir


def prepare_datasets(data_dir: str, seed: int = 42):
    """
    Loads dataset from data_dir using ImageFolder and splits into 70% train, 15% val, 15% test.
    """
    resolved_dir = resolve_dataset_dir(data_dir)

    # Load raw dataset without transforms to apply split-specific transforms later
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

    train_transform, val_transform = get_transforms()

    train_dataset = TransformedSubset(train_sub, transform=train_transform)
    val_dataset = TransformedSubset(val_sub, transform=val_transform)
    test_dataset = TransformedSubset(test_sub, transform=val_transform)

    return train_dataset, val_dataset, test_dataset, raw_dataset.class_to_idx


def get_dataloaders(data_dir: str, batch_size: int = 32, num_workers: int = 0, seed: int = 42):
    """
    Creates DataLoader objects for train, validation, and test splits with hardware optimizations.
    """
    train_ds, val_ds, test_ds, class_to_idx = prepare_datasets(data_dir, seed=seed)
    pin_memory = torch.cuda.is_available()
    persistent = (num_workers > 0)
    prefetch = 4 if num_workers > 0 else None

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=persistent, prefetch_factor=prefetch
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=persistent, prefetch_factor=prefetch
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=persistent, prefetch_factor=prefetch
    )

    return train_loader, val_loader, test_loader, class_to_idx


def get_dataset_stats(loader: DataLoader, name: str = "Dataset Split"):
    """
    Prints class distribution and total sample counts for a given DataLoader/split.
    """
    dataset = loader.dataset
    total_samples = len(dataset)

    # Efficiently extract targets if available via subset structure
    targets = []
    curr = dataset
    while hasattr(curr, 'subset'):
        curr = curr.subset

    if isinstance(curr, Subset):
        indices = curr.indices
        base_ds = curr.dataset
        if hasattr(base_ds, 'targets'):
            targets = [base_ds.targets[i] for i in indices]
        elif hasattr(base_ds, 'samples'):
            targets = [base_ds.samples[i][1] for i in indices]

    # Fallback to iteration if targets couldn't be extracted directly
    if not targets:
        targets = [y for _, y in loader]
        if targets and isinstance(targets[0], torch.Tensor):
            targets = torch.cat(targets).tolist()

    class_counts = {}
    for t in targets:
        class_counts[t] = class_counts.get(t, 0) + 1

    print(f"\n--- {name} Statistics ---")
    print(f"Total Samples: {total_samples}")
    print("Class Distribution:")
    for class_idx in sorted(class_counts.keys()):
        count = class_counts[class_idx]
        pct = (count / total_samples) * 100 if total_samples > 0 else 0
        print(f"  Class {class_idx}: {count} samples ({pct:.2f}%)")

    return {"total": total_samples, "counts": class_counts}


def visualize_samples(loader: DataLoader, n: int = 8, save_path: str = "backend/ml/sample_grid.png"):
    """
    Saves a grid of sample images with labels to the specified output path.
    """
    images, labels = next(iter(loader))
    n = min(n, len(images))
    images = images[:n]
    labels = labels[:n]

    # Denormalize ImageNet normalization
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    images_denorm = images * std + mean
    images_denorm = torch.clamp(images_denorm, 0, 1)

    grid = vutils.make_grid(images_denorm, nrow=4, padding=2)
    np_grid = grid.permute(1, 2, 0).cpu().numpy()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(12, 6))
    plt.imshow(np_grid)
    plt.axis("off")
    plt.title(f"Sample Batch Grid (n={n})\nLabels: {labels.tolist()}")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Sample grid saved successfully to '{save_path}'.")


if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data"
    try:
        print(f"Loading datasets from '{data_dir}'...")
        train_l, val_l, test_l, class_map = get_dataloaders(data_dir)
        print(f"Class mapping: {class_map}")
        get_dataset_stats(train_l, name="Train Split")
        get_dataset_stats(val_l, name="Validation Split")
        get_dataset_stats(test_l, name="Test Split")
        visualize_samples(train_l, n=8)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
