"""PyTorch Dataset and DataLoaders for CUB-200 classification.

Usage:

    from src.data.dataset import get_dataloaders

    train_loader, val_loader, test_loader = get_dataloaders(batch_size=32)
    for images, labels in train_loader:  # images (32, 3, 224, 224), labels (32,)
        ...

On Windows, code that uses num_workers > 0 must run inside `if __name__ == "__main__":`.
"""
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.data_loader import load_cub_dataset
from src.data.splits import SEED, SPLIT_PATH, load_split
from src.data.transforms import IMAGE_SIZE, build_eval_transform, build_train_transform


class CUBClassificationDataset(Dataset):
    """CUB images for classification: returns (preprocessed image tensor, species label)."""

    def __init__(self, hf_dataset, indices, transform):
        self.hf_dataset = hf_dataset  # the Hugging Face split
        self.indices = list(indices)  # which images of that split to use
        self.transform = transform    # training or evaluation transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        sample = self.hf_dataset[int(self.indices[i])]
        image = self.transform(sample["image"])  # preprocessing happens here, every time
        return image, sample["label"]


def get_datasets(image_size=IMAGE_SIZE, split_path=SPLIT_PATH, random_crop=False):
    """Train (augmented), validation and test datasets built from the saved split."""
    train_dataset, test_dataset = load_cub_dataset()
    split = load_split(split_path)

    train_set = CUBClassificationDataset(train_dataset, split["train"], build_train_transform(image_size, random_crop))
    val_set = CUBClassificationDataset(train_dataset, split["val"], build_eval_transform(image_size))
    test_set = CUBClassificationDataset(test_dataset, range(len(test_dataset)), build_eval_transform(image_size))
    return train_set, val_set, test_set


def get_dataloaders(batch_size=32, num_workers=2, image_size=IMAGE_SIZE,
                    split_path=SPLIT_PATH, seed=SEED, random_crop=False):
    """Train / validation / test DataLoaders. Only the training loader is shuffled."""
    train_set, val_set, test_set = get_datasets(image_size, split_path, random_crop)

    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),  # faster copies to the GPU
        "persistent_workers": num_workers > 0,    # keep workers alive between epochs
    }
    generator = torch.Generator().manual_seed(seed)  # reproducible shuffling
    train_loader = DataLoader(train_set, shuffle=True, generator=generator, **common)
    val_loader = DataLoader(val_set, shuffle=False, **common)
    test_loader = DataLoader(test_set, shuffle=False, **common)
    return train_loader, val_loader, test_loader


def get_class_names():
    """The 200 species names, indexed by label."""
    train_dataset, _ = load_cub_dataset()
    return train_dataset.features["label"].names


if __name__ == "__main__":
    # Quick check: python -m src.data.dataset
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=32)
    for name, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
        print(f"{name}: {len(loader.dataset)} images, {len(loader)} batches")

    images, labels = next(iter(train_loader))
    print("images:", tuple(images.shape), images.dtype, f"range {images.min():.2f} to {images.max():.2f}")
    print("labels:", tuple(labels.shape), labels.dtype)
