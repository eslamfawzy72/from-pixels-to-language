"""Duplicate removal and the stratified train / validation split.

The split is saved to data/splits/ and committed, so the whole team trains and tunes on the
same images. Regenerate it (only needed if the rules change) with:

    python -m src.data.splits
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SEED = 42
VAL_FRACTION = 0.15
SPLIT_PATH = PROJECT_ROOT / "data" / "splits" / f"train_val_seed{SEED}.json"


def dhash(img, size=8):
    """Perceptual hash: near-identical images get the same 64-bit value, even at a different size."""
    gray = img.convert("L").resize((size + 1, size), Image.BILINEAR)
    pixels = np.asarray(gray, dtype=np.int16)
    bits = (pixels[:, 1:] > pixels[:, :-1]).flatten()
    return f"{int(''.join('1' if b else '0' for b in bits), 2):016x}"


def find_train_duplicates(train_dataset, test_dataset):
    """Training images to remove: copies of test images (leakage) and repeated copies inside train.

    The test set is never changed.
    """
    test_hashes = {}
    for idx, img in enumerate(test_dataset["image"]):
        test_hashes.setdefault(dhash(img), []).append(idx)

    class_names = train_dataset.features["label"].names
    seen = set()
    removed = []
    for idx, (img, label) in enumerate(zip(train_dataset["image"], train_dataset["label"])):
        h = dhash(img)
        if h in test_hashes:
            reason = "same photo as test " + ", ".join(f"#{i}" for i in test_hashes[h])
        elif h in seen:
            reason = "second copy of a training image"
        else:
            seen.add(h)
            continue
        seen.add(h)
        removed.append({"idx": idx, "species": class_names[label], "reason": reason})
    return removed


def make_split(train_dataset, removed, val_fraction=VAL_FRACTION, seed=SEED):
    """Stratified split of the remaining training images (same share of every species)."""
    labels = np.array(train_dataset["label"])
    removed_idx = {r["idx"] for r in removed}
    keep_idx = np.array([i for i in range(len(train_dataset)) if i not in removed_idx])

    train_idx, val_idx = train_test_split(
        keep_idx,
        test_size=val_fraction,
        stratify=labels[keep_idx],
        random_state=seed,
    )
    return {
        "description": "Indices into the Hugging Face split Multimodal-Fatima/CUB_train",
        "seed": seed,
        "val_fraction": val_fraction,
        "removed_duplicates": removed,
        "train": sorted(int(i) for i in train_idx),
        "val": sorted(int(i) for i in val_idx),
    }


def save_split(split, path=SPLIT_PATH):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split, indent=1))


def load_split(path=SPLIT_PATH):
    """Load the saved split: a dict with "train" and "val" lists of indices into CUB_train."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found. Create it with: python -m src.data.splits")
    return json.loads(path.read_text())


if __name__ == "__main__":
    from src.data.data_loader import load_cub_dataset

    train_dataset, test_dataset = load_cub_dataset()

    print("Finding duplicates (reads every image, about a minute)...")
    removed = find_train_duplicates(train_dataset, test_dataset)
    for r in removed:
        print(f"  remove train #{r['idx']} ({r['species']}): {r['reason']}")

    split = make_split(train_dataset, removed)
    save_split(split)
    print(f"Train: {len(split['train'])} | Validation: {len(split['val'])} | saved to {SPLIT_PATH}")
