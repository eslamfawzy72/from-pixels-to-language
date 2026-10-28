# Image Preprocessing: How to Use the Data

This guide explains how to get preprocessed CUB-200 images for training, and documents every function in `src/data/`.

**Short version:** call `get_dataloaders()` and loop over the batches. Everything else (cleaning, splitting, resizing, augmentation, normalization) happens automatically.

```python
from src.data.dataset import get_dataloaders

if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=32)

    for images, labels in train_loader:
        # images: float tensor (32, 3, 224, 224), normalized
        # labels: int tensor (32,), species numbers 0-199
        ...
```

## Contents

1. [Setup](#1-setup)
2. [What the preprocessing does](#2-what-the-preprocessing-does)
3. [The files](#3-the-files)
4. [Function reference](#4-function-reference)
5. [Common tasks](#5-common-tasks)
6. [Rules for the team](#6-rules-for-the-team)
7. [Troubleshooting](#7-troubleshooting)

---

## 1. Setup

**Local machine:**

```bash
git pull
pip install -r requirements.txt
python -m src.data.dataset      # checks that everything works
```

Run commands from the project root (`from-pixels-to-language/`), so that `src` can be imported.

**Google Colab / Kaggle:**

```python
!git clone https://github.com/<your-team>/from-pixels-to-language.git
%cd from-pixels-to-language
!pip install -r requirements.txt
```

The dataset (~1.1 GB) downloads from Hugging Face automatically the first time and is cached after that. No preprocessed files need to be shared: everything is computed on the fly from the original images.

Expected output of `python -m src.data.dataset`:

```
train: 5091 images, 160 batches
val: 899 images, 29 batches
test: 5794 images, 182 batches
images: (32, 3, 224, 224) torch.float32 range -2.12 to 2.64
labels: (32,) torch.int64
```

---

## 2. What the preprocessing does

### Which images (done once, saved in `data/splits/train_val_seed42.json`)

| Step | Result |
|---|---|
| Remove training images that duplicate test images (data leakage) | 2 removed |
| Remove second copies of duplicates inside train | 2 removed |
| Stratified 15% validation split, seed 42 | 5,091 train / 899 validation (4-5 per species) |
| Test set | unchanged: 5,794 images |

### How each image is prepared (on the fly, every time an image is loaded)

**Training images** (a new random version every epoch):

```
ToRGB -> ColorJitter -> Letterbox 224 -> RandomHorizontalFlip -> RandomAffine -> to tensor + normalize
```

**Validation and test images** (always the same result):

```
ToRGB -> Letterbox 224 -> to tensor + normalize
```

| Step | What it does | Why |
|---|---|---|
| `ToRGB` | Converts the 8 grayscale images to 3 channels | The CNN expects 3 channels |
| `ColorJitter` | Brightness and contrast ±20%, saturation ±10% (no hue) | Lighting varies; hue is not changed because color identifies species |
| `Letterbox` | Resize so the longest side is 224, pad to a square | The bird is never cut off or distorted |
| `RandomHorizontalFlip` | Mirror, 50% chance | A bird facing left is the same species |
| `RandomAffine` | Rotate ±15°, shift ±5%, zoom 0.9-1.1x | Camera angle and framing vary |
| Normalize | `(x - mean) / std` with the ImageNet values | Pretrained models expect it; padding becomes 0 |

The reasons behind every decision, with figures, are in `notebooks/eda.ipynb` and `notebooks/image_preprocessing.ipynb`.

---

## 3. The files

```
src/data/
├── data_loader.py    WHERE the images come from (Hugging Face)
├── splits.py         WHICH images are used (duplicates, train/val split)
├── transforms.py     HOW each image is prepared (RGB, letterbox, augmentation, normalization)
└── dataset.py        PUTS IT TOGETHER (Dataset + DataLoaders)  <- what you normally use
data/splits/
└── train_val_seed42.json   the saved split (committed; do not edit by hand)
```

How they connect when you call `get_dataloaders()`:

```
get_dataloaders()                        dataset.py
└── get_datasets()                       dataset.py
    ├── load_cub_dataset()               data_loader.py   -> the images
    ├── load_split()                     splits.py        -> which images
    ├── build_train_transform()          transforms.py    -> how (training)
    ├── build_eval_transform()           transforms.py    -> how (val / test)
    └── CUBClassificationDataset x 3     dataset.py       -> train / val / test
└── DataLoader x 3                       PyTorch          -> batches
```

During training, each time the `DataLoader` needs an image it calls `CUBClassificationDataset.__getitem__`, which loads the image and runs the transform. That is why augmentation is different every epoch.

---

## 4. Function reference

### `src/data/dataset.py`: the main entry point

#### `get_dataloaders(batch_size=32, num_workers=2, image_size=224, split_path=SPLIT_PATH, seed=42)`

Builds the train, validation and test DataLoaders. **This is the function you normally need.**

| Argument | Default | Meaning |
|---|---|---|
| `batch_size` | `32` | Images per batch. Lower it if the GPU runs out of memory. |
| `num_workers` | `2` | Processes preparing batches in parallel. Use `0` inside a Jupyter notebook on Windows. |
| `image_size` | `224` | Output size (square). For example `448` for a higher-resolution experiment. |
| `split_path` | `data/splits/train_val_seed42.json` | Which saved split to use. |
| `seed` | `42` | Seed for the training shuffle order (reproducible runs). |

**Returns:** `(train_loader, val_loader, test_loader)`

| Loader | Images | Augmented | Shuffled |
|---|---|---|---|
| `train_loader` | 5,091 | yes | yes |
| `val_loader` | 899 | no | no |
| `test_loader` | 5,794 | no | no |

Each batch is `(images, labels)`: `images` is a float tensor `(batch_size, 3, image_size, image_size)`, `labels` is an int64 tensor `(batch_size,)` with values 0-199. The last batch can be smaller.

```python
train_loader, val_loader, test_loader = get_dataloaders(batch_size=64, num_workers=4)
```

#### `get_datasets(image_size=224, split_path=SPLIT_PATH)`

Same as above but returns the three **datasets** without DataLoaders. Useful to look at single images or to build your own DataLoader (for example a different sampler).

**Returns:** `(train_set, val_set, test_set)`, each a `CUBClassificationDataset`.

```python
train_set, val_set, test_set = get_datasets()
image, label = train_set[0]   # one preprocessed image: tensor (3, 224, 224), int label
print(len(train_set))         # 5091
```

#### `get_class_names()`

**Returns:** list of the 200 species names, indexed by label (`class_names[57]` is the name of label 57). Use it for plots, confusion matrices and printing predictions.

```python
class_names = get_class_names()
print(class_names[0])   # "Black footed Albatross"
```

#### `class CUBClassificationDataset(hf_dataset, indices, transform)`

A PyTorch `Dataset`. You normally get it from `get_datasets()` instead of creating it yourself.

| Argument | Meaning |
|---|---|
| `hf_dataset` | A Hugging Face split (from `load_cub_dataset()`) |
| `indices` | Which images of that split to use (e.g. `split["train"]`) |
| `transform` | The preprocessing to apply (from `build_train_transform()` or `build_eval_transform()`) |

- `len(dataset)`: number of images.
- `dataset[i]`: loads image `indices[i]`, applies `transform`, returns `(tensor, label)`.

---

### `src/data/transforms.py`: how each image is prepared

#### Constants

| Name | Value | Meaning |
|---|---|---|
| `IMAGE_SIZE` | `224` | Default output size |
| `IMAGENET_MEAN` | `[0.485, 0.456, 0.406]` | Per-channel mean used for normalization |
| `IMAGENET_STD` | `[0.229, 0.224, 0.225]` | Per-channel standard deviation used for normalization |
| `PAD_COLOR` | `(124, 116, 104)` | Padding color: the ImageNet mean x 255, becomes 0 after normalization |

#### `build_train_transform(image_size=224)`

**Returns** the training transform: a callable that takes a PIL image and returns a normalized, **randomly augmented** tensor `(3, image_size, image_size)`. Every call gives a different result.

```python
train_tf = build_train_transform()
x = train_tf(pil_image)   # tensor (3, 224, 224)
```

#### `build_eval_transform(image_size=224)`

**Returns** the evaluation transform: PIL image -> normalized tensor, **no randomness**. Use it for validation, test, and for predicting on a new image (for example in the CLI).

```python
from PIL import Image
eval_tf = build_eval_transform()
x = eval_tf(Image.open("bird.jpg")).unsqueeze(0)   # (1, 3, 224, 224): a batch of one
```

#### `denormalize(x)`

Undoes the normalization of **one** image tensor `(3, H, W)` so it can be displayed. **Returns** a NumPy array `(H, W, 3)` with values 0-1. Use it to plot model inputs, wrong predictions or Grad-CAM overlays (a normalized tensor looks wrong when plotted directly).

```python
import matplotlib.pyplot as plt
images, labels = next(iter(val_loader))
plt.imshow(denormalize(images[0]))
```

#### `class ToRGB()`

Callable: converts any PIL image to RGB. Grayscale values are copied into all three channels. RGB images are returned unchanged.

#### `class Letterbox(size=224, fill=PAD_COLOR)`

Callable: resizes a PIL image so its longest side is `size`, then pads it to a `size x size` square, centered. Nothing is cut off or distorted.

```python
square = Letterbox(224)(pil_image)   # PIL image, 224 x 224
```

#### `to_tensor()`

**Returns** the normalization step on its own: PIL image -> float tensor `(3, H, W)`, scaled to 0-1, then normalized with the ImageNet mean and std. It is the last step of both transforms above.

---

### `src/data/splits.py`: which images are used

You normally only use `load_split()` (and `get_dataloaders()` already calls it). The other functions recreate the split file.

#### Constants

| Name | Value |
|---|---|
| `SEED` | `42` |
| `VAL_FRACTION` | `0.15` |
| `SPLIT_PATH` | `data/splits/train_val_seed42.json` |

#### `load_split(path=SPLIT_PATH)`

Reads the saved split. **Returns** a dict:

| Key | Contents |
|---|---|
| `"train"` | 5,091 indices into the Hugging Face `CUB_train` split |
| `"val"` | 899 indices into the Hugging Face `CUB_train` split |
| `"removed_duplicates"` | the 4 removed images, with species and reason |
| `"seed"`, `"val_fraction"`, `"description"` | how the split was made |

Raises `FileNotFoundError` if the file is missing.

```python
split = load_split()
train_images = train_dataset.select(split["train"])   # Hugging Face subset of the training images
```

#### `find_train_duplicates(train_dataset, test_dataset)`

Finds training images to remove, using a perceptual hash: copies of test images (leakage) and repeated copies inside train. The test set is never changed. Reads every image (~30 seconds). **Returns** a list of dicts `{"idx", "species", "reason"}`.

#### `make_split(train_dataset, removed, val_fraction=0.15, seed=42)`

Removes the `removed` images, then makes a stratified train/validation split. **Returns** the split dict (same format as `load_split()`).

#### `save_split(split, path=SPLIT_PATH)`

Writes a split dict to a JSON file.

#### `dhash(img, size=8)`

Perceptual hash of a PIL image: a 16-character hex string. Near-identical images (even at different sizes) get the same value.

#### Recreate the split file

```bash
python -m src.data.splits
```

Only needed if the rules change. With the same rules and seed it produces exactly the same file.

---

### `src/data/data_loader.py`: the raw data

#### `load_cub_dataset()`

**Returns** `(train_dataset, test_dataset)`: the raw Hugging Face splits `Multimodal-Fatima/CUB_train` (5,994 images) and `Multimodal-Fatima/CUB_test` (5,794 images). Each record has `image` (PIL), `label` (0-199), `description` (captions), `file_name` and `id`. These are **not** preprocessed and **not** deduplicated.

#### `inspect_dataset(dataset, name="Dataset")`

Prints the number of records, the columns, the feature types and the first record.

---

## 5. Common tasks

### Train a classifier

```python
import torch
from torch import nn
from src.data.dataset import get_dataloaders


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=32, num_workers=2)

    model = ...  # your CNN, with 200 outputs
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(30):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            loss = loss_fn(model(images), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                correct += (model(images).argmax(dim=1) == labels).sum().item()
                total += labels.size(0)
        print(f"epoch {epoch + 1}: validation accuracy {correct / total:.3f}")


if __name__ == "__main__":  # required on Windows when num_workers > 0
    main()
```

### Use a pretrained model (transfer learning)

No change is needed: the images are already 224x224 and normalized with the ImageNet values that torchvision's pretrained models expect.

```python
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights

model = resnet50(weights=ResNet50_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 200)   # 200 bird species
```

### Look at a batch

```python
import matplotlib.pyplot as plt
from src.data.dataset import get_dataloaders, get_class_names
from src.data.transforms import denormalize

train_loader, _, _ = get_dataloaders(batch_size=8, num_workers=0)
class_names = get_class_names()
images, labels = next(iter(train_loader))

fig, axes = plt.subplots(2, 4, figsize=(12, 6))
for ax, image, label in zip(axes.flat, images, labels):
    ax.imshow(denormalize(image))
    ax.set_title(class_names[int(label)], fontsize=8)
    ax.axis("off")
plt.show()
```

### Predict one new image

Uses a trained `model`, `device` and `class_names` from the examples above.

```python
from PIL import Image
from src.data.transforms import build_eval_transform

x = build_eval_transform()(Image.open("bird.jpg")).unsqueeze(0).to(device)  # (1, 3, 224, 224)
with torch.no_grad():
    predicted = model(x).argmax(dim=1).item()
print(class_names[predicted])
```

### Higher resolution experiment

```python
train_loader, val_loader, test_loader = get_dataloaders(batch_size=16, image_size=448)
```

About 4x slower than 224. Lower the batch size if the GPU runs out of memory.

### Final model: train on train + validation

After tuning, the final model can be retrained on all 5,990 clean training images:

```python
from torch.utils.data import DataLoader
from src.data.dataset import CUBClassificationDataset
from src.data.data_loader import load_cub_dataset
from src.data.splits import load_split
from src.data.transforms import build_train_transform

train_dataset, _ = load_cub_dataset()
split = load_split()
full_train = CUBClassificationDataset(train_dataset, split["train"] + split["val"], build_train_transform())
full_loader = DataLoader(full_train, batch_size=32, shuffle=True, num_workers=2)
```

---

## 6. Rules for the team

1. **Always load data through `get_dataloaders()` / `get_datasets()`.** Then every model uses the same cleaned images, the same split and the same preprocessing, which keeps the comparisons (from-scratch vs transfer learning, tuning runs) fair.
2. **Tune on `val_loader`, never on `test_loader`.** Hyperparameter search, early stopping and choosing the best model all use validation. The test set is used once, for the final evaluation; otherwise the reported score is biased.
3. **Do not edit `data/splits/train_val_seed42.json` by hand.** If the split must change, change the code in `splits.py`, run `python -m src.data.splits`, and tell the team: earlier results are then no longer comparable.
4. **Do not use the raw `load_cub_dataset()` splits for training.** They still contain the 4 duplicate images, including 2 copies of test images.
5. **If you change a transform** (for example the augmentation), say so in the experiment tracking, so runs can be compared correctly.

---

## 7. Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'src'` | Run from the project root (`from-pixels-to-language/`), or in a notebook add the project root to `sys.path` (see the first cells of `notebooks/image_preprocessing.ipynb`). |
| `FileNotFoundError: ...train_val_seed42.json not found` | Run `git pull` (the file is committed), or recreate it: `python -m src.data.splits`. |
| Error about worker processes / `freeze_support` on Windows | Put your code inside `if __name__ == "__main__":`, or use `num_workers=0`. |
| Data loading hangs or fails inside a Jupyter notebook on Windows | Use `get_dataloaders(num_workers=0)`. |
| `CUDA out of memory` | Lower `batch_size` (e.g. 16), especially with `image_size=448`. |
| Training is slow and the GPU is idle | Increase `num_workers` (2-4 on Colab/Kaggle). |
| Images look strange when plotted | Use `denormalize(image)` before `plt.imshow`. |
| Warnings about HF_TOKEN or symlinks | Harmless: the dataset is public and still downloads and caches correctly. |
