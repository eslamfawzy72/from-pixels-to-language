# From Pixels to Language

## Objectives

This project aims to build an end-to-end multimodal deep learning system that performs three tasks:

- **Image Classification:** Train a CNN encoder from scratch and compare its performance with transfer learning.
- **Image Colorization:** Reconstruct color images from grayscale inputs.
- **Image Captioning:** Generate natural-language descriptions of images using a visual encoder and a sequence decoder.

The same visual encoder serves as the backbone for all three tasks.

## Project Structure

```text
from-pixels-to-language/
├── README.md
├── requirements.txt          # List of required Python libraries and dependencies used in the project
├── configs/                  # Hyperparameters and run settings
├── notebooks/                # Experiments only (not part of the final code)
└── src/
    ├── data/                 # Datasets, transforms, augmentation
    ├── models/               # Encoder, colorization model, caption decoder
    ├── training/             # Training loops
    └── evaluation/           # Metrics and benchmarking
```
## Installation

### 1. Create a virtual environment (Optional but recommended)

```bash
python -m venv .venv
```

### 2. Activate the virtual environment

**Windows (PowerShell):**

```powershell
.\.venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**

```cmd
.venv\Scripts\activate.bat
```

### 3. Install dependencies

Install all required Python libraries listed in `requirements.txt`:

```bash
python -m pip install -r requirements.txt
```

## Dataset

The project uses the **CUB-200-2011 dataset with captions**, which contains images of 200 bird species, with ten human-written descriptions for each image.

### Loading the Dataset

Install the required library:

```bash
pip install datasets
```

```bash
Load the training and test splits using Hugging Face:

from datasets import load_datasettrain_dataset = load_dataset( "Multimodal-Fatima/CUB_train",split="train")
test_dataset = load_dataset("Multimodal-Fatima/CUB_test",split="test")
print(train_dataset[0])


```
The dataset includes the following fields:
- image: The bird photograph.
- label: The bird species.
- description: Ten captions separated by newline characters.
- file_name: The original image filename.
- id: The record identifier.
For image colorization, grayscale images will be generated from the original color images, which will serve as the target outputs.

## Preprocessed Data for Classification

All preprocessing lives in `src/data/` and runs on the fly while training, so no preprocessed files need to be shared. The dataset downloads automatically from Hugging Face on first use.

| Module | Contents |
|---|---|
| `src/data/transforms.py` | RGB conversion, letterbox resize to 224x224, training augmentation, ImageNet normalization |
| `src/data/splits.py` | Duplicate removal and the stratified train / validation split |
| `src/data/dataset.py` | PyTorch `Dataset` and `get_dataloaders()` |
| `data/splits/train_val_seed42.json` | The saved split (5,091 train / 899 validation); committed so everyone uses the same images |

Load the data in a training script:

```python
from src.data.dataset import get_dataloaders, get_class_names

if __name__ == "__main__":  # required on Windows when num_workers > 0
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=32, num_workers=2)
    class_names = get_class_names()

    for images, labels in train_loader:  # images: (32, 3, 224, 224), labels: (32,) in 0-199
        ...
```

- Training batches are augmented and shuffled; validation and test batches are not.
- Use the validation loader for tuning and model selection; use the test loader only for the final evaluation.
- Check the pipeline: `python -m src.data.dataset`
- Regenerate the split (only if the rules change): `python -m src.data.splits`

**References**
- CUB-200-2011 Official Dataset Website
- CUB Training Dataset — Hugging Face
- CUB Test Dataset — Hugging Face
