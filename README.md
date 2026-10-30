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
tests/                        # Synthetic-data correctness tests (unittest)
results/                      # Generated benchmark output (not source code)
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

## CNN Architectures (Classification Encoder)

Nine architectures from the course lecture *CNN Architectures* are implemented **from scratch** in PyTorch (random initialization, no pretrained weights, no torchvision models):

| Name | Family | Notes |
|---|---|---|
| `alexnet` | AlexNet | Lecture layer table (conv1 64×11×11/4, …, fc 4096-4096) |
| `vgg16`, `vgg19` | VGG | 3×3 convs, 2×2 pools, stages (2,2,3,3,3) / (2,2,4,4,4) |
| `googlenet` | GoogLeNet / Inception v1 | Inception modules, global average pooling, optional auxiliary classifiers |
| `resnet18`, `resnet34` | ResNet | Basic blocks |
| `resnet50`, `resnet101`, `resnet152` | ResNet | Bottleneck blocks |

Architecture details, deviations from the lecture and the data-integration contract are in [src/models/README_cnn_architectures.md](src/models/README_cnn_architectures.md).

### Code layout

```text
src/models/
├── blocks.py        # ConvBlock, InceptionModule, InceptionAux, BasicBlock, Bottleneck, weight init
├── alexnet.py       # AlexNet
├── vgg.py           # Config-driven VGG builder (vgg16, vgg19)
├── googlenet.py     # GoogLeNet + GoogLeNetOutputs
├── resnet.py        # Config-driven ResNet builder (resnet18 … resnet152)
├── factory.py       # create_model(name, num_classes, **kwargs)
└── utils.py         # count_parameters
src/training/
├── config.py        # TrainConfig dataclass
├── losses.py        # Loss handling for models with auxiliary outputs
├── train.py         # set_seed, build_optimizer, train_one_epoch, fit
├── validate.py      # validate (no gradient updates)
└── checkpoint.py    # save_checkpoint / load_checkpoint
src/evaluation/
├── metrics.py       # accuracy
└── benchmark.py     # parameter counts, latency, peak memory (CUDA)
```

### Creating a model

```python
import torch
from src.models import available_models, create_model, count_parameters

print(available_models())
# ['alexnet', 'vgg16', 'vgg19', 'googlenet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']

model = create_model("resnet18", num_classes=200)
logits = model(torch.randn(8, 3, 224, 224))      # [8, 200], raw logits (no softmax)
features = model.forward_features(torch.randn(8, 3, 224, 224))  # last conv feature map
total, trainable = count_parameters(model)
```

Common options (passed as keyword arguments to `create_model`):

| Option | Models | Default | Meaning |
|---|---|---|---|
| `in_channels` | all | `3` | Use `1` for grayscale input |
| `dropout` | AlexNet, VGG, GoogLeNet | 0.5 / 0.5 / 0.4 | Classifier dropout |
| `batch_norm` | VGG, GoogLeNet | `False` | Add BatchNorm after every conv (not in the original networks) |
| `aux_logits` | GoogLeNet | `True` | Build the two auxiliary classifiers |

Model names are case-insensitive; `inception_v1` is an alias for `googlenet`. Unknown names or an invalid `num_classes` raise `ValueError`.

**GoogLeNet output exception:** with `aux_logits=True` and `model.train()`, GoogLeNet returns `GoogLeNetOutputs(logits, aux1, aux2)`; in eval mode (or with `aux_logits=False`) it returns plain logits like every other model. The training utilities handle both automatically.

### Running the correctness tests

The tests use random tensors only (no dataset needed) and the standard-library `unittest`:

```bash
python -m unittest -v                                   # everything
python -m unittest tests.test_shapes -v                 # one file
```

Environment variables narrow the run on slower machines:

```powershell
# PowerShell
$env:TEST_MODELS = "alexnet,resnet18"; python -m unittest -v
```

```bash
# bash
TEST_MODELS=alexnet,resnet18 TEST_IMAGE_SIZE=160 python -m unittest -v
```

Supported variables: `TEST_MODELS`, `TEST_BATCH_SIZE` (default 2), `TEST_NUM_CLASSES` (default 10), `TEST_IMAGE_SIZE` (default 224). These values are for technical testing only and do not reflect the real dataset.

> VGG-16/19 have ~140M parameters; a backward pass needs more than 1 GB of free RAM. The per-model backward test runs each model in its own subprocess so memory is released between models.

### Running the technical benchmark

```bash
python -m src.evaluation.benchmark                                  # all models, CPU or GPU (auto)
python -m src.evaluation.benchmark --models resnet18 vgg16 --batch-size 8 --device cuda
```

Results are written to `results/benchmark_<device>.json` (with the hardware/software environment) and `.csv`. The benchmark measures parameter counts and eval-mode forward latency (with warm-up and CUDA synchronization); peak memory is reported on CUDA only. `--num-classes` defaults to 1000 so parameter counts are comparable with the lecture's ImageNet figures. These are cost measurements, **not** accuracy results.

### Training with a data loader

The training code does not load or preprocess data. It accepts any iterable (normally a `torch.utils.data.DataLoader`) yielding `(images, labels)` with `images` as a float tensor `[N, C, H, W]` and `labels` as a long tensor `[N]` of class indices.

```python
from src.models import create_model
from src.training import TrainConfig, fit, set_seed

config = TrainConfig(model_name="resnet18", num_classes=200, epochs=30,
                     optimizer="sgd", learning_rate=0.01, batch_size=32, seed=42)
set_seed(config.seed)                                  # reproducible weight initialization
model = create_model(config.model_name, num_classes=config.num_classes, **config.model_kwargs)

result = fit(model, train_loader, val_loader, config, run_dir="outputs/runs/resnet18",
             on_epoch_end=lambda epoch, metrics: print(epoch, metrics))
```

`fit` writes `last.pt`, `best.pt` (selected by `config.monitor`: `val_accuracy` or `val_loss`) and `history.json` (config plus per-epoch metrics) into `run_dir`. The `on_epoch_end` hook is where an experiment tracker such as MLflow or W&B can log metrics. Lower-level pieces (`train_one_epoch`, `validate`, `save_checkpoint`, `load_checkpoint`, `build_optimizer`) can be used directly.

### Known limitations

- No real-data training or accuracy results yet; integration with the preprocessing pipeline is pending.
- No learning-rate scheduler, early stopping or mixed precision yet.
- Peak memory is only measured on CUDA.
- Training VGG and GoogLeNet without BatchNorm from scratch on ~6k images may be unstable; `batch_norm=True` is available as a documented deviation.

**References**
- CUB-200-2011 Official Dataset Website
- CUB Training Dataset — Hugging Face
- CUB Test Dataset — Hugging Face
