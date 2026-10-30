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
├── config.py        # TrainConfig dataclass (optimizer, LR schedule, early stopping, ...)
├── losses.py        # Loss handling for models with auxiliary outputs
├── train.py         # set_seed, build_optimizer, build_scheduler, train_one_epoch, fit
├── validate.py      # validate (no gradient updates)
├── checkpoint.py    # save_checkpoint / load_checkpoint
└── tracking.py      # MLflowTracker (params, per-epoch metrics, artifacts)
src/evaluation/
├── metrics.py       # accuracy
└── benchmark.py     # parameter counts, latency, peak memory (CUDA)
src/tasks/
└── classification.py  # train / evaluate / predict: wires data + model + training + tracking
src/cli.py           # command line interface
configs/classification.yaml  # default classification settings
```

### Command line interface

All training, evaluation and inference runs go through `src/cli.py`. Settings come from [configs/classification.yaml](configs/classification.yaml); shortcut flags and `--set section.key=value` override any value.

```bash
# Train (defaults from the YAML file: resnet18, 60 epochs, SGD + cosine schedule, MLflow on)
python -m src.cli train --task classification --config configs/classification.yaml

# Change the model / hyperparameters from the command line
python -m src.cli train --task classification --model googlenet --epochs 80 --lr 0.05 \
    --set train.model_kwargs.batch_norm=true --set train.model_kwargs.aux_logits=false

# Evaluate the best checkpoint on the held-out test split (writes test_metrics.json next to it)
python -m src.cli evaluate --task classification --checkpoint outputs/runs/<run_name>/best.pt --split test

# Predict the species of one or more images (top-5)
python -m src.cli predict --task classification --checkpoint outputs/runs/<run_name>/best.pt --image bird.jpg
```

Useful flags for `train`: `--model`, `--epochs`, `--lr`, `--batch-size`, `--optimizer`, `--scheduler`, `--seed`, `--run-name`, `--output-dir`, `--experiment`, `--tracking-dir`, `--tracking-uri`, `--no-tracking`, `--set`. Run `python -m src.cli train --help` for the full list.

Each run is saved to `outputs/runs/<run_name>/` (`best.pt`, `last.pt`, `history.json`). Checkpoints also store the class names and input size, so `predict` works without downloading the dataset.

### Experiment tracking (MLflow)

With `tracking.enabled: true` (the default), every training run is logged to MLflow: all hyperparameters, parameter counts, per-epoch `train_loss`, `train_accuracy`, `val_loss`, `val_accuracy` and `lr` (shown as curves), the best epoch, `history.json`, and the checkpoint path. Set `tracking.log_checkpoints: true` to upload the `.pt` files too (VGG checkpoints exceed 1 GB).

Runs are stored in one tracking folder (`tracking.dir`, default `outputs/mlflow/`) holding an SQLite database `mlflow.db` and an `artifacts/` folder. (Recent MLflow versions no longer accept the old `./mlruns` folder backend.) Open the MLflow UI with:

```bash
mlflow ui --backend-store-uri sqlite:///outputs/mlflow/mlflow.db   # then open http://127.0.0.1:5000
```

On Colab, keep both the run folders and the tracking folder on Google Drive so they survive the session:

```bash
python -m src.cli train --task classification \
    --output-dir /content/drive/MyDrive/from-pixels-to-language/runs \
    --tracking-dir /content/drive/MyDrive/from-pixels-to-language/mlflow
```

To browse those runs on your PC, download the Drive `mlflow` folder and run `mlflow ui --backend-store-uri sqlite:///<path>/mlflow/mlflow.db`. Artifact links point to the Drive paths; the metrics and parameters are all inside `mlflow.db`.

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

### Training from Python (without the CLI)

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

`fit` writes `last.pt`, `best.pt` (selected by `config.monitor`: `val_accuracy` or `val_loss`) and `history.json` (config plus per-epoch metrics) into `run_dir`. The `on_epoch_end` hook is where `MLflowTracker.log_epoch` plugs in. Lower-level pieces (`train_one_epoch`, `validate`, `save_checkpoint`, `load_checkpoint`, `build_optimizer`, `build_scheduler`) can be used directly.

Training options in `TrainConfig` (and the `train:` section of the YAML file):

| Option | Values | Meaning |
|---|---|---|
| `optimizer` | `sgd`, `adam`, `adamw` | With `learning_rate`, `momentum` (SGD), `weight_decay` |
| `scheduler` | `none`, `cosine`, `step`, `plateau` | Per-epoch LR schedule; `min_lr` (cosine), `step_size`/`gamma` (step), `plateau_patience`/`gamma` (plateau) |
| `warmup_epochs` | int | Linear LR warm-up before the schedule (not with `plateau`) |
| `label_smoothing` | 0 – <1 | Applied to the training loss only; validation loss stays plain cross-entropy |
| `early_stopping_patience` | int or `null` | Stop after this many epochs without improvement in `monitor` |
| `aux_weight` | float | Weight of GoogLeNet's auxiliary losses (default 0.3) |

### Known limitations

- No accuracy results reported yet; only a 3-epoch sanity run (ResNet-18, ~2% val accuracy) has been done on Colab.
- No mixed precision; no resuming from `last.pt`.
- Peak memory is only measured on CUDA.
- Training VGG and GoogLeNet without BatchNorm from scratch on ~6k images may be unstable; `batch_norm=True` is available as a documented deviation.

**References**
- CUB-200-2011 Official Dataset Website
- CUB Training Dataset — Hugging Face
- CUB Test Dataset — Hugging Face
