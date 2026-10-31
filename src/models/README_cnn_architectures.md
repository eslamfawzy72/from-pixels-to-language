# CNN Architectures: Implementation Notes and Integration Contract

This document explains how the nine classification architectures in `src/models/` relate to the course lecture **"CSE 485 Deep Learning, CNN Architectures" (Prof. Mahmoud Khalil)**. It lists every assumption made where the lecture does not specify a detail, and defines what the models need from the preprocessing pipeline.

All models are randomly initialized and built only from PyTorch layers (`nn.Conv2d`, `nn.BatchNorm2d`, `nn.Linear`, pooling, ...). No pretrained weights and no `torchvision.models` code are used.

## 1. What the lecture specifies vs. what was assumed

### AlexNet (`alexnet.py`)

| Item | Source | Implementation |
|---|---|---|
| conv1 64×11×11 /4 pad 2, pool 3×3/2, conv2 192×5×5 pad 2, pool, conv3 384, conv4 256, conv5 256 (3×3 pad 1), pool → 256×6×6 = 9216 | Lecture table | Exact |
| fc6 4096, fc7 4096, fc8 classes; ReLU | Lecture table | Exact (fc8 = `num_classes`) |
| Local Response Normalization | Lecture: "not used anymore" | Omitted |
| Dropout 0.5 after fc6 and fc7 | **Assumption**, from the original paper | Configurable `dropout` |
| Adaptive average pool to 6×6 before flatten | **Design choice** | Identity at 224/227 input; allows other sizes |

Per-layer parameter counts match the lecture table (23,296 / 307,392 / 663,936 / 884,992 / 590,080 / 37,752,832 / 16,781,312 / 4,097,000). The total is 61,100,840 at 1000 classes.

### VGG-16 / VGG-19 (`vgg.py`)

| Item | Source | Implementation |
|---|---|---|
| All convs 3×3 stride 1 pad 1; all max pools 2×2 stride 2 | Lecture design rules | Exact |
| Channels 64→128→256→512→512, 5 stages | Lecture | Exact |
| Convs per stage: VGG-16 (2,2,3,3,3), VGG-19 (2,2,4,4,4) | Lecture | Exact; one builder driven by `VGG_STAGE_DEPTHS` |
| FC 4096 → FC 4096 → FC classes | Lecture | Exact |
| Dropout 0.5 in fc6/fc7 | **Assumption**, original paper | Configurable |
| BatchNorm | Not in lecture/paper | Off by default; `batch_norm=True` adds it |
| Adaptive average pool to 7×7 | **Design choice** | Identity at 224; keeps the 25,088-d flatten |

Parameter counts at 1000 classes: VGG-16 138,357,544; VGG-19 143,667,240. Both match the published reference counts.

### GoogLeNet / Inception v1 (`googlenet.py`)

| Item | Source | Implementation |
|---|---|---|
| Stem: conv 7×7/2 (64) → maxpool 3×3/2 → conv 1×1 (64) → conv 3×3 (192) → maxpool 3×3/2, 224 → 28 | Lecture stem table | Exact |
| Inception module: 1×1 ∥ 1×1→3×3 ∥ 1×1→5×5 ∥ 3×3 maxpool→1×1, concatenated | Lecture | Exact |
| Global average pooling + one linear layer | Lecture | Exact (adaptive pool to 1×1) |
| Auxiliary classifiers at intermediate points | Lecture (concept only) | Two heads, after inception 4a and 4d |
| Channel widths of modules 3a–5b | **Assumption**, Szegedy et al. 2015, Table 1 | `INCEPTION_SPECS` |
| Pools between stages 3/4 and 4/5: 3×3/2 pad 1 | **Assumption**, paper | 28 → 14 → 7 |
| Aux head: avgpool 5×5/3 → 1×1 conv 128 → FC 1024 → dropout 0.7 → FC classes | **Assumption**, paper | `InceptionAux` |
| Main dropout 0.4; auxiliary loss weight 0.3 | **Assumption**, paper | `dropout`, `TrainConfig.aux_weight` |
| LRN | Not in lecture stem table | Omitted |
| BatchNorm | Not in the original network | Off by default; `batch_norm=True` adds it |

The main network has 6,998,552 parameters at 1000 classes (13,378,280 with both auxiliary heads). The paper's Table 1 adds up to about 6.8M, but that table is approximate (it omits biases and the stem's 1×1 conv). torchvision's GoogLeNet is not a valid reference here because it uses BatchNorm and a 3×3 conv in the "5×5" branch.

**Output contract exception:** in training mode with `aux_logits=True`, `forward` returns `GoogLeNetOutputs(logits, aux1, aux2)`. In eval mode, or with `aux_logits=False`, it returns only the main logits. `src/training/losses.classification_loss` computes `loss = main + 0.3 × (aux1 + aux2)`. Accuracy and evaluation always use the main logits only. The lecture notes that auxiliary classifiers are unnecessary once BatchNorm is used, so `aux_logits=False` with `batch_norm=True` is a reasonable experiment.

### ResNet-18/34/50/101/152 (`resnet.py`, `blocks.py`)

| Item | Source | Implementation |
|---|---|---|
| Stem: conv 7×7/2 (64) + maxpool 3×3/2, 224 → 56 | Lecture | Exact (+ BN + ReLU after the conv) |
| 4 stages, C = 64/128/256/512; first block of stages 2–4 halves resolution (stride 2) and doubles channels | Lecture | Exact |
| Basic block: two 3×3 convs; bottleneck: 1×1 (4C→C), 3×3 (C→C), 1×1 (C→4C) | Lecture | Exact |
| Block order conv-BN-ReLU-conv-BN, add, ReLU (original, post-activation) | Lecture "Original ResNet block" | Exact |
| Blocks per stage: 18 (2,2,2,2), 34 (3,4,6,3), 50 (3,4,6,3), 101 (3,4,23,3), 152 (3,8,36,3) | Lecture table | Exact (`RESNET_SPECS`) |
| Global average pooling + single linear layer | Lecture | Exact |
| Shortcut when shape changes: 1×1 conv (stride s) + BN | **Assumption**, He et al. option B | Identity otherwise |
| Stride position in bottleneck blocks: on the 3×3 conv | **Assumption** ("ResNet v1.5"); the original paper strides the first 1×1 | Does not change parameter counts |
| Pre-activation blocks, ResNeXt | Shown in lecture as later improvements | Not implemented (not required) |

Parameter counts at 1000 classes match the published reference counts: 11,689,512 / 21,797,672 / 25,557,032 / 44,549,160 / 60,192,808.

### Shared choices

- **Initialization:** Kaiming-normal for convs (fan-out, ReLU), BatchNorm weight 1 / bias 0, Linear N(0, 0.01). Conv biases are dropped when a BatchNorm follows.
- **Outputs are raw logits.** No softmax inside the models; use `nn.CrossEntropyLoss`.
- **Encoder + head.** Every model is `ImageClassifier(encoder, head)` (`classifier.py`). Each architecture file defines only the encoder (conv layers -> last feature map); the fully connected part is a head in `classifier.py` (`FCHead` for AlexNet/VGG, `LinearHead` for ResNet, `GoogLeNetHead` for GoogLeNet). `model.encoder` (or `build_encoder(name)` in `encoder.py`) is the hook for reusing the encoder in the colorization and captioning tasks and for choosing a Grad-CAM target layer. The layers and parameter counts are unchanged by the split.
- **`in_channels`** is configurable on every model (for example `1` for grayscale input).

## 2. Input-size behaviour

The lecture's reference sizes are 227×227 for AlexNet and 224×224 for the rest. Adaptive pooling before each classifier keeps the lecture's dimensions at those sizes and lets the models accept other sizes without code changes.

| Model | Feature map at 224 | Notes |
|---|---|---|
| AlexNet | 256×6×6 | 227 and 224 both give 6×6 after pool5 |
| VGG-16/19 | 512×7×7 | |
| GoogLeNet | 1024×7×7 | With `aux_logits=True`, training inputs must be at least 72×72 (64×64 fails in the auxiliary 5×5 pool) |
| ResNet-18/34 | 512×7×7 | |
| ResNet-50/101/152 | 2048×7×7 | |

All nine models were also tested at 160×160. Sizes much smaller than that have not been tested systematically.

## 3. Integration note for the preprocessing pipeline

The models and the training loop are independent of the dataset. To connect them, the preprocessing side needs to provide the following.

**Data loader contract**
- Training and validation loaders (normally `torch.utils.data.DataLoader`) that yield `(images, labels)` tuples. If the Hugging Face dataset returns dicts, a `collate_fn` should convert each batch to this tuple.
- `images`: `torch.float32`, shape `[N, C, H, W]`, already resized, normalized and augmented (augmentation on the training split only).
- `labels`: `torch.int64`, shape `[N]`, class indices in `[0, num_classes - 1]` (not one-hot).

**Information to confirm**
1. `num_classes`: expected 200 for CUB-200-2011.
2. Final input resolution (for example 224×224) and channel count (3 for RGB; 1 if grayscale input is used for the colorization encoder).
3. Normalization statistics: ImageNet mean/std, or statistics computed on the training split.
4. Split strategy: the dataset has only train and test splits, so a validation split must be carved from train (stratified by class, fixed seed). The test split must not be used for model selection.
5. Batch size that fits the target GPU (VGG at 224 is the most memory-hungry).

**Integration steps**
1. Build `train_loader` and `val_loader` that satisfy the contract above.
2. Create a `TrainConfig` with the confirmed `num_classes` (and `batch_size` for the record).
3. Call `set_seed(config.seed)`, then `create_model(config.model_name, config.num_classes, **config.model_kwargs)`.
4. Call `fit(model, train_loader, val_loader, config, run_dir=...)`; optionally pass `on_epoch_end` to log to MLflow or W&B.
5. Evaluate the `best.pt` checkpoint on the held-out test split with `load_checkpoint` + `validate`.
6. Re-run `python -m src.evaluation.benchmark --num-classes 200 --image-size <final size>` on the training GPU to record costs with the real settings.
