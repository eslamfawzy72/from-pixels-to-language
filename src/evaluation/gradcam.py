"""Grad-CAM (Selvaraju et al., ICCV 2017): which image regions support a class score.

For a class c and the feature maps A^k of a convolutional layer:

    weights a_k = global average of d(logit_c) / d(A^k)
    heatmap     = ReLU(sum_k a_k * A^k), upsampled to the input size and scaled to 0-1

Works with every model from ``src.models.create_model`` (from-scratch and pretrained):

    cam = GradCAM(model)                       # the target layer is found automatically
    result = cam(images)                       # explains each image's predicted class
    result = cam(images, target_classes=labels)  # explains the given classes instead
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple, Optional, Sequence, Union

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset

PathLike = Union[str, Path]


def find_target_layer(model: nn.Module) -> nn.Module:
    """Return the module whose output is the model's last convolutional feature map.

        ResNet (ours and torchvision's):  last residual block of ``layer4``
        GoogLeNet:                        ``inception5b``
        VGG, AlexNet (ours and torchvision's): the last conv + ReLU, ``features[-2]``

    VGG and AlexNet end ``features`` with a max pool, which is skipped because it halves
    the heatmap's resolution (at 224 input: 14x14 instead of 7x7 for VGG, 13x13 instead of
    6x6 for AlexNet).

    Raises:
        ValueError: If the model has none of these layouts; pass ``target_layer`` to
            :class:`GradCAM` for such a model.
    """
    if hasattr(model, "layer4"):
        return model.layer4[-1]
    if hasattr(model, "inception5b"):
        return model.inception5b
    features = getattr(model, "features", None)
    if isinstance(features, nn.Sequential) and isinstance(features[-1], nn.MaxPool2d):
        return features[-2]
    raise ValueError(f"No known Grad-CAM target layer for {type(model).__name__}; pass target_layer explicitly")


class GradCAMResult(NamedTuple):
    """One entry per image; all tensors are on the CPU."""

    heatmaps: torch.Tensor        # [N, H, W] in 0-1, same H and W as the input
    target_classes: torch.Tensor  # [N] the class each heatmap explains
    predictions: torch.Tensor     # [N] the model's predicted class
    confidences: torch.Tensor     # [N] softmax probability of the predicted class


class GradCAM:
    """Compute Grad-CAM heatmaps for a batch of images.

    Args:
        model: The classifier. It runs in eval mode during the call and then returns to
            its previous mode; its parameters' ``.grad`` are not touched and no hook stays
            registered.
        target_layer: The layer to explain; default :func:`find_target_layer`.
    """

    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None) -> None:
        self.model = model
        self.target_layer = find_target_layer(model) if target_layer is None else target_layer

    def __call__(
        self,
        images: torch.Tensor,
        target_classes: Optional[Union[torch.Tensor, Sequence[int]]] = None,
    ) -> GradCAMResult:
        """Explain ``target_classes`` (default: each image's predicted class) for ``images`` ``[N, C, H, W]``."""
        captured = {}

        def capture(_module: nn.Module, _inputs: tuple, output: torch.Tensor) -> None:
            captured["activations"] = output

        was_training = self.model.training
        handle = self.target_layer.register_forward_hook(capture)
        self.model.eval()
        try:
            with torch.enable_grad():
                # Requiring gradients on the input keeps the graph when the backbone is frozen.
                logits = self.model(images.detach().clone().requires_grad_(True))
                predictions = logits.argmax(dim=1)
                targets = predictions if target_classes is None else torch.as_tensor(target_classes, device=logits.device)
                # In eval mode the images do not affect each other, so the gradient of the
                # sum gives every image its own gradient. Raw logits, as in the paper.
                score = logits.gather(1, targets.view(-1, 1)).sum()
                activations = captured["activations"]
                gradients = torch.autograd.grad(score, activations)[0]
        finally:
            handle.remove()
            self.model.train(was_training)

        weights = gradients.mean(dim=(2, 3), keepdim=True)
        heatmaps = F.relu((weights * activations.detach()).sum(dim=1, keepdim=True))
        heatmaps = F.interpolate(heatmaps, size=images.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
        # Scaled by the peak only, so a map without positive evidence stays all zero.
        peaks = heatmaps.flatten(1).amax(dim=1).view(-1, 1, 1)
        heatmaps = heatmaps / peaks.clamp_min(1e-12)

        confidences = torch.softmax(logits.detach(), dim=1).gather(1, predictions.view(-1, 1)).squeeze(1)
        return GradCAMResult(heatmaps.cpu(), targets.cpu(), predictions.cpu(), confidences.cpu())


def colorize_heatmap(heatmap: np.ndarray, colormap: str = "inferno") -> np.ndarray:
    """Map a heatmap ``[H, W]`` in 0-1 to an RGB array ``[H, W, 3]`` in 0-1."""
    from matplotlib import colormaps

    return colormaps[colormap](heatmap)[..., :3]


def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.7, colormap: str = "inferno") -> np.ndarray:
    """Blend a heatmap ``[H, W]`` over an RGB image ``[H, W, 3]``, both in 0-1.

    The blend is proportional to the heatmap value, so regions without evidence keep
    their original colors instead of being tinted by the colormap's low end.
    """
    blend = alpha * heatmap[..., None]
    return (1 - blend) * image + blend * colorize_heatmap(heatmap, colormap)


def save_gradcam_figure(
    path: PathLike,
    image: np.ndarray,
    heatmap: np.ndarray,
    actual: str,
    predicted: str,
    confidence: float,
    explained: Optional[str] = None,
    colormap: str = "inferno",
) -> None:
    """Save original image, heatmap and overlay side by side with the labels and confidence.

    ``image`` and ``heatmap`` are as in :func:`overlay_heatmap`; ``explained`` is the class
    name the heatmap explains (default: ``predicted``).
    """
    from matplotlib.figure import Figure  # no pyplot: works without a display and keeps no global state

    figure = Figure(figsize=(10.5, 4.8), layout="constrained")
    axes = figure.subplots(1, 3)
    panels = (("Original image", image, {}),
              ("Grad-CAM heatmap", heatmap, {"cmap": colormap, "vmin": 0, "vmax": 1}),
              ("Overlay", overlay_heatmap(image, heatmap, colormap=colormap), {}))
    for axis, (title, data, options) in zip(axes, panels):
        shown = axis.imshow(data, **options)
        axis.set_title(title, fontsize=11)
        axis.set_axis_off()
        if options:
            heatmap_image = shown
    colorbar = figure.colorbar(heatmap_image, ax=list(axes), orientation="horizontal", shrink=0.3, aspect=30)
    colorbar.set_ticks([0, 1], labels=["low", "high"])
    colorbar.set_label("Evidence for the explained class (relative to this image's peak)", fontsize=9)
    colorbar.ax.tick_params(labelsize=9)
    figure.suptitle(f"Actual: {actual}   |   Predicted: {predicted}\n"
                    f"Confidence: {confidence:.1%}   |   Heatmap explains: {explained or predicted}", fontsize=12)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150, facecolor="white")


class Example(NamedTuple):
    """One classified image of a dataset."""

    index: int         # position in the dataset
    actual: int
    predicted: int
    confidence: float  # softmax probability of the predicted class


def select_examples(
    model: nn.Module,
    dataset: Dataset,
    count: int = 5,
    device: Union[str, torch.device] = "cpu",
    batch_size: int = 64,
    num_workers: int = 2,
    seed: int = 42,
) -> tuple[list[Example], list[Example]]:
    """Pick up to ``count`` correctly and ``count`` incorrectly classified examples.

    The images are visited in a seeded random order, because the CUB splits are sorted by
    species and the first hits would all be one species. Each list prefers examples of
    different species and repeats a species only if it would otherwise fall short. The scan
    stops as soon as both lists are full.
    """
    order = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed)).tolist()
    loader = DataLoader(Subset(dataset, order), batch_size=batch_size, num_workers=num_workers)
    picked: dict[bool, list[Example]] = {True: [], False: []}
    spare: dict[bool, list[Example]] = {True: [], False: []}
    species: dict[bool, set[int]] = {True: set(), False: set()}

    was_training = model.training
    model.eval()
    position = 0
    with torch.inference_mode():
        for images, labels in loader:
            values, predictions = torch.softmax(model(images.to(device)), dim=1).max(dim=1)
            for actual, predicted, confidence in zip(labels.tolist(), predictions.tolist(), values.tolist()):
                example = Example(order[position], actual, predicted, confidence)
                position += 1
                correct = actual == predicted
                if len(picked[correct]) < count and actual not in species[correct]:
                    picked[correct].append(example)
                    species[correct].add(actual)
                else:
                    spare[correct].append(example)
            if all(len(group) == count for group in picked.values()):
                break
    model.train(was_training)

    for correct, group in picked.items():
        group.extend(spare[correct][:count - len(group)])
    return picked[True], picked[False]
