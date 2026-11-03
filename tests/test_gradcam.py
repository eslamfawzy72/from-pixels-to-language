"""Grad-CAM: target layers of every model, heatmap correctness, figures and example selection."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset

from src.evaluation.gradcam import (GradCAM, colorize_heatmap, find_target_layer, overlay_heatmap,
                                    save_gradcam_figure, select_examples)
from src.models import create_model, pretrained_models
from tests._utils import models_under_test

NUM_CLASSES = 5
IMAGE_SIZE = 96  # small, so that VGG and ResNet-152 stay fast; all architectures accept it

# Output of the target layer for a 96x96 input: (channels, spatial size).
EXPECTED_FEATURE_MAPS = {
    "alexnet": (256, 5), "alexnet_pretrained": (256, 5),
    "vgg16": (512, 6), "vgg19": (512, 6), "vgg16_pretrained": (512, 6),
    "googlenet": (1024, 3),
    "resnet18": (512, 3), "resnet34": (512, 3), "resnet18_pretrained": (512, 3),
    "resnet50": (2048, 3), "resnet101": (2048, 3), "resnet152": (2048, 3), "resnet50_pretrained": (2048, 3),
}


def build(name: str, **kwargs) -> nn.Module:
    if name in pretrained_models():
        kwargs.setdefault("pretrained", False)  # the tests never download weights
    return create_model(name, num_classes=NUM_CLASSES, **kwargs).eval()


class TwoRegionModel(nn.Module):
    """Global average pooling + linear, where class c reads exactly feature channel c."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 2, kernel_size=1, bias=False)
        self.fc = nn.Linear(2, 2, bias=False)
        nn.init.dirac_(self.conv.weight)
        nn.init.eye_(self.fc.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv(x).mean(dim=(2, 3)))


class TestTargetLayers(unittest.TestCase):
    def test_every_model(self) -> None:
        for name in [*models_under_test(), *pretrained_models()]:
            with self.subTest(model=name):
                model = build(name)
                captured = []
                handle = find_target_layer(model).register_forward_hook(lambda _m, _i, out: captured.append(out))
                with torch.no_grad():
                    model(torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE))
                handle.remove()
                channels, size = EXPECTED_FEATURE_MAPS[name]
                self.assertEqual(captured[-1].shape, (1, channels, size, size))

                result = GradCAM(model)(torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE))
                self.assertEqual(result.heatmaps.shape, (2, IMAGE_SIZE, IMAGE_SIZE))
                self.assertTrue(torch.isfinite(result.heatmaps).all())
                self.assertGreaterEqual(result.heatmaps.min().item(), 0)
                self.assertLessEqual(result.heatmaps.max().item(), 1)
                del model

    def test_unknown_model_needs_an_explicit_layer(self) -> None:
        with self.assertRaises(ValueError):
            find_target_layer(nn.Linear(2, 2))


class TestHeatmaps(unittest.TestCase):
    def setUp(self) -> None:
        # Channel 0 is active in the left half, channel 1 in the right half.
        self.images = torch.zeros(1, 2, 8, 8)
        self.images[0, 0, :, :4] = 1
        self.images[0, 1, :, 4:] = 2
        self.model = TwoRegionModel()
        self.cam = GradCAM(self.model, target_layer=self.model.conv)

    def test_heatmap_is_class_specific(self) -> None:
        left = self.cam(self.images, [0]).heatmaps[0]
        right = self.cam(self.images, [1]).heatmaps[0]
        self.assertGreater(left[:, :2].mean(), left[:, 6:].mean())
        self.assertGreater(right[:, 6:].mean(), right[:, :2].mean())
        self.assertAlmostEqual(left.max().item(), 1, places=5)

    def test_default_target_is_the_prediction(self) -> None:
        result = self.cam(self.images)  # channel 1 is stronger
        self.assertEqual(result.predictions.tolist(), [1])
        self.assertEqual(result.target_classes.tolist(), [1])
        self.assertTrue(torch.equal(result.heatmaps, self.cam(self.images, [1]).heatmaps))
        self.assertAlmostEqual(result.confidences.item(), torch.softmax(torch.tensor([0.5, 1.0]), 0)[1].item(), places=5)

    def test_no_positive_evidence_gives_a_zero_map(self) -> None:
        with torch.no_grad():
            self.model.fc.weight.copy_(-torch.eye(2))
        heatmap = self.cam(self.images, [0]).heatmaps
        self.assertTrue(torch.equal(heatmap, torch.zeros_like(heatmap)))

    def test_batch_matches_single_images(self) -> None:
        model = build("resnet18")
        images = torch.randn(3, 3, IMAGE_SIZE, IMAGE_SIZE)
        cam = GradCAM(model)
        batch = cam(images, [0, 1, 2]).heatmaps
        for i in range(3):
            single = cam(images[i:i + 1], [i]).heatmaps[0]
            self.assertTrue(torch.allclose(batch[i], single, atol=1e-4))


class TestModelStaysUntouched(unittest.TestCase):
    def test_mode_gradients_and_hooks(self) -> None:
        model = build("resnet18").train()
        layer = find_target_layer(model)
        GradCAM(model)(torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE))
        self.assertTrue(model.training)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        self.assertEqual(len(layer._forward_hooks), 0)

    def test_frozen_backbone(self) -> None:
        model = build("resnet18_pretrained", freeze_backbone=True)
        heatmaps = GradCAM(model)(torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)).heatmaps
        self.assertTrue(torch.isfinite(heatmaps).all())

    def test_googlenet_in_training_mode_returns_plain_logits(self) -> None:
        model = build("googlenet", aux_logits=True).train()
        result = GradCAM(model)(torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE))
        self.assertEqual(result.heatmaps.shape, (1, IMAGE_SIZE, IMAGE_SIZE))
        self.assertTrue(model.training)

    def test_failing_forward_removes_the_hook(self) -> None:
        model = build("resnet18")
        layer = find_target_layer(model)
        with self.assertRaises(RuntimeError):
            GradCAM(model)(torch.randn(1, 4, IMAGE_SIZE, IMAGE_SIZE))
        self.assertEqual(len(layer._forward_hooks), 0)


class TestVisualization(unittest.TestCase):
    def test_overlay(self) -> None:
        image = np.full((4, 4, 3), 0.5)
        heatmap = np.zeros((4, 4))
        heatmap[0, 0] = 1.0
        overlay = overlay_heatmap(image, heatmap)
        self.assertEqual(overlay.shape, image.shape)
        self.assertTrue(np.array_equal(overlay[3, 3], image[3, 3]))  # no evidence: the original color stays
        self.assertFalse(np.array_equal(overlay[0, 0], image[0, 0]))
        self.assertTrue(((overlay >= 0) & (overlay <= 1)).all())
        self.assertEqual(colorize_heatmap(heatmap).shape, (4, 4, 3))

    def test_figure_is_saved_in_a_new_folder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "correct" / "figure.png"
            save_gradcam_figure(path, np.zeros((8, 8, 3)), np.zeros((8, 8)), "Blue Jay", "Blue Jay", 0.5)
            self.assertGreater(path.stat().st_size, 0)


class PredictsFirstPixel(nn.Module):
    """Predicts the class stored in the first pixel of the image."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.one_hot(x[:, 0, 0, 0].long(), NUM_CLASSES).float() * 5


class TestSelectExamples(unittest.TestCase):
    def dataset(self, labels: list[int], predictions: list[int]) -> TensorDataset:
        images = torch.zeros(len(labels), 1, 2, 2)
        images[:, 0, 0, 0] = torch.tensor(predictions, dtype=torch.float32)
        return TensorDataset(images, torch.tensor(labels))

    def select(self, labels: list[int], predictions: list[int], count: int, seed: int = 0):
        return select_examples(PredictsFirstPixel(), self.dataset(labels, predictions), count, batch_size=4,
                               num_workers=0, seed=seed)

    def test_examples_are_sorted_by_outcome_and_recorded_correctly(self) -> None:
        labels = [0, 0, 1, 1, 2, 2, 3, 3]
        predictions = [0, 1, 1, 2, 2, 3, 3, 4]
        correct, incorrect = self.select(labels, predictions, 4)
        self.assertEqual(len(correct), 4)
        self.assertEqual(len(incorrect), 4)
        for example in correct + incorrect:
            self.assertEqual(example.actual, labels[example.index])
            self.assertEqual(example.predicted, predictions[example.index])
            self.assertGreater(example.confidence, 0.5)
        self.assertTrue(all(e.actual == e.predicted for e in correct))
        self.assertTrue(all(e.actual != e.predicted for e in incorrect))

    def test_different_species_are_preferred_over_the_sorted_order(self) -> None:
        labels = [0] * 20 + [1] * 20 + [2] * 20  # sorted by species, like the CUB splits
        correct, _ = self.select(labels, labels, 3)
        self.assertEqual(sorted(e.actual for e in correct), [0, 1, 2])

    def test_species_repeat_only_to_reach_the_count(self) -> None:
        correct, _ = self.select([0, 0, 0, 1], [0, 0, 0, 1], 3)
        self.assertEqual(len({e.index for e in correct}), 3)
        self.assertEqual({e.actual for e in correct}, {0, 1})

    def test_too_few_examples(self) -> None:
        correct, incorrect = self.select([0, 1], [0, 1], 5)
        self.assertEqual(len(correct), 2)
        self.assertEqual(incorrect, [])

    def test_same_seed_same_examples(self) -> None:
        labels = list(range(NUM_CLASSES)) * 4
        self.assertEqual(self.select(labels, labels, 3, seed=1), self.select(labels, labels, 3, seed=1))


if __name__ == "__main__":
    unittest.main()
