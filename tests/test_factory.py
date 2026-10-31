"""Model factory tests."""

from __future__ import annotations

import unittest

import torch
from torch import nn

from src.models import available_models, create_model, pretrained_models

from ._utils import NUM_CLASSES, dummy_batch

ALL_MODELS = [
    "alexnet", "vgg16", "vgg19", "googlenet",
    "resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
]


class TestFactory(unittest.TestCase):
    def test_all_nine_architectures_registered(self) -> None:
        self.assertEqual(sorted(available_models()), sorted(ALL_MODELS))

    def test_create_returns_module_with_requested_classes(self) -> None:
        model = create_model("alexnet", num_classes=7)
        self.assertIsInstance(model, nn.Module)
        last_linear = [m for m in model.modules() if isinstance(m, nn.Linear)][-1]
        self.assertEqual(last_linear.out_features, 7)

    def test_name_is_case_and_dash_insensitive(self) -> None:
        self.assertEqual(type(create_model(" AlexNet ", num_classes=NUM_CLASSES)).__name__, "AlexNet")

    def test_googlenet_alias(self) -> None:
        if "googlenet" not in available_models():
            self.skipTest("googlenet not registered yet")
        model = create_model("Inception-v1", num_classes=NUM_CLASSES, aux_logits=False)
        self.assertEqual(type(model).__name__, "GoogLeNet")

    def test_unknown_name_lists_available_models(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            create_model("resnet9000", num_classes=NUM_CLASSES)
        self.assertIn("alexnet", str(ctx.exception))

    def test_invalid_num_classes(self) -> None:
        for bad in (0, -3, 2.5, "10", True, None):
            with self.subTest(num_classes=bad):
                with self.assertRaises(ValueError):
                    create_model("alexnet", num_classes=bad)

    def test_unknown_kwarg_is_rejected(self) -> None:
        with self.assertRaises(TypeError):
            create_model("alexnet", num_classes=NUM_CLASSES, not_an_option=1)

    def test_vgg_batch_norm_option(self) -> None:
        if "vgg16" not in available_models():
            self.skipTest("vgg16 not registered yet")
        plain = create_model("vgg16", num_classes=NUM_CLASSES)
        with_bn = create_model("vgg16", num_classes=NUM_CLASSES, batch_norm=True)
        self.assertFalse(any(isinstance(m, nn.BatchNorm2d) for m in plain.modules()))
        self.assertEqual(sum(isinstance(m, nn.BatchNorm2d) for m in with_bn.modules()), 13)

    def test_models_are_randomly_initialized(self) -> None:
        a = create_model("alexnet", num_classes=NUM_CLASSES)
        b = create_model("alexnet", num_classes=NUM_CLASSES)
        self.assertFalse(all((pa == pb).all() for pa, pb in zip(a.parameters(), b.parameters())))


class TestPretrained(unittest.TestCase):
    """pretrained=False builds the same network without downloading the ImageNet weights."""

    def test_registered_apart_from_the_scratch_models(self) -> None:
        self.assertEqual(pretrained_models(), ["resnet50_pretrained"])
        self.assertNotIn("resnet50_pretrained", available_models())

    def test_new_last_layer_outputs_num_classes(self) -> None:
        model = create_model("resnet50_pretrained", num_classes=NUM_CLASSES, pretrained=False).eval()
        images, _ = dummy_batch()
        with torch.no_grad():
            self.assertEqual(tuple(model(images).shape), (images.shape[0], NUM_CLASSES))
        self.assertEqual(model.fc.in_features, 2048)

    def test_fine_tuning_trains_every_layer(self) -> None:
        model = create_model("resnet50_pretrained", num_classes=NUM_CLASSES, pretrained=False)
        self.assertTrue(all(p.requires_grad for p in model.parameters()))

    def test_freeze_backbone_trains_only_the_last_layer(self) -> None:
        model = create_model("resnet50_pretrained", num_classes=NUM_CLASSES, pretrained=False,
                             freeze_backbone=True)
        trainable = [name for name, p in model.named_parameters() if p.requires_grad]
        self.assertEqual(trainable, ["fc.weight", "fc.bias"])

    def test_invalid_num_classes(self) -> None:
        with self.assertRaises(ValueError):
            create_model("resnet50_pretrained", num_classes=0, pretrained=False)


if __name__ == "__main__":
    unittest.main()
