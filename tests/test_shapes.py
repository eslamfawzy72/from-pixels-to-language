"""Forward-pass and feature-map shape tests on synthetic tensors."""

from __future__ import annotations

import unittest

import torch
from torch import nn

from src.models import create_model

from ._utils import BATCH_SIZE, NUM_CLASSES, dummy_batch, models_under_test

# Final feature map of forward_features() for a 224x224 input.
EXPECTED_FEATURES_224 = {
    "alexnet": (256, 6, 6),
    "vgg16": (512, 7, 7),
    "vgg19": (512, 7, 7),
    "googlenet": (1024, 7, 7),
    "resnet18": (512, 7, 7),
    "resnet34": (512, 7, 7),
    "resnet50": (2048, 7, 7),
    "resnet101": (2048, 7, 7),
    "resnet152": (2048, 7, 7),
}


def capture_shapes(model: nn.Module, modules: dict[str, nn.Module], x: torch.Tensor) -> dict[str, tuple]:
    """Run ``model(x)`` and record the output shape (without batch) of each module."""
    shapes: dict[str, tuple] = {}
    handles = [
        module.register_forward_hook(lambda _m, _i, out, key=key: shapes.__setitem__(key, tuple(out.shape[1:])))
        for key, module in modules.items()
    ]
    try:
        with torch.no_grad():
            model(x)
    finally:
        for handle in handles:
            handle.remove()
    return shapes


def requires(*names: str):
    """Skip a test unless all named models are selected for testing."""
    selected = set(models_under_test())
    return unittest.skipUnless(set(names) <= selected, f"{', '.join(names)} not selected")


class TestOutputShapes(unittest.TestCase):
    def test_logits_shape_in_eval_mode(self) -> None:
        images, _ = dummy_batch()
        for name in models_under_test():
            with self.subTest(model=name):
                model = create_model(name, num_classes=NUM_CLASSES).eval()
                with torch.no_grad():
                    logits = model(images)
                self.assertIsInstance(logits, torch.Tensor)
                self.assertEqual(tuple(logits.shape), (BATCH_SIZE, NUM_CLASSES))
                self.assertTrue(torch.isfinite(logits).all())

    def test_final_feature_map_224(self) -> None:
        images, _ = dummy_batch(image_size=224)
        for name in models_under_test():
            with self.subTest(model=name):
                model = create_model(name, num_classes=NUM_CLASSES).eval()
                with torch.no_grad():
                    features = model.forward_features(images)
                self.assertEqual(tuple(features.shape), (BATCH_SIZE, *EXPECTED_FEATURES_224[name]))

    def test_grayscale_input_channels(self) -> None:
        images, _ = dummy_batch(batch_size=1, channels=1)
        for name in models_under_test():
            with self.subTest(model=name):
                model = create_model(name, num_classes=NUM_CLASSES, in_channels=1).eval()
                with torch.no_grad():
                    self.assertEqual(tuple(model(images).shape), (1, NUM_CLASSES))

    def test_non_default_input_size(self) -> None:
        # 160x160 exercises the adaptive pooling path in every architecture.
        images, _ = dummy_batch(batch_size=1, image_size=160)
        for name in models_under_test():
            with self.subTest(model=name):
                model = create_model(name, num_classes=NUM_CLASSES).eval()
                with torch.no_grad():
                    self.assertEqual(tuple(model(images).shape), (1, NUM_CLASSES))


class TestIntermediateShapes(unittest.TestCase):
    @requires("alexnet")
    def test_alexnet_matches_lecture_table_at_227(self) -> None:
        model = create_model("alexnet", num_classes=NUM_CLASSES).eval()
        layers = {
            "conv1": model.encoder.features[0], "pool1": model.encoder.features[2],
            "conv2": model.encoder.features[3], "pool2": model.encoder.features[5],
            "conv3": model.encoder.features[6], "conv4": model.encoder.features[8],
            "conv5": model.encoder.features[10], "pool5": model.encoder.features[12],
        }
        shapes = capture_shapes(model, layers, torch.randn(1, 3, 227, 227))
        expected = {
            "conv1": (64, 56, 56), "pool1": (64, 27, 27),
            "conv2": (192, 27, 27), "pool2": (192, 13, 13),
            "conv3": (384, 13, 13), "conv4": (256, 13, 13),
            "conv5": (256, 13, 13), "pool5": (256, 6, 6),
        }
        self.assertEqual(shapes, expected)
        self.assertEqual(model.head.classifier[1].in_features, 9216)

    @requires("vgg16", "vgg19")
    def test_vgg_stage_outputs(self) -> None:
        expected_pools = [(64, 112, 112), (128, 56, 56), (256, 28, 28), (512, 14, 14), (512, 7, 7)]
        for name, convs in (("vgg16", 13), ("vgg19", 16)):
            with self.subTest(model=name):
                model = create_model(name, num_classes=NUM_CLASSES).eval()
                pools = [m for m in model.encoder.features if isinstance(m, nn.MaxPool2d)]
                conv_layers = [m for m in model.encoder.features.modules() if isinstance(m, nn.Conv2d)]
                self.assertEqual(len(conv_layers), convs)
                self.assertTrue(all(c.kernel_size == (3, 3) and c.stride == (1, 1) and c.padding == (1, 1)
                                    for c in conv_layers))
                shapes = capture_shapes(model, {str(i): p for i, p in enumerate(pools)}, torch.randn(1, 3, 224, 224))
                self.assertEqual([shapes[str(i)] for i in range(5)], expected_pools)
                self.assertEqual(model.head.classifier[1].in_features, 25088)

    @requires("googlenet")
    def test_googlenet_stem_and_inception_outputs(self) -> None:
        model = create_model("googlenet", num_classes=NUM_CLASSES).eval()
        names = ["stem", "inception3a", "inception3b", "inception4a", "inception4b", "inception4c",
                 "inception4d", "inception4e", "inception5a", "inception5b"]
        shapes = capture_shapes(model, {n: getattr(model.encoder, n) for n in names}, torch.randn(1, 3, 224, 224))
        expected = {
            "stem": (192, 28, 28),
            "inception3a": (256, 28, 28), "inception3b": (480, 28, 28),
            "inception4a": (512, 14, 14), "inception4b": (512, 14, 14), "inception4c": (512, 14, 14),
            "inception4d": (528, 14, 14), "inception4e": (832, 14, 14),
            "inception5a": (832, 7, 7), "inception5b": (1024, 7, 7),
        }
        self.assertEqual(shapes, expected)
        for n in names[1:]:
            self.assertEqual(getattr(model.encoder, n).out_channels, expected[n][0])

    @requires("googlenet")
    def test_googlenet_train_mode_returns_aux_outputs(self) -> None:
        images, _ = dummy_batch()
        model = create_model("googlenet", num_classes=NUM_CLASSES).train()
        outputs = model(images)
        self.assertEqual(type(outputs).__name__, "GoogLeNetOutputs")
        for logits in outputs:
            self.assertEqual(tuple(logits.shape), (BATCH_SIZE, NUM_CLASSES))
        # Eval mode returns only the main logits.
        model.eval()
        with torch.no_grad():
            self.assertIsInstance(model(images), torch.Tensor)

    @requires("googlenet")
    def test_googlenet_without_aux_returns_tensor_in_train_mode(self) -> None:
        images, _ = dummy_batch()
        model = create_model("googlenet", num_classes=NUM_CLASSES, aux_logits=False).train()
        self.assertIsNone(model.head.aux1)
        self.assertIsInstance(model(images), torch.Tensor)

    @requires("resnet18", "resnet34", "resnet50", "resnet101", "resnet152")
    def test_resnet_stage_outputs(self) -> None:
        for name, expansion in (("resnet18", 1), ("resnet34", 1), ("resnet50", 4), ("resnet101", 4), ("resnet152", 4)):
            with self.subTest(model=name):
                model = create_model(name, num_classes=NUM_CLASSES).eval()
                stages = {n: getattr(model.encoder, n) for n in ("stem", "layer1", "layer2", "layer3", "layer4")}
                shapes = capture_shapes(model, stages, torch.randn(1, 3, 224, 224))
                self.assertEqual(shapes["stem"], (64, 56, 56))
                for i, (width, size) in enumerate(((64, 56), (128, 28), (256, 14), (512, 7)), start=1):
                    self.assertEqual(shapes[f"layer{i}"], (width * expansion, size, size))

    @requires("resnet18", "resnet34", "resnet50", "resnet101", "resnet152")
    def test_resnet_block_counts(self) -> None:
        expected = {
            "resnet18": (2, 2, 2, 2), "resnet34": (3, 4, 6, 3), "resnet50": (3, 4, 6, 3),
            "resnet101": (3, 4, 23, 3), "resnet152": (3, 8, 36, 3),
        }
        for name, counts in expected.items():
            with self.subTest(model=name):
                model = create_model(name, num_classes=NUM_CLASSES)
                self.assertEqual(tuple(len(getattr(model.encoder, f"layer{i}")) for i in range(1, 5)), counts)

    @requires("resnet18", "resnet50")
    def test_resnet_shortcuts(self) -> None:
        from src.models.blocks import BasicBlock, Bottleneck

        identity = BasicBlock(64, 64, stride=1)
        self.assertIsInstance(identity.shortcut, nn.Identity)
        x = torch.randn(1, 64, 8, 8)
        self.assertEqual(tuple(identity(x).shape), (1, 64, 8, 8))

        downsample = BasicBlock(64, 128, stride=2)
        self.assertIsInstance(downsample.shortcut, nn.Sequential)
        self.assertEqual(tuple(downsample(x).shape), (1, 128, 4, 4))

        expand = Bottleneck(64, 64, stride=1)  # 64 -> 256 needs a projection
        self.assertIsInstance(expand.shortcut, nn.Sequential)
        self.assertEqual(tuple(expand(x).shape), (1, 256, 8, 8))

        same = Bottleneck(256, 64, stride=1)
        self.assertIsInstance(same.shortcut, nn.Identity)


if __name__ == "__main__":
    unittest.main()
