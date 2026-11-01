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

    FEATURES = {"resnet18_pretrained": 512, "resnet50_pretrained": 2048, "vgg16_pretrained": 4096,
                "alexnet_pretrained": 4096, "resnet50_article": 2048}  # inputs of the last layer
    LAST_LAYER = {"resnet18_pretrained": "fc", "resnet50_pretrained": "fc", "vgg16_pretrained": "classifier.6",
                  "alexnet_pretrained": "classifier.6", "resnet50_article": "fc"}

    def test_registered_apart_from_the_scratch_models(self) -> None:
        self.assertEqual(pretrained_models(), list(self.FEATURES))
        self.assertFalse(set(pretrained_models()) & set(available_models()))

    def test_new_last_layer_outputs_num_classes(self) -> None:
        images, _ = dummy_batch()
        for name, features in self.FEATURES.items():
            with self.subTest(model=name):
                model = create_model(name, num_classes=NUM_CLASSES, pretrained=False).eval()
                with torch.no_grad():
                    self.assertEqual(tuple(model(images).shape), (images.shape[0], NUM_CLASSES))
                self.assertEqual(model.get_submodule(self.LAST_LAYER[name]).in_features, features)

    def test_fine_tuning_trains_every_layer(self) -> None:
        for name in self.FEATURES:
            with self.subTest(model=name):
                model = create_model(name, num_classes=NUM_CLASSES, pretrained=False)
                self.assertTrue(all(p.requires_grad for p in model.parameters()))

    def test_freeze_backbone_trains_only_the_last_layer(self) -> None:
        for name in self.FEATURES:
            with self.subTest(model=name):
                model = create_model(name, num_classes=NUM_CLASSES, pretrained=False, freeze_backbone=True)
                trainable = [n for n, p in model.named_parameters() if p.requires_grad]
                last = self.LAST_LAYER[name]
                self.assertEqual(trainable, [f"{last}.weight", f"{last}.bias"])

    def test_invalid_num_classes(self) -> None:
        for name in self.FEATURES:
            with self.subTest(model=name), self.assertRaises(ValueError):
                create_model(name, num_classes=0, pretrained=False)


class TestResNet50Article(unittest.TestCase):
    """The attention variant of ``resnet50_article``."""

    def setUp(self) -> None:
        self.images, self.labels = dummy_batch()
        self.model = create_model("resnet50_article", num_classes=NUM_CLASSES, pretrained=False, use_attention=True)

    def test_output_shape(self) -> None:
        with torch.no_grad():
            output = self.model.eval()(self.images)
        self.assertEqual(tuple(output.shape), (self.images.shape[0], NUM_CLASSES))

    def test_attention_starts_as_a_no_op(self) -> None:
        self.model.eval()
        with torch.no_grad():
            self.assertTrue(torch.equal(self.model(self.images), self.model.base(self.images)))

    def test_attention_learns(self) -> None:
        nn.functional.cross_entropy(self.model.train()(self.images), self.labels).backward()
        self.assertGreater(self.model.att1.bn.weight.grad.abs().sum().item(), 0)

    def test_checkpoint_round_trip(self) -> None:
        kwargs = {"num_classes": NUM_CLASSES, "pretrained": False, "use_attention": True}
        rebuilt = create_model("resnet50_article", **kwargs)
        rebuilt.load_state_dict(self.model.state_dict())
        self.model.eval(), rebuilt.eval()
        with torch.no_grad():
            self.assertTrue(torch.equal(self.model(self.images), rebuilt(self.images)))


if __name__ == "__main__":
    unittest.main()
