"""Encoder / classifier split: encoders on their own, and checkpoints saved before the split."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from src.models import available_encoders, build_encoder, create_model, upgrade_state_dict
from src.training import TrainConfig

from ._utils import NUM_CLASSES, dummy_batch

FAST_ENCODERS = ("alexnet", "googlenet", "resnet18", "resnet18_pretrained", "resnet50_pretrained")


def encoder_options(name: str) -> dict:
    return {"pretrained": False} if name.endswith("_pretrained") else {}


def old_style(state: dict) -> dict:
    """What a checkpoint saved before the split looked like: no ``encoder.`` / ``head.`` prefix."""
    return {key.split(".", 1)[1]: value for key, value in state.items()}


class TestEncoders(unittest.TestCase):
    def test_every_model_has_an_encoder(self) -> None:
        self.assertEqual(len(available_encoders()), 11)

    def test_out_channels_matches_the_feature_map(self) -> None:
        images, _ = dummy_batch()
        for name in FAST_ENCODERS:
            with self.subTest(encoder=name):
                encoder = build_encoder(name, **encoder_options(name)).eval()
                with torch.no_grad():
                    features = encoder(images)
                self.assertEqual(features.dim(), 4)
                self.assertEqual(features.shape[1], encoder.out_channels)

    def test_classifier_uses_the_same_encoder(self) -> None:
        model = create_model("resnet18", num_classes=NUM_CLASSES)
        self.assertEqual(type(model.encoder), type(build_encoder("resnet18")))
        self.assertEqual(model.head.fc.in_features, model.encoder.out_channels)

    def test_frozen_pretrained_encoder(self) -> None:
        encoder = build_encoder("resnet18_pretrained", pretrained=False, freeze=True)
        self.assertFalse(any(p.requires_grad for p in encoder.parameters()))

    def test_unknown_encoder(self) -> None:
        with self.assertRaises(ValueError):
            build_encoder("resnet9000")


class TestOldCheckpoints(unittest.TestCase):
    def test_upgrade_adds_the_prefixes(self) -> None:
        model = create_model("googlenet", num_classes=NUM_CLASSES)
        state = model.state_dict()
        self.assertEqual(upgrade_state_dict(old_style(state)).keys(), state.keys())
        self.assertIs(upgrade_state_dict(state), state)  # new-style state dicts are left alone

    def test_old_checkpoint_loads_with_the_same_predictions(self) -> None:
        from src.tasks.classification import load_model_from_checkpoint

        images, _ = dummy_batch()
        for name in ("resnet18", "resnet18_pretrained"):
            with self.subTest(model=name), tempfile.TemporaryDirectory() as tmp:
                options = encoder_options(name)
                config = TrainConfig(model_name=name, num_classes=NUM_CLASSES, model_kwargs=options)
                model = create_model(name, num_classes=NUM_CLASSES, **options).eval()
                path = Path(tmp) / "best.pt"
                torch.save({"config": config.to_dict(), "model_state": old_style(model.state_dict()),
                            "extra": {}, "epoch": 1}, path)
                loaded, _ = load_model_from_checkpoint(path, torch.device("cpu"))
                with torch.no_grad():
                    self.assertTrue(torch.equal(model(images), loaded(images)))


if __name__ == "__main__":
    unittest.main()
