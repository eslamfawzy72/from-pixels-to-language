"""Loss, backpropagation, optimizer-step, mode and state-dict tests."""

from __future__ import annotations

import gc
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from src.models import count_parameters, create_model

from ._utils import NUM_CLASSES, dummy_batch, models_under_test

# Parameter counts for num_classes=1000, in_channels=3. These are the
# published torchvision counts for the same layer configurations; matching
# them confirms layer sizes, biases and shortcut projections. GoogLeNet is
# excluded because torchvision's version deviates from the paper (3x3 instead
# of 5x5 in branch 3, BatchNorm everywhere).
REFERENCE_PARAMS_1000 = {
    "alexnet": 61_100_840,
    "vgg16": 138_357_544,
    "vgg19": 143_667_240,
    "resnet18": 11_689_512,
    "resnet34": 21_797_672,
    "resnet50": 25_557_032,
    "resnet101": 44_549_160,
    "resnet152": 60_192_808,
}


class TestTrainingStep(unittest.TestCase):
    def test_loss_backward_and_optimizer_step(self) -> None:
        # Each model runs in its own process; see tests/_train_step.py.
        root = Path(__file__).resolve().parents[1]
        for name in models_under_test():
            with self.subTest(model=name):
                result = subprocess.run(
                    [sys.executable, "-m", "tests._train_step", name],
                    cwd=root, capture_output=True, text=True, timeout=600,
                )
                self.assertEqual(result.returncode, 0,
                                 f"exit code {result.returncode}\n{result.stderr[-2000:]}")
                self.assertEqual(json.loads(result.stdout.strip().splitlines()[-1])["model"], name)

    def test_training_step_in_process(self) -> None:
        # Same checks without a subprocess, on a small model, so failures show a direct traceback.
        from ._train_step import check_training_step

        self.assertTrue(math.isfinite(check_training_step("resnet18")["loss"]))


class TestModesAndSerialization(unittest.TestCase):
    def test_eval_mode_is_deterministic(self) -> None:
        images, _ = dummy_batch()
        for name in models_under_test():
            with self.subTest(model=name):
                model = create_model(name, num_classes=NUM_CLASSES).eval()
                self.assertFalse(model.training)
                with torch.no_grad():
                    self.assertTrue(torch.equal(model(images), model(images)))
                model.train()
                self.assertTrue(all(m.training for m in model.modules()))

    def test_state_dict_round_trip(self) -> None:
        images, _ = dummy_batch()
        for name in models_under_test():
            with self.subTest(model=name):
                with tempfile.TemporaryDirectory() as tmp:
                    path = os.path.join(tmp, f"{name}.pt")
                    source = create_model(name, num_classes=NUM_CLASSES).eval()
                    with torch.no_grad():
                        expected = source(images)
                    torch.save(source.state_dict(), path)
                    del source
                    gc.collect()

                    target = create_model(name, num_classes=NUM_CLASSES).eval()
                    target.load_state_dict(torch.load(path, weights_only=True, mmap=True))
                    with torch.no_grad():
                        self.assertTrue(torch.equal(expected, target(images)))
                    del target
                    gc.collect()


class TestParameterCounts(unittest.TestCase):
    def test_counts_match_reference(self) -> None:
        for name in models_under_test():
            if name not in REFERENCE_PARAMS_1000:
                continue
            with self.subTest(model=name):
                total, trainable = count_parameters(create_model(name, num_classes=1000))
                self.assertEqual(total, REFERENCE_PARAMS_1000[name])
                self.assertEqual(trainable, total)

    def test_alexnet_layer_params_match_lecture(self) -> None:
        if "alexnet" not in models_under_test():
            self.skipTest("alexnet not selected")
        model = create_model("alexnet", num_classes=1000)
        convs_and_fcs = [m for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
        counts = [sum(p.numel() for p in m.parameters()) for m in convs_and_fcs]
        # Lecture table, params (k): 23, 307, 664, 885, 590, 37749, 16777, 4097.
        expected = [23_296, 307_392, 663_936, 884_992, 590_080, 37_752_832, 16_781_312, 4_097_000]
        self.assertEqual(counts, expected)

    def test_googlenet_aux_heads_are_counted(self) -> None:
        if "googlenet" not in models_under_test():
            self.skipTest("googlenet not selected")
        with_aux, _ = count_parameters(create_model("googlenet", num_classes=1000))
        without_aux, _ = count_parameters(create_model("googlenet", num_classes=1000, aux_logits=False))
        self.assertGreater(with_aux, without_aux)
        # Main network is ~7M parameters (Szegedy et al. report ~6.8M).
        self.assertTrue(6_000_000 < without_aux < 7_500_000, without_aux)


if __name__ == "__main__":
    unittest.main()
