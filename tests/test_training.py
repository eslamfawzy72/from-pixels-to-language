"""Technical tests of the training framework on tiny synthetic data.

These only check that the plumbing works (losses, checkpoints, history,
best-model selection, seeding). The accuracies they produce are meaningless.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.benchmark import benchmark_model, main as benchmark_main
from src.evaluation.metrics import accuracy
from src.models import create_model
from src.training import (
    TrainConfig, build_scheduler, classification_loss, fit, load_checkpoint, save_checkpoint, set_seed,
    split_outputs, train_one_epoch, validate,
)

NUM_CLASSES = 4


def synthetic_loader(samples: int = 8, image_size: int = 64, batch_size: int = 4) -> DataLoader:
    generator = torch.Generator().manual_seed(0)
    images = torch.randn(samples, 3, image_size, image_size, generator=generator)
    labels = torch.randint(0, NUM_CLASSES, (samples,), generator=generator)
    return DataLoader(TensorDataset(images, labels), batch_size=batch_size)


class TestLosses(unittest.TestCase):
    def test_split_outputs(self) -> None:
        logits = torch.randn(2, 3)
        self.assertIs(split_outputs(logits)[0], logits)
        main, aux = split_outputs((logits, logits + 1, logits + 2))
        self.assertIs(main, logits)
        self.assertEqual(len(aux), 2)
        with self.assertRaises(TypeError):
            split_outputs({"logits": logits})

    def test_aux_weight_applied(self) -> None:
        criterion = nn.CrossEntropyLoss()
        logits, targets = torch.randn(4, 3), torch.tensor([0, 1, 2, 0])
        main_only, _ = classification_loss(logits, targets, criterion)
        with_aux, _ = classification_loss((logits, logits, logits), targets, criterion, aux_weight=0.3)
        self.assertTrue(torch.allclose(with_aux, main_only * 1.6))

    def test_accuracy(self) -> None:
        logits = torch.tensor([[2.0, 1.0], [0.0, 3.0], [5.0, 0.0]])
        self.assertAlmostEqual(accuracy(logits, torch.tensor([0, 1, 1])), 2 / 3)


class TestTrainingLoop(unittest.TestCase):
    def test_train_and_validate_return_metrics(self) -> None:
        model = create_model("resnet18", num_classes=NUM_CLASSES)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        device = torch.device("cpu")
        train_metrics = train_one_epoch(model, synthetic_loader(), criterion, optimizer, device)
        val_metrics = validate(model, synthetic_loader(), criterion, device)
        for metrics in (train_metrics, val_metrics):
            self.assertEqual(set(metrics), {"loss", "accuracy"})
            self.assertTrue(0.0 <= metrics["accuracy"] <= 1.0)
        self.assertFalse(model.training)

    def test_googlenet_aux_outputs_in_training_loop(self) -> None:
        model = create_model("googlenet", num_classes=NUM_CLASSES)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        metrics = train_one_epoch(model, synthetic_loader(image_size=96), nn.CrossEntropyLoss(),
                                  optimizer, torch.device("cpu"))
        self.assertTrue(torch.isfinite(torch.tensor(metrics["loss"])))
        val = validate(model, synthetic_loader(image_size=96), nn.CrossEntropyLoss(), torch.device("cpu"))
        self.assertIn("accuracy", val)

    def test_fit_writes_checkpoints_and_history(self) -> None:
        config = TrainConfig(model_name="resnet18", num_classes=NUM_CLASSES, epochs=2,
                             learning_rate=0.01, device="cpu")
        model = create_model(config.model_name, num_classes=config.num_classes)
        epochs_seen = []
        with tempfile.TemporaryDirectory() as tmp:
            result = fit(model, synthetic_loader(), synthetic_loader(), config, tmp,
                         on_epoch_end=lambda epoch, _metrics: epochs_seen.append(epoch))
            run_dir = Path(tmp)
            self.assertTrue((run_dir / "best.pt").is_file())
            self.assertTrue((run_dir / "last.pt").is_file())
            summary = json.loads((run_dir / "history.json").read_text(encoding="utf-8"))

            restored = create_model("resnet18", num_classes=NUM_CLASSES)
            checkpoint = load_checkpoint(run_dir / "best.pt", restored)
            self.assertEqual(checkpoint["epoch"], result["best_epoch"])
            self.assertEqual(checkpoint["config"]["model_name"], "resnet18")

        self.assertEqual(epochs_seen, [1, 2])
        self.assertEqual(len(result["history"]), 2)
        self.assertEqual(summary["config"]["epochs"], 2)
        best = max(result["history"], key=lambda m: m["val_accuracy"])
        self.assertEqual(result["best_metric"], best["val_accuracy"])

    def test_best_epoch_uses_monitor_direction(self) -> None:
        from src.training.train import _is_improvement

        self.assertTrue(_is_improvement(0.5, None, "max"))
        self.assertTrue(_is_improvement(0.6, 0.5, "max"))
        self.assertFalse(_is_improvement(0.6, 0.5, "min"))

    def test_invalid_config(self) -> None:
        bad_values = [
            {"optimizer": "rmsprop"}, {"monitor": "f1"}, {"scheduler": "linear"},
            {"warmup_epochs": 30}, {"scheduler": "plateau", "warmup_epochs": 2},
            {"label_smoothing": 1.0}, {"early_stopping_patience": 0},
        ]
        for values in bad_values:
            with self.subTest(**values), self.assertRaises(ValueError):
                TrainConfig(model_name="resnet18", num_classes=4, **values)

    def test_early_stopping(self) -> None:
        # lr=1e-9 on a model without BatchNorm: validation accuracy cannot improve after epoch 1.
        config = TrainConfig(model_name="alexnet", num_classes=NUM_CLASSES, epochs=10, learning_rate=1e-9,
                             scheduler="none", early_stopping_patience=2, device="cpu")
        model = create_model("alexnet", num_classes=NUM_CLASSES)
        with tempfile.TemporaryDirectory() as tmp:
            result = fit(model, synthetic_loader(), synthetic_loader(), config, tmp)
        self.assertTrue(result["stopped_early"])
        self.assertEqual(len(result["history"]), 3)
        self.assertEqual(result["best_epoch"], 1)

    def test_checkpoint_extra_is_saved(self) -> None:
        config = TrainConfig(model_name="resnet18", num_classes=NUM_CLASSES, epochs=1, device="cpu")
        model = create_model("resnet18", num_classes=NUM_CLASSES)
        with tempfile.TemporaryDirectory() as tmp:
            fit(model, synthetic_loader(), synthetic_loader(), config, tmp, checkpoint_extra={"image_size": 64})
            checkpoint = load_checkpoint(Path(tmp) / "best.pt", create_model("resnet18", num_classes=NUM_CLASSES))
        self.assertEqual(checkpoint["extra"], {"image_size": 64})


class TestSchedulers(unittest.TestCase):
    @staticmethod
    def learning_rates(**overrides) -> list[float]:
        config = TrainConfig(model_name="resnet18", num_classes=NUM_CLASSES,
                             **{"epochs": 10, "learning_rate": 0.1, **overrides})
        optimizer = torch.optim.SGD([nn.Parameter(torch.zeros(1))], lr=config.learning_rate)
        scheduler = build_scheduler(optimizer, config)
        rates = []
        for _ in range(config.epochs):
            rates.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(0.5)  # constant metric -> plateau
            elif scheduler is not None:
                scheduler.step()
        return rates

    def test_none_is_constant(self) -> None:
        self.assertEqual(set(self.learning_rates(scheduler="none")), {0.1})

    def test_cosine_decays(self) -> None:
        rates = self.learning_rates(scheduler="cosine")
        self.assertAlmostEqual(rates[0], 0.1)
        self.assertTrue(all(a >= b for a, b in zip(rates, rates[1:])))
        self.assertLess(rates[-1], 0.01)

    def test_warmup_then_cosine(self) -> None:
        rates = self.learning_rates(scheduler="cosine", warmup_epochs=3)
        self.assertLess(rates[0], rates[1])
        self.assertLess(rates[1], rates[2])
        self.assertAlmostEqual(rates[3], 0.1)
        self.assertLess(rates[-1], rates[3])

    def test_step(self) -> None:
        rates = self.learning_rates(scheduler="step", step_size=4, gamma=0.5)
        self.assertAlmostEqual(rates[3], 0.1)
        self.assertAlmostEqual(rates[4], 0.05)
        self.assertAlmostEqual(rates[8], 0.025)

    def test_plateau_reduces_when_metric_stalls(self) -> None:
        rates = self.learning_rates(scheduler="plateau", plateau_patience=2, gamma=0.1)
        self.assertAlmostEqual(rates[0], 0.1)
        self.assertLess(rates[-1], 0.1)


class TestCheckpointAndSeed(unittest.TestCase):
    def test_checkpoint_round_trip_with_optimizer(self) -> None:
        model = create_model("alexnet", num_classes=NUM_CLASSES)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        with tempfile.TemporaryDirectory() as tmp:
            path = save_checkpoint(Path(tmp) / "ckpt.pt", model, optimizer, epoch=3, metrics={"val_accuracy": 0.5})
            other = create_model("alexnet", num_classes=NUM_CLASSES)
            other_optimizer = torch.optim.SGD(other.parameters(), lr=0.01, momentum=0.9)
            checkpoint = load_checkpoint(path, other, other_optimizer)
        self.assertEqual(checkpoint["epoch"], 3)
        for a, b in zip(model.state_dict().values(), other.state_dict().values()):
            self.assertTrue(torch.equal(a, b))

    def test_seed_makes_initialization_reproducible(self) -> None:
        set_seed(123)
        a = create_model("resnet18", num_classes=NUM_CLASSES)
        set_seed(123)
        b = create_model("resnet18", num_classes=NUM_CLASSES)
        for pa, pb in zip(a.parameters(), b.parameters()):
            self.assertTrue(torch.equal(pa, pb))


class TestBenchmark(unittest.TestCase):
    def test_benchmark_model_on_cpu(self) -> None:
        model = create_model("resnet18", num_classes=NUM_CLASSES)
        result = benchmark_model(model, "resnet18", NUM_CLASSES, torch.device("cpu"),
                                 batch_size=1, image_size=64, warmup=1, iterations=2)
        self.assertGreater(result.total_params, 0)
        self.assertGreater(result.latency_ms_mean, 0)
        self.assertIsNone(result.peak_memory_mb)

    def test_benchmark_cli_writes_json_and_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "bench"
            benchmark_main(["--models", "alexnet", "--image-size", "64", "--warmup", "1",
                            "--iterations", "2", "--device", "cpu", "--output", str(output)])
            payload = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
            self.assertTrue(output.with_suffix(".csv").is_file())
        self.assertEqual(payload["results"][0]["model"], "alexnet")
        self.assertIn("torch", payload["environment"])


if __name__ == "__main__":
    unittest.main()
