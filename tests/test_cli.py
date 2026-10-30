"""CLI, config handling and end-to-end train/evaluate/predict on fake data.

The real data pipeline (``src.data.dataset`` / ``src.data.transforms``) needs
the CUB download and torchvision, so these tests swap in a tiny fake module
with the same functions. Accuracies produced here are meaningless.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import torch
from torch.utils.data import DataLoader, TensorDataset

from src import cli

NUM_CLASSES = 3
IMAGE_SIZE = 32
HAS_MLFLOW = importlib.util.find_spec("mlflow") is not None
REPO_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "classification.yaml"


def fake_data_modules() -> dict[str, types.ModuleType]:
    """Stand-ins for src.data.dataset and src.data.transforms."""
    dataset = types.ModuleType("src.data.dataset")

    def get_dataloaders(batch_size=32, num_workers=2, image_size=IMAGE_SIZE, seed=42, **_):
        generator = torch.Generator().manual_seed(0)

        def loader(n):
            images = torch.randn(n, 3, image_size, image_size, generator=generator)
            labels = torch.randint(0, NUM_CLASSES, (n,), generator=generator)
            return DataLoader(TensorDataset(images, labels), batch_size=4)

        return loader(8), loader(4), loader(4)

    dataset.get_dataloaders = get_dataloaders
    dataset.get_class_names = lambda: [f"species_{i}" for i in range(NUM_CLASSES)]

    transforms = types.ModuleType("src.data.transforms")

    def build_eval_transform(image_size=IMAGE_SIZE):
        def transform(image):
            image = image.convert("RGB").resize((image_size, image_size))
            return torch.tensor(list(image.getdata()), dtype=torch.float32).T.reshape(3, image_size, image_size) / 255
        return transform

    transforms.build_eval_transform = build_eval_transform
    return {"src.data.dataset": dataset, "src.data.transforms": transforms}


def write_config(directory: Path, tracking_uri: str, tracking_enabled: bool) -> Path:
    config = {
        "train": {"model_name": "resnet18", "num_classes": NUM_CLASSES, "epochs": 2, "batch_size": 4,
                  "learning_rate": 0.01, "warmup_epochs": 1, "device": "cpu"},
        "data": {"num_workers": 0, "image_size": IMAGE_SIZE},
        "output": {"dir": str(directory / "runs"), "run_name": "test_run"},
        "tracking": {"enabled": tracking_enabled, "experiment": "unit-test", "tracking_uri": tracking_uri},
    }
    path = directory / "config.yaml"
    path.write_text(json.dumps(config), encoding="utf-8")  # JSON is valid YAML
    return path


class TestSettings(unittest.TestCase):
    def parse_train(self, *extra: str):
        return cli.build_parser().parse_args(["train", "--task", "classification", "--config", str(REPO_CONFIG), *extra])

    def test_repo_config_builds_a_valid_train_config(self) -> None:
        from src.training import TrainConfig

        settings = cli.resolve_train_settings(self.parse_train())
        config = TrainConfig.from_dict(settings["train"])
        self.assertEqual(config.num_classes, 200)

    def test_shortcut_flags_and_set_overrides(self) -> None:
        args = self.parse_train("--model", "vgg16", "--epochs", "5", "--lr", "0.1", "--no-tracking",
                                "--set", "train.model_kwargs.batch_norm=true", "--set", "data.num_workers=0")
        settings = cli.resolve_train_settings(args)
        self.assertEqual(settings["train"]["model_name"], "vgg16")
        self.assertEqual(settings["train"]["epochs"], 5)
        self.assertEqual(settings["train"]["learning_rate"], 0.1)
        self.assertIs(settings["train"]["model_kwargs"]["batch_norm"], True)
        self.assertEqual(settings["data"]["num_workers"], 0)
        self.assertFalse(settings["tracking"]["enabled"])

    def test_bad_overrides(self) -> None:
        for bad in ("train.epochs", "epochs=3"):
            with self.subTest(override=bad), self.assertRaises(ValueError):
                cli.apply_override({"train": {}}, bad)

    def test_unknown_train_key_is_rejected(self) -> None:
        from src.training import TrainConfig

        with self.assertRaises(ValueError) as ctx:
            TrainConfig.from_dict({"model_name": "resnet18", "num_classes": 3, "learnig_rate": 0.1})
        self.assertIn("learnig_rate", str(ctx.exception))

    def test_task_is_required(self) -> None:
        with self.assertRaises(SystemExit), mock.patch("sys.stderr"):
            cli.build_parser().parse_args(["train"])


class TestEndToEnd(unittest.TestCase):
    def run_pipeline(self, tracking_enabled: bool) -> None:
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(sys.modules, fake_data_modules()):
            tmp_path = Path(tmp)
            tracking_uri = (tmp_path / "mlruns").as_uri()
            config_path = write_config(tmp_path, tracking_uri, tracking_enabled)

            with mock.patch("sys.stdout"):
                cli.main(["train", "--task", "classification", "--config", str(config_path)])
            run_dir = tmp_path / "runs" / "test_run"
            checkpoint = run_dir / "best.pt"
            self.assertTrue(checkpoint.is_file())
            saved = torch.load(checkpoint, weights_only=True)
            self.assertEqual(saved["extra"]["class_names"], [f"species_{i}" for i in range(NUM_CLASSES)])
            history = json.loads((run_dir / "history.json").read_text(encoding="utf-8"))
            self.assertEqual(len(history["history"]), 2)
            self.assertLess(history["history"][0]["lr"], history["history"][1]["lr"])  # warm-up

            with mock.patch("sys.stdout"):
                cli.main(["evaluate", "--task", "classification", "--checkpoint", str(checkpoint),
                          "--num-workers", "0", "--device", "cpu"])
            test_metrics = json.loads((run_dir / "test_metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(test_metrics["images"], 4)

            from PIL import Image

            image_path = tmp_path / "bird.png"
            Image.new("RGB", (50, 40), (120, 80, 30)).save(image_path)
            from src.tasks.classification import predict_classification

            top = predict_classification(checkpoint, [image_path], top_k=2, device="cpu")[0]
            self.assertEqual(len(top), 2)
            self.assertTrue(0 < sum(p for _, p in top) <= 1 + 1e-6)
            self.assertGreaterEqual(top[0][1], top[1][1])
            self.assertTrue(all(name.startswith("species_") for name, _ in top))

            if tracking_enabled:
                import mlflow

                mlflow.set_tracking_uri(tracking_uri)
                runs = mlflow.search_runs(experiment_names=["unit-test"])
                self.assertEqual(len(runs), 1)
                run = runs.iloc[0]
                self.assertEqual(run["status"], "FINISHED")
                self.assertEqual(run["params.model_name"], "resnet18")
                self.assertIn("metrics.val_accuracy", runs.columns)
                client = mlflow.tracking.MlflowClient()
                self.assertEqual(len(client.get_metric_history(run["run_id"], "val_accuracy")), 2)
                self.assertEqual(saved["extra"]["mlflow_run_id"], run["run_id"])

    def test_train_evaluate_predict_without_tracking(self) -> None:
        self.run_pipeline(tracking_enabled=False)

    @unittest.skipUnless(HAS_MLFLOW, "mlflow not installed")
    def test_train_with_mlflow_tracking(self) -> None:
        self.run_pipeline(tracking_enabled=True)


if __name__ == "__main__":
    unittest.main()
