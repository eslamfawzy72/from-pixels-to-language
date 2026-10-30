"""Hyperparameter search: search spaces, the study runner and the ``tune`` CLI.

The study tests use a fake training function and the CLI test uses tiny fake
data (see ``tests.test_cli``), so the accuracies here are meaningless.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any

import yaml

from src import cli
from src.training import TrainConfig
from src.tuning import comparison_rows, grid_trials, load_study, markdown_table, random_trials, run_study
from tests.test_cli import HAS_MLFLOW, IMAGE_SIZE, NUM_CLASSES, fake_data_modules, swapped_modules

REPO_TUNING = Path(__file__).resolve().parents[1] / "configs" / "tuning.yaml"
BASE = {"model_name": "resnet18", "num_classes": NUM_CLASSES, "epochs": 3, "device": "cpu"}


def make_temp_dir(test: unittest.TestCase) -> Path:
    # Removed with ignore_errors: on Windows MLflow's SQLite connection keeps mlflow.db locked.
    path = Path(tempfile.mkdtemp())
    test.addCleanup(shutil.rmtree, path, ignore_errors=True)
    return path


def fake_fit_result(config: TrainConfig, run_dir: Path) -> dict[str, Any]:
    """A fit()-shaped result whose val accuracy peaks at learning_rate 0.05 (epoch 2)."""
    peak = 1.0 - abs(config.learning_rate - 0.05)
    history = [{"epoch": e, "train_loss": 1.0, "train_accuracy": 0.9, "val_loss": 2.0 - peak / e,
                "val_accuracy": peak * (1.0 if e == 2 else 0.9)} for e in range(1, config.epochs + 1)]
    run_dir.mkdir(parents=True, exist_ok=True)
    return {"history": history, "best_epoch": 2, "best_metric": peak, "stopped_early": False,
            "run_dir": str(run_dir)}


class TestSearchSpace(unittest.TestCase):
    def test_grid_is_the_cartesian_product(self) -> None:
        trials = grid_trials({"learning_rate": [0.02, 0.05, 0.1], "weight_decay": [5e-4, 2e-3]})
        self.assertEqual(len(trials), 6)
        self.assertEqual(len({tuple(t.items()) for t in trials}), 6)
        self.assertEqual(trials[0], {"learning_rate": 0.02, "weight_decay": 5e-4})

    def test_bad_grid(self) -> None:
        for grid in ({}, {"learning_rate": []}, {"learning_rate": 0.1}):
            with self.subTest(grid=grid), self.assertRaises(ValueError):
                grid_trials(grid)

    def test_random_is_reproducible_and_within_bounds(self) -> None:
        space = {"learning_rate": {"type": "loguniform", "low": 0.01, "high": 0.15},
                 "momentum": {"type": "uniform", "low": 0.8, "high": 0.95},
                 "warmup_epochs": {"type": "int", "low": 0, "high": 2},
                 "label_smoothing": {"type": "choice", "values": [0.0, 0.1, 0.2]}}
        trials = random_trials(space, num_trials=50, seed=1)
        self.assertEqual(trials, random_trials(space, num_trials=50, seed=1))
        self.assertNotEqual(trials, random_trials(space, num_trials=50, seed=2))
        for trial in trials:
            self.assertTrue(0.01 <= trial["learning_rate"] <= 0.15)
            self.assertTrue(0.8 <= trial["momentum"] <= 0.95)
            self.assertIn(trial["warmup_epochs"], (0, 1, 2))
            self.assertIn(trial["label_smoothing"], (0.0, 0.1, 0.2))
            self.assertEqual(trial["learning_rate"], float(f"{trial['learning_rate']:.3g}"))

    def test_loguniform_covers_the_low_decade(self) -> None:
        # Uniform sampling on [0.001, 1] would put ~0.1% below 0.002; log-uniform puts ~10% there.
        values = [t["x"] for t in random_trials({"x": {"type": "loguniform", "low": 0.001, "high": 1.0}}, 1000)]
        self.assertGreater(sum(v < 0.002 for v in values), 50)

    def test_bad_distributions(self) -> None:
        bad = [{"type": "normal", "low": 0, "high": 1}, {"type": "uniform", "low": 2, "high": 1},
               {"type": "loguniform", "low": 0, "high": 1}, {"type": "loguniform", "low": "1e-4", "high": 1},
               {"type": "uniform", "low": 0}, {"type": "choice", "values": []}]
        for spec in bad:
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                random_trials({"x": spec}, num_trials=1)
        with self.assertRaises(ValueError):
            random_trials({"x": {"type": "choice", "values": [1]}}, num_trials=0)


class TestStudy(unittest.TestCase):
    TRIALS = [{"learning_rate": 0.02}, {"learning_rate": 0.05}, {"learning_rate": 0.1}]

    def run_fake_study(self, study_dir: Path, base: dict[str, Any] = BASE, trials=TRIALS) -> tuple[dict, list]:
        calls = []

        def train_fn(config: TrainConfig, run_dir: Path, run_name: str) -> dict[str, Any]:
            calls.append(run_name)
            return fake_fit_result(config, run_dir)

        summary = run_study(base, trials, study_dir, "grid", train_fn, space={"learning_rate": [0.02, 0.05, 0.1]})
        return summary, calls

    def test_best_trial_and_result_files(self) -> None:
        study_dir = make_temp_dir(self) / "study"
        summary, calls = self.run_fake_study(study_dir)
        self.assertEqual(calls, ["study_trial_00", "study_trial_01", "study_trial_02"])
        self.assertEqual(summary["best"]["trial"], "trial_01")
        self.assertEqual(summary["best"]["params"], {"learning_rate": 0.05})
        self.assertEqual(summary["best"]["val_accuracy"], 1.0)
        self.assertEqual(summary["best"]["epochs_run"], 3)
        self.assertAlmostEqual(summary["total_seconds"], sum(r["seconds"] for r in summary["trials"]))

        saved = load_study(study_dir)
        self.assertEqual(saved["best"]["trial"], "trial_01")
        self.assertEqual(saved["space"], {"learning_rate": [0.02, 0.05, 0.1]})
        csv_lines = (study_dir / "results.csv").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(csv_lines), 4)
        self.assertTrue(csv_lines[0].startswith("trial,learning_rate,val_accuracy"))

    def test_rerun_reuses_finished_trials(self) -> None:
        study_dir = make_temp_dir(self) / "study"
        first, _ = self.run_fake_study(study_dir)
        (study_dir / "trial_02" / "trial_result.json").unlink()  # trial_02 "interrupted"
        second, calls = self.run_fake_study(study_dir)
        self.assertEqual(calls, ["study_trial_02"])
        self.assertEqual(second["best"], first["best"])
        self.assertEqual(len(second["trials"]), 3)

    def test_rerun_with_a_different_space_is_rejected(self) -> None:
        study_dir = make_temp_dir(self) / "study"
        self.run_fake_study(study_dir)
        with self.assertRaises(ValueError):
            self.run_fake_study(study_dir, trials=[{"learning_rate": 0.3}])

    def test_invalid_trial_fails_before_training(self) -> None:
        study_dir = make_temp_dir(self) / "study"
        trials = [{"learning_rate": 0.05}, {"label_smoothing": 1.0}]
        with self.assertRaises(ValueError) as ctx:
            self.run_fake_study(study_dir, trials=trials)
        self.assertIn("trial_01", str(ctx.exception))
        self.assertFalse((study_dir / "trial_00").exists())

    def test_val_loss_monitor_picks_the_minimum(self) -> None:
        study_dir = make_temp_dir(self) / "study"

        def train_fn(config: TrainConfig, run_dir: Path, run_name: str) -> dict[str, Any]:
            result = fake_fit_result(config, run_dir)
            return {**result, "best_metric": result["history"][1]["val_loss"]}

        summary = run_study({**BASE, "monitor": "val_loss"}, self.TRIALS, study_dir, "random", train_fn)
        self.assertEqual(summary["best"]["trial"], "trial_01")  # highest peak -> lowest val_loss

    def test_unknown_method(self) -> None:
        with self.assertRaises(ValueError):
            run_study(BASE, self.TRIALS, make_temp_dir(self), "bayesian", lambda *_: {})

    def test_comparison_table(self) -> None:
        root = make_temp_dir(self)
        grid, _ = self.run_fake_study(root / "grid_study")
        rows = comparison_rows([grid, {**grid, "study": "partial", "method": "random", "trials": [],
                                       "best": None, "total_seconds": 0.0}])
        self.assertEqual(rows[0]["trials"], "3/3")
        self.assertEqual(rows[0]["best_value"], 1.0)
        self.assertEqual(rows[1]["best_value"], None)
        table = markdown_table(rows).splitlines()
        self.assertEqual(len(table), 4)
        self.assertIn("learning_rate=0.05", table[2])


class TestTuneCli(unittest.TestCase):
    def test_repo_tuning_config(self) -> None:
        tuning = cli.load_settings(REPO_TUNING)
        grid, _ = cli.build_trials("grid", tuning)
        random_a, space = cli.build_trials("random", tuning)
        random_b, _ = cli.build_trials("random", tuning)
        self.assertEqual(len(grid), 6)
        self.assertEqual(len(random_a), 6)
        self.assertEqual(random_a, random_b)
        self.assertEqual(set(space), {"learning_rate", "weight_decay", "label_smoothing"})
        self.assertEqual(len(cli.build_trials("random", tuning, num_trials=3, search_seed=7)[0]), 3)
        # every trial must form a valid TrainConfig with the repo base config
        base = cli.load_settings(REPO_TUNING.with_name("classification.yaml"))["train"]
        for params in grid + random_a:
            TrainConfig.from_dict({**base, **tuning["trial"], **params})

    def test_printed_override_parses_back(self) -> None:
        for value in (0.05, 1e-05, 0.000123, 3, 0.0):
            settings = {"train": {}}
            cli.apply_override(settings, f"train.x={cli.yaml_scalar(value)}")
            self.assertEqual(settings["train"]["x"], value)

    def write_files(self, root: Path, tracking_enabled: bool) -> tuple[Path, Path]:
        config = {
            "train": {"model_name": "resnet18", "num_classes": NUM_CLASSES, "epochs": 5, "batch_size": 4,
                      "learning_rate": 0.01, "device": "cpu"},
            "data": {"num_workers": 0, "image_size": IMAGE_SIZE},
            "output": {"dir": str(root / "runs")},
            "tracking": {"enabled": tracking_enabled, "dir": str(root / "mlflow")},
        }
        space = {
            "trial": {"epochs": 1, "warmup_epochs": 0},
            "grid": {"learning_rate": [0.01, 0.02]},
            "random": {"num_trials": 3, "seed": 0,
                       "space": {"learning_rate": {"type": "loguniform", "low": 0.001, "high": 0.1}}},
            "experiment": "unit-test-tuning",
        }
        config_path, space_path = root / "config.yaml", root / "tuning.yaml"
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        space_path.write_text(yaml.safe_dump(space), encoding="utf-8")
        return config_path, space_path

    def run_tune(self, tracking_enabled: bool) -> Path:
        root = make_temp_dir(self)
        config_path, space_path = self.write_files(root, tracking_enabled)
        with swapped_modules(fake_data_modules()):
            for method in ("grid", "random"):
                cli.main(["tune", "--task", "classification", "--method", method, "--config", str(config_path),
                          "--space", str(space_path), "--num-trials", "2"])
        runs = root / "runs"
        for method in ("grid", "random"):
            study = load_study(runs / f"tune_resnet18_{method}")
            self.assertEqual(study["method"], method)
            self.assertEqual(len(study["trials"]), 2)
            for record in study["trials"]:
                self.assertEqual(record["epochs_run"], 1)
                self.assertTrue((Path(record["run_dir"]) / "best.pt").is_file())
        self.assertEqual([r["params"] for r in load_study(runs / "tune_resnet18_grid")["trials"]],
                         [{"learning_rate": 0.01}, {"learning_rate": 0.02}])

        table_path = root / "comparison.md"
        cli.main(["compare-tuning", str(runs / "tune_resnet18_grid"), str(runs / "tune_resnet18_random"),
                  "--output", str(table_path)])
        self.assertEqual(len(table_path.read_text(encoding="utf-8").splitlines()), 4)
        return root

    def test_tune_and_compare_without_tracking(self) -> None:
        root = self.run_tune(tracking_enabled=False)
        self.assertFalse((root / "mlflow").exists())

    @unittest.skipUnless(HAS_MLFLOW, "mlflow not installed")
    def test_tune_with_mlflow_parent_and_child_runs(self) -> None:
        import mlflow

        root = self.run_tune(tracking_enabled=True)
        mlflow.set_tracking_uri(f"sqlite:///{(root / 'mlflow' / 'mlflow.db').resolve().as_posix()}")
        runs = mlflow.search_runs(experiment_names=["unit-test-tuning"])
        parents = runs[runs["tags.role"] == "study"]
        children = runs[runs["tags.role"] == "trial"]
        self.assertEqual(len(parents), 2)
        self.assertEqual(len(children), 4)
        self.assertTrue((runs["status"] == "FINISHED").all())
        self.assertTrue(set(children["tags.mlflow.parentRunId"]) <= set(parents["run_id"]))
        grid = parents[parents["tags.method"] == "grid"].iloc[0]
        self.assertIn(json.loads(grid["tags.best_params"])["learning_rate"], (0.01, 0.02))
        self.assertIn("metrics.best_val_accuracy", runs.columns)


if __name__ == "__main__":
    unittest.main()
