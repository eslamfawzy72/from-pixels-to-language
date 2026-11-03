"""Background removal: masks, preprocessing, and that it only runs when a config asks for it."""

from __future__ import annotations

import io
import json
import tarfile
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from PIL import Image

from src import cli
from src.data.background import (BOX_MARGIN, BackgroundRemovedDataset, build_eval_steps, build_train_steps,
                                 check_dataset, download_archive, extract_masks, find_masks, remove_background)
from src.data.transforms import IMAGE_SIZE, PAD_COLOR
from tests.test_cli import IMAGE_SIZE as FAKE_IMAGE_SIZE, fake_data_modules, swapped_modules, write_config

GREEN = (0, 200, 0)
RED = (200, 0, 0)
REPO_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "classification_resnet50_scratch_nobg.yaml"


def bird_photo(size=(200, 100), box=(80, 30, 120, 70), value=255) -> tuple[Image.Image, Image.Image]:
    """A green background with a red "bird", and its mask."""
    image = Image.new("RGB", size, GREEN)
    mask = Image.new("L", size, 0)
    image.paste(RED, box)
    mask.paste(value, box)
    return image, mask


class TestRemoveBackground(unittest.TestCase):
    def test_background_is_filled_and_bird_is_kept(self) -> None:
        pixels = np.array(remove_background(*bird_photo()))
        center = pixels.shape[0] // 2
        self.assertEqual(tuple(pixels[center, center].tolist()), RED)
        self.assertEqual(tuple(pixels[0, 0].tolist()), PAD_COLOR)

    def test_only_a_thin_rim_of_background_stays(self) -> None:
        image, mask = bird_photo()
        pixels = np.array(remove_background(image, mask))
        rim = (pixels == GREEN).all(axis=2).sum()
        self.assertEqual(rim, 44 ** 2 - 40 ** 2)  # MASK_GROW adds 2 px on each side of the 40 px bird

    def test_crops_to_the_bird_with_a_margin(self) -> None:
        result = remove_background(*bird_photo())
        grown = 40 + 2 * 2  # MASK_GROW adds 2 px on each side
        expected = grown + 2 * round(BOX_MARGIN * grown)
        self.assertEqual(result.size, (expected, expected))

    def test_crop_stays_inside_the_image(self) -> None:
        result = remove_background(*bird_photo(box=(0, 0, 50, 50)))
        self.assertLessEqual(result.width, 200)
        self.assertLess(result.width, 70)
        self.assertLess(result.height, 70)

    def test_gray_levels_follow_the_threshold(self) -> None:
        kept = np.array(remove_background(*bird_photo(value=153)))
        self.assertIn(RED, {tuple(color) for color in kept.reshape(-1, 3).tolist()})
        image, mask = bird_photo(value=102)
        self.assertEqual(remove_background(image, mask).size, image.size)  # below the threshold: an empty mask

    def test_empty_mask_keeps_the_image(self) -> None:
        image = Image.new("RGB", (50, 40), GREEN)
        self.assertIs(remove_background(image, Image.new("L", (50, 40), 0)), image)

    def test_mask_of_another_size_is_an_error(self) -> None:
        with self.assertRaises(ValueError):
            remove_background(Image.new("RGB", (50, 40)), Image.new("L", (40, 50)))

    def test_rgb_mask_works(self) -> None:
        image, mask = bird_photo()
        self.assertEqual(remove_background(image, mask.convert("RGB")).size, remove_background(image, mask).size)


class TestDataset(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.image, mask = bird_photo()
        self.mask_path = Path(tmp.name) / "bird_1.png"
        mask.save(self.mask_path)
        self.hf = [{"image": self.image, "label": 7, "file_name": "bird_1.jpg"}]
        self.masks = {"bird_1": self.mask_path}

    def dataset(self, steps) -> BackgroundRemovedDataset:
        return BackgroundRemovedDataset(self.hf, [0], self.masks, *steps)

    def test_returns_tensor_and_label(self) -> None:
        for random_crop in (False, True):
            with self.subTest(random_crop=random_crop):
                image, label = self.dataset(build_train_steps(IMAGE_SIZE, random_crop))[0]
                self.assertEqual(tuple(image.shape), (3, IMAGE_SIZE, IMAGE_SIZE))
                self.assertEqual(label, 7)

    def test_evaluation_is_repeatable(self) -> None:
        dataset = self.dataset(build_eval_steps(IMAGE_SIZE))
        self.assertTrue(torch.equal(dataset[0][0], dataset[0][0]))

    def test_training_is_random(self) -> None:
        dataset = self.dataset(build_train_steps(IMAGE_SIZE, True))
        torch.manual_seed(0)
        self.assertFalse(torch.equal(dataset[0][0], dataset[0][0]))

    def test_color_changes_leave_the_fill_alone(self) -> None:
        dataset = self.dataset(build_train_steps(IMAGE_SIZE))
        for seed in range(5):
            torch.manual_seed(seed)
            image = dataset[0][0]
            self.assertLess(image[:, 0, 0].abs().max().item(), 0.02)  # a corner: the mean color becomes ~0

    def test_check_finds_each_problem(self) -> None:
        hf = self.hf + [{"image": self.image, "label": 0, "file_name": "none.jpg"},
                        {"image": Image.new("RGB", (9, 9)), "label": 0, "file_name": "bird_1.jpg"}]
        empty_path = self.mask_path.with_name("empty.png")
        Image.new("L", self.image.size, 0).save(empty_path)
        hf.append({"image": self.image, "label": 0, "file_name": "empty.jpg"})
        masks = {**self.masks, "empty": empty_path}
        dataset = BackgroundRemovedDataset(hf, range(4), masks, None, None)
        self.assertEqual(check_dataset(dataset), {"missing": 1, "wrong size": 1, "empty": 1})


class TestMaskFiles(unittest.TestCase):
    @staticmethod
    def archive(path: Path, name: str) -> None:
        with tarfile.open(path, "w:gz") as archive:
            info = tarfile.TarInfo(name)
            info.size = 4
            archive.addfile(info, io.BytesIO(b"mask"))

    def test_extracts_inside_the_folder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.archive(Path(tmp) / "masks.tgz", "segmentations/species/bird.png")
            extract_masks(Path(tmp) / "masks.tgz", tmp)
            self.assertTrue((Path(tmp) / "segmentations" / "species" / "bird.png").is_file())

    def test_refuses_paths_outside_the_folder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.archive(Path(tmp) / "masks.tgz", "../outside.png")
            with self.assertRaises(ValueError):
                extract_masks(Path(tmp) / "masks.tgz", Path(tmp) / "masks")

    def test_complete_folder_is_not_downloaded_again(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, mock.patch("src.data.background.NUM_MASKS", 1), \
                mock.patch("urllib.request.urlopen") as download:
            (Path(tmp) / "bird.png").write_bytes(b"mask")
            self.assertEqual(set(find_masks(tmp)), {"bird"})
            download.assert_not_called()

    def test_download_names_the_client_and_unpacks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, mock.patch("src.data.background.NUM_MASKS", 1), \
                mock.patch("urllib.request.urlopen") as download:
            archive_path = Path(tmp) / "source.tgz"
            self.archive(archive_path, "segmentations/species/bird.png")
            response = io.BytesIO(archive_path.read_bytes())
            response.headers = {"Content-Length": str(archive_path.stat().st_size)}
            download.return_value = response
            masks_dir = Path(tmp) / "masks"
            self.assertEqual(set(find_masks(masks_dir)), {"bird"})
            self.assertEqual(download.call_args.args[0].get_header("User-agent"), "from-pixels-to-language")
            self.assertFalse((masks_dir / "segmentations.tgz").exists())

    def test_short_download_is_retried(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, mock.patch("urllib.request.urlopen") as download, \
                mock.patch("builtins.print"):
            short = io.BytesIO(b"ma")
            short.headers = {"Content-Length": "4"}
            full = io.BytesIO(b"mask")
            full.headers = {"Content-Length": "4"}
            download.side_effect = [short, full]
            download_archive(Path(tmp) / "masks.tgz")
            self.assertEqual(download.call_count, 2)
            self.assertEqual((Path(tmp) / "masks.tgz").read_bytes(), b"mask")

    def test_download_gives_up_with_a_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, mock.patch("urllib.request.urlopen", side_effect=OSError("down")), \
                mock.patch("builtins.print"), self.assertRaises(RuntimeError):
            download_archive(Path(tmp) / "masks.tgz", attempts=2)


def modules_with_background(calls: list[bool]) -> dict[str, types.ModuleType]:
    """The fake data modules of test_cli, recording which loader each step uses."""
    modules = fake_data_modules()
    real_loader = modules["src.data.dataset"].get_dataloaders
    background = types.ModuleType("src.data.background")

    def plain_loader(**kwargs):
        calls.append(False)
        return real_loader(**kwargs)

    def background_loader(**kwargs):
        calls.append(True)
        return real_loader(**kwargs)

    modules["src.data.dataset"].get_dataloaders = plain_loader
    background.get_background_dataloaders = background_loader
    modules["src.data.background"] = background
    return modules


class TestSwitch(unittest.TestCase):
    def run_steps(self, extra: list[str]) -> tuple[list[bool], dict]:
        calls: list[bool] = []
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp, \
                swapped_modules(modules_with_background(calls)):
            tmp_path = Path(tmp)
            config_path = write_config(tmp_path, tmp_path / "mlflow", tracking_enabled=False)
            cli.main(["train", "--task", "classification", "--config", str(config_path), *extra])
            checkpoint = tmp_path / "runs" / "test_run" / "best.pt"
            cli.main(["evaluate", "--task", "classification", "--checkpoint", str(checkpoint),
                      "--num-workers", "0", "--device", "cpu"])
            cli.main(["gradcam", "--task", "classification", "--checkpoint", str(checkpoint), "--num-examples", "1",
                      "--output-dir", str(tmp_path / "GradCAM"), "--num-workers", "0", "--device", "cpu"])
            image_path = tmp_path / "bird.png"
            Image.new("RGB", (50, 40)).save(image_path)
            from src.tasks.classification import predict_classification

            with mock.patch("builtins.print") as printed:
                predict_classification(checkpoint, [image_path], top_k=1, device="cpu")
            warning = any("background" in str(call) for call in printed.call_args_list)
            saved = torch.load(checkpoint, weights_only=True)["extra"]
        return calls, {"saved": saved.get("remove_background"), "warning": warning}

    def test_off_by_default_everywhere(self) -> None:
        calls, result = self.run_steps([])
        self.assertEqual(calls, [False, False, False])
        self.assertEqual(result, {"saved": False, "warning": False})

    def test_on_for_train_evaluate_and_gradcam(self) -> None:
        calls, result = self.run_steps(["--set", "data.remove_background=true"])
        self.assertEqual(calls, [True, True, True])
        self.assertEqual(result, {"saved": True, "warning": True})

    def test_tuning_refuses_the_setting(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tuning.yaml"
            path.write_text(json.dumps({"grid": {"learning_rate": [0.1]}}), encoding="utf-8")
            config = write_config(Path(tmp), Path(tmp) / "mlflow", tracking_enabled=False)
            with self.assertRaises(ValueError):
                cli.main(["tune", "--task", "classification", "--method", "grid", "--space", str(path),
                          "--config", str(config), "--set", "data.remove_background=true"])

    def test_repo_config_is_valid(self) -> None:
        from src.training import TrainConfig

        settings = cli.load_settings(REPO_CONFIG)
        config = TrainConfig.from_dict(settings["train"])
        self.assertEqual((config.model_name, config.epochs, config.early_stopping_patience), ("resnet50", 100, 25))
        self.assertIs(settings["data"]["remove_background"], True)
        self.assertIs(settings["data"]["random_crop"], True)
        self.assertEqual(FAKE_IMAGE_SIZE, 32)  # the fake data of test_cli stays small


if __name__ == "__main__":
    unittest.main()
