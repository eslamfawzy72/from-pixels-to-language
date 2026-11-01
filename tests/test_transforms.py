"""Image preprocessing tests."""

from __future__ import annotations

import unittest

import torch
from PIL import Image
from torchvision.transforms import v2 as T

from src.data.transforms import build_eval_transform, build_train_transform

SIZE = 96


def picture() -> Image.Image:
    """A small non-square image whose pixels all differ, so any crop or shift changes the result."""
    image = Image.new("RGB", (120, 80))
    image.putdata([(x * 2, y * 3, (x + y) % 256) for y in range(80) for x in range(120)])
    return image


def has_step(transform: T.Compose, step: type) -> bool:
    return any(isinstance(t, step) for t in transform.transforms)


class TestRandomCrop(unittest.TestCase):
    def test_off_by_default(self) -> None:
        transform = build_train_transform(SIZE)
        self.assertFalse(has_step(transform, T.RandomResizedCrop))
        affine = next(t for t in transform.transforms if isinstance(t, T.RandomAffine))
        self.assertEqual(tuple(affine.scale), (0.9, 1.1))

    def test_replaces_the_zoom(self) -> None:
        transform = build_train_transform(SIZE, random_crop=True)
        self.assertTrue(has_step(transform, T.RandomResizedCrop))
        affine = next(t for t in transform.transforms if isinstance(t, T.RandomAffine))
        self.assertIsNone(affine.scale)

    def test_keeps_the_image_size(self) -> None:
        for random_crop in (False, True):
            with self.subTest(random_crop=random_crop):
                output = build_train_transform(SIZE, random_crop=random_crop)(picture())
                self.assertEqual(tuple(output.shape), (3, SIZE, SIZE))

    def test_is_random(self) -> None:
        transform = build_train_transform(SIZE, random_crop=True)
        torch.manual_seed(0)
        self.assertFalse(torch.equal(transform(picture()), transform(picture())))

    def test_evaluation_stays_repeatable(self) -> None:
        transform = build_eval_transform(SIZE)
        self.assertFalse(has_step(transform, T.RandomResizedCrop))
        self.assertTrue(torch.equal(transform(picture()), transform(picture())))


if __name__ == "__main__":
    unittest.main()
