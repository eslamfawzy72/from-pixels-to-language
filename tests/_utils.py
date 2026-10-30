"""Shared settings for the synthetic-data tests.

These values exist only to exercise the code. They are NOT the real dataset's
number of classes or image size. Override with environment variables, e.g.
``TEST_MODELS=alexnet,vgg16 TEST_IMAGE_SIZE=160``.
"""

from __future__ import annotations

import os

import torch

from src.models import available_models

BATCH_SIZE = int(os.environ.get("TEST_BATCH_SIZE", 2))
NUM_CLASSES = int(os.environ.get("TEST_NUM_CLASSES", 10))
IMAGE_SIZE = int(os.environ.get("TEST_IMAGE_SIZE", 224))


def models_under_test() -> list[str]:
    """Models selected via TEST_MODELS (comma-separated), default: all."""
    selected = os.environ.get("TEST_MODELS")
    if not selected:
        return available_models()
    return [name.strip() for name in selected.split(",") if name.strip()]


def dummy_batch(
    batch_size: int = BATCH_SIZE,
    image_size: int = IMAGE_SIZE,
    channels: int = 3,
    num_classes: int = NUM_CLASSES,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Random images ``[N, C, H, W]`` and integer labels ``[N]``."""
    images = torch.randn(batch_size, channels, image_size, image_size)
    labels = torch.randint(0, num_classes, (batch_size,))
    return images, labels
