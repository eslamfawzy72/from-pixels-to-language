"""Image preprocessing for classification: RGB conversion, letterbox resize,
augmentation (training only) and ImageNet normalization.

Decisions and evidence are in notebooks/eda.ipynb and notebooks/image_preprocessing.ipynb.
"""
import torch
from PIL import Image
from torchvision.transforms import v2 as T

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ImageNet mean in 0-255: padding becomes ~0 after normalization, so it carries no information
PAD_COLOR = tuple(round(m * 255) for m in IMAGENET_MEAN)


class ToRGB:
    """Convert any image to 3-channel RGB. A grayscale value g becomes (g, g, g)."""

    def __call__(self, img):
        return img if img.mode == "RGB" else img.convert("RGB")


class Letterbox:
    """Resize so the longest side is `size`, then pad to a square. Nothing is cut off or distorted."""

    def __init__(self, size=IMAGE_SIZE, fill=PAD_COLOR):
        self.size = size
        self.fill = fill

    def __call__(self, img):
        scale = self.size / max(img.size)
        new_w, new_h = round(img.width * scale), round(img.height * scale)
        resized = img.resize((new_w, new_h), Image.BICUBIC)

        canvas = Image.new("RGB", (self.size, self.size), self.fill)
        canvas.paste(resized, ((self.size - new_w) // 2, (self.size - new_h) // 2))
        return canvas


def to_tensor():
    """PIL image -> float tensor (3, H, W), normalized with the ImageNet mean and std."""
    return T.Compose([
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),  # 0-255 -> 0.0-1.0
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def build_train_transform(image_size=IMAGE_SIZE):
    """Training images: random color change -> letterbox -> random flip/rotation/shift/zoom -> tensor.

    Color changes happen before the letterbox so the padding keeps exactly the mean color.
    No hue change (color identifies species), no vertical flip, no random crop (it can cut the bird).
    """
    return T.Compose([
        ToRGB(),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        Letterbox(image_size),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomAffine(degrees=15, translate=(0.05, 0.05), scale=(0.9, 1.1), fill=PAD_COLOR),
        to_tensor(),
    ])


def build_eval_transform(image_size=IMAGE_SIZE):
    """Validation / test images: letterbox -> tensor. No randomness, so results are repeatable."""
    return T.Compose([
        ToRGB(),
        Letterbox(image_size),
        to_tensor(),
    ])


def denormalize(x):
    """Undo the normalization of one tensor (3, H, W) for display: returns an H x W x 3 array in 0-1."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (x.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
