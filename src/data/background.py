"""Background removal with the official CUB-200-2011 segmentation masks.

Used only by runs whose config sets ``data.remove_background: true``. Every image is cropped to
the bird and everything else is filled with the padding color. ``transforms.py`` and ``dataset.py``
are not changed, so all other runs keep their preprocessing.

The masks are human-made and needed for test images too, so these scores only apply when the bird
is already marked. They are not comparable with models that see the full image.

    python -m src.data.background    # checks every image against its mask and saves examples
"""
import tarfile
import urllib.request
from pathlib import Path

import torch
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2 as T

from src.data.transforms import CROP_MIN_AREA, IMAGE_SIZE, PAD_COLOR, Letterbox, ToRGB, to_tensor

MASKS_URL = "https://data.caltech.edu/records/w9d68-gec53/files/segmentations.tgz?download=1"
MASKS_DIR = Path.home() / ".cache" / "cub_segmentations"
NUM_MASKS = 11788

MASK_THRESHOLD = 128  # the masks have gray levels where annotators disagree; keep what most marked as bird
MASK_GROW = 5         # odd filter size: grows the mask ~2 px, so its exact outline is harder to learn
BOX_MARGIN = 0.1      # crop margin around the bird, as a fraction of its longest side


def extract_masks(archive_path, masks_dir):
    """Unpack the mask archive, refusing any path that would land outside ``masks_dir``."""
    root = Path(masks_dir).resolve()
    with tarfile.open(archive_path) as archive:
        members = archive.getmembers()
        for member in members:
            if not member.isdir() and not member.isfile():
                raise ValueError(f"Unexpected entry in the mask archive: {member.name}")
            if not (root / member.name).resolve().is_relative_to(root):
                raise ValueError(f"Unsafe path in the mask archive: {member.name}")
        options = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}  # missing in older Python versions
        archive.extractall(root, members=members, **options)


def find_masks(masks_dir=MASKS_DIR):
    """Return {image name without extension: mask path}, downloading the masks the first time."""
    masks_dir = Path(masks_dir)
    masks = {path.stem: path for path in masks_dir.rglob("*.png")}
    if len(masks) >= NUM_MASKS:
        return masks

    masks_dir.mkdir(parents=True, exist_ok=True)
    archive_path = masks_dir / "segmentations.tgz"
    print(f"Downloading the CUB masks (~40 MB) to {masks_dir}")
    urllib.request.urlretrieve(MASKS_URL, archive_path)
    extract_masks(archive_path, masks_dir)
    archive_path.unlink()

    masks = {path.stem: path for path in masks_dir.rglob("*.png")}
    if len(masks) != NUM_MASKS:
        raise RuntimeError(f"Expected {NUM_MASKS} masks in {masks_dir}, found {len(masks)}")
    return masks


def remove_background(image, mask):
    """Crop an RGB image to the bird and fill the rest with the padding color.

    An empty mask returns the image unchanged. A mask of a different size raises ValueError.
    """
    if mask.size != image.size:
        raise ValueError(f"Mask size {mask.size} differs from image size {image.size}")
    bird = mask.convert("L").point(lambda value: 255 if value >= MASK_THRESHOLD else 0)
    bird = bird.filter(ImageFilter.MaxFilter(MASK_GROW))
    box = bird.getbbox()
    if box is None:
        return image

    filled = Image.composite(image, Image.new("RGB", image.size, PAD_COLOR), bird)
    left, top, right, bottom = box
    margin = round(BOX_MARGIN * max(right - left, bottom - top))
    return filled.crop((max(left - margin, 0), max(top - margin, 0),
                        min(right + margin, image.width), min(bottom + margin, image.height)))


def build_train_steps(image_size=IMAGE_SIZE, random_crop=False):
    """Steps before and after background removal, for training images.

    Color changes happen first, so the fill keeps exactly the mean color (as in build_train_transform).
    """
    crop = [T.RandomResizedCrop(image_size, scale=(CROP_MIN_AREA, 1.0), ratio=(0.9, 1.1), antialias=True)]
    before = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1)
    after = T.Compose([
        Letterbox(image_size),
        T.RandomHorizontalFlip(p=0.5),
        *(crop if random_crop else []),
        T.RandomAffine(degrees=15, translate=(0.05, 0.05), scale=None if random_crop else (0.9, 1.1), fill=PAD_COLOR),
        to_tensor(),
    ])
    return before, after


def build_eval_steps(image_size=IMAGE_SIZE):
    """Steps before and after background removal, for validation and test images: no randomness."""
    return None, T.Compose([Letterbox(image_size), to_tensor()])


class BackgroundRemovedDataset(Dataset):
    """CUB images without background: returns (preprocessed image tensor, species label)."""

    def __init__(self, hf_dataset, indices, masks, before, after):
        self.hf_dataset = hf_dataset
        self.indices = list(indices)
        self.masks = masks    # {image name without extension: mask path}
        self.before = before  # runs on the full image; None to skip
        self.after = after    # runs on the image without background

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        sample = self.hf_dataset[int(self.indices[i])]
        image = ToRGB()(sample["image"])
        if self.before is not None:
            image = self.before(image)
        with Image.open(self.masks[Path(sample["file_name"]).stem]) as mask:
            image = remove_background(image, mask)
        return self.after(image), sample["label"]


def get_background_datasets(image_size=IMAGE_SIZE, split_path=None, random_crop=False, masks_dir=MASKS_DIR):
    """Train (augmented), validation and test datasets, using the same saved split as every other run."""
    from src.data.data_loader import load_cub_dataset
    from src.data.splits import SPLIT_PATH, load_split

    train_dataset, test_dataset = load_cub_dataset()
    split = load_split(split_path or SPLIT_PATH)
    masks = find_masks(masks_dir)

    train_steps = build_train_steps(image_size, random_crop)
    eval_steps = build_eval_steps(image_size)
    return (BackgroundRemovedDataset(train_dataset, split["train"], masks, *train_steps),
            BackgroundRemovedDataset(train_dataset, split["val"], masks, *eval_steps),
            BackgroundRemovedDataset(test_dataset, range(len(test_dataset)), masks, *eval_steps))


def get_background_dataloaders(batch_size=32, num_workers=2, image_size=IMAGE_SIZE, split_path=None, seed=42,
                               random_crop=False, masks_dir=MASKS_DIR):
    """Same as get_dataloaders in dataset.py, for images without background."""
    train_set, val_set, test_set = get_background_datasets(image_size, split_path, random_crop, masks_dir)

    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_set, shuffle=True, generator=generator, **common)
    val_loader = DataLoader(val_set, shuffle=False, **common)
    test_loader = DataLoader(test_set, shuffle=False, **common)
    return train_loader, val_loader, test_loader


def check_dataset(dataset):
    """Count the images of a BackgroundRemovedDataset with a missing, wrongly sized or empty mask."""
    problems = {"missing": 0, "wrong size": 0, "empty": 0}
    for index in dataset.indices:
        sample = dataset.hf_dataset[int(index)]
        path = dataset.masks.get(Path(sample["file_name"]).stem)
        if path is None:
            problems["missing"] += 1
            continue
        with Image.open(path) as mask:
            if mask.size != sample["image"].size:
                problems["wrong size"] += 1
            elif mask.convert("L").point(lambda value: 255 if value >= MASK_THRESHOLD else 0).getbbox() is None:
                problems["empty"] += 1
    return problems


def save_examples(dataset, output, count=8):
    """Save the images a model receives, for a visual check."""
    from matplotlib.figure import Figure

    from src.data.transforms import denormalize

    figure = Figure(figsize=(count * 2, 2.2), layout="constrained")
    for axis, index in zip(figure.subplots(1, count), torch.randperm(len(dataset))[:count].tolist()):
        axis.imshow(denormalize(dataset[index][0]))
        axis.set_axis_off()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150, facecolor="white")
    print(f"Saved {output}")


if __name__ == "__main__":
    train_set, val_set, test_set = get_background_datasets(random_crop=True)
    for name, dataset in (("train", train_set), ("val", val_set), ("test", test_set)):
        print(f"{name}: {len(dataset)} images, problems: {check_dataset(dataset)}")
    save_examples(train_set, "outputs/eda/background_train_examples.png")
    save_examples(test_set, "outputs/eda/background_test_examples.png")
