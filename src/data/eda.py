"""Exploratory data analysis of CUB-200: classes, images and captions.

Collects the analysis from notebooks/eda.ipynb and notebooks/preprocessing.ipynb into one script.
It prints the findings and saves every plot to outputs/eda/:

    python -m src.data.eda                  # first run reads every image (a few minutes)
    python -m src.data.eda --recompute      # ignore the cached image statistics
    python -m src.data.eda --output-dir DIR

Captions are measured raw, with a simple tokenizer, because the point is to see the noise that
src/data/captions.py has to clean.
"""
import argparse
import re
import sys
from collections import Counter
from pathlib import Path
from string import ascii_lowercase

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from src.data.captions import MAX_WORDS, MIN_FREQ, split_captions
from src.data.data_loader import load_cub_dataset
from src.data.splits import dhash

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "eda"
STATS_FILE = "image_stats.csv"

STATS_SIZE = 224         # channel mean/std are measured at the size the model will see
GRAYSCALE_DIFF = 5       # largest R-G / G-B difference (out of 255) of a grayscale photo
LOW_SATURATION = 0.05
TYPO_REFERENCE_FREQ = 50  # a typo is a rare word one edit away from a word at least this common
WORD_ENDINGS = {"s", "es", "y", "d", "ed", "ly", "r", "er", "ing"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

COLOR_WORDS = ["black", "white", "brown", "gray", "grey", "yellow", "red", "orange",
               "blue", "green", "tan", "buff", "olive", "pink", "purple", "rust"]

# Plot styling only.main() applies STYLE to every figure.
SPLIT_COLORS = {"train": "#2a78d6", "test": "#eb6834"}
MAIN_COLOR = SPLIT_COLORS["train"]
SECOND_COLOR = SPLIT_COLORS["test"]
MUTED_INK = "#52514e"
STYLE = {
    "figure.facecolor": "#fcfcfb",
    "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": MUTED_INK,
    "axes.titlesize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": "#e1e0d9",
    "grid.linewidth": 0.6,
    "xtick.color": MUTED_INK,
    "ytick.color": MUTED_INK,
    "xtick.major.size": 0,
    "ytick.major.size": 0,
    "legend.frameon": False,
    "figure.constrained_layout.use": True,
    "savefig.dpi": 150,
}


def section(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def save(fig, output_dir, name):
    fig.savefig(output_dir / f"{name}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"[saved {name}.png]")


def hbars(ax, labels, values, title, xlabel="Count", color=MAIN_COLOR):
    """Horizontal bars, largest at the top."""
    ax.barh(labels[::-1], values[::-1], color=color, height=0.7)
    ax.grid(axis="y", visible=False)
    ax.set_title(title)
    ax.set_xlabel(xlabel)


def hist_by_split(ax, stats, column, title, values=None):
    """One outline per split on shared bins, so the two stay readable where they overlap."""
    values = stats[column] if values is None else values
    edges = np.histogram_bin_edges(values, bins=40)
    for split, color in SPLIT_COLORS.items():
        ax.hist(values[stats["split"] == split], bins=edges, histtype="step",
                linewidth=1.5, color=color, label=split)
    ax.set_title(title)
    ax.set_ylabel("Images")
    ax.legend()


def show_images(datasets, rows, output_dir, name, title=None, ncols=6, size=2.6, suptitle=None):
    """Save a grid of dataset images from rows of the image statistics."""
    rows = list(rows.itertuples(index=False))
    if not rows:
        print(f"Nothing to show for {name}.")
        return
    nrows = -(-len(rows) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * size, nrows * size + 0.5), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, row in zip(axes.flat, rows):
        ax.imshow(datasets[row.split][int(row.idx)]["image"].convert("RGB"))
        ax.set_title(title(row) if title else f"{row.split} #{row.idx}", fontsize=8)
    if suptitle:
        fig.suptitle(suptitle)
    save(fig, output_dir, name)


# ---------------------------------------------------------------- classes

def analyze_classes(datasets, class_names, output_dir):
    section("CLASS DISTRIBUTION")
    counts = pd.DataFrame({
        split: pd.Series(list(dataset["label"])).value_counts().reindex(range(len(class_names)), fill_value=0)
        for split, dataset in datasets.items()
    })
    counts.insert(0, "species", class_names)

    print("Species:", len(class_names))
    for split in datasets:
        c = counts[split]
        ratio = f"{c.max() / c.min():.2f}" if c.min() > 0 else "inf (a species has no images)"
        print(f"{split:<6} images {c.sum():>5} | per species: min {c.min()}, max {c.max()}, "
              f"mean {c.mean():.2f}, median {c.median():.0f} | max/min ratio {ratio}")

    print("\nSpecies with the fewest test images:")
    print(counts.nsmallest(10, "test")[["species", "train", "test"]].to_string(index=False))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    bins = np.arange(counts[["train", "test"]].min().min(), counts[["train", "test"]].max().max() + 2) - 0.5
    for ax, split in zip(axes, datasets):
        ax.hist(counts[split], bins=bins, color=SPLIT_COLORS[split], edgecolor="#fcfcfb", linewidth=1)
        ax.grid(axis="x", visible=False)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_title(f"Images per species ({split})")
        ax.set_xlabel("Images in the species")
        ax.set_ylabel("Number of species")
    smallest = counts.nsmallest(15, "test")
    hbars(axes[2], smallest["species"].tolist(), smallest["test"].tolist(),
          "15 smallest species in the test split", "Test images", SPLIT_COLORS["test"])
    save(fig, output_dir, "01_class_distribution")

    train = datasets["train"]
    fig, axes = plt.subplots(3, 4, figsize=(14, 10))
    for i, ax in enumerate(axes.flat):
        sample = train[i * len(train) // 12]  # spread over the classes, which are stored in order
        ax.imshow(sample["image"].convert("RGB"))
        ax.set_title(class_names[sample["label"]], fontsize=9)
        ax.axis("off")
    save(fig, output_dir, "02_sample_images")


# ---------------------------------------------------------------- images

def compute_image_stats(img):
    rgb = np.asarray(img.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

    # Largest R-G / G-B difference on a small copy: ~0 means a grayscale photo
    small = cv2.resize(rgb, (64, 64), interpolation=cv2.INTER_AREA).astype(np.int16)
    channel_diff = max(
        np.abs(small[..., 0] - small[..., 1]).max(),
        np.abs(small[..., 1] - small[..., 2]).max(),
    )

    resized = cv2.resize(rgb, (STATS_SIZE, STATS_SIZE), interpolation=cv2.INTER_AREA) / 255.0

    return {
        "width": img.width,
        "height": img.height,
        "mode": img.mode,
        "sharpness": cv2.Laplacian(gray, cv2.CV_64F).var(),  # low values suggest blur
        "brightness": gray.mean(),
        "contrast": gray.std(),
        "saturation": hsv[..., 1].mean() / 255,
        "channel_diff": channel_diff,
        "dhash": dhash(img),
        **{f"mean_{c}": resized[..., i].mean() for i, c in enumerate("rgb")},
        **{f"sqmean_{c}": (resized[..., i] ** 2).mean() for i, c in enumerate("rgb")},
    }


def load_image_stats(datasets, class_names, stats_path, recompute=False):
    """One row per image. Slow the first time, then cached as a CSV."""
    if stats_path.exists() and not recompute:
        stats = pd.read_csv(stats_path, dtype={"dhash": str})
        print("Loaded cached statistics from", stats_path)
    else:
        rows = []
        for split, dataset in datasets.items():
            labels = list(dataset["label"])
            for idx in range(len(dataset)):
                row = {"split": split, "idx": idx, "label": labels[idx], "error": None}
                try:
                    # Inside the try: a corrupted file fails here, when the record is decoded
                    row.update(compute_image_stats(dataset[idx]["image"]))
                except Exception as e:
                    row["error"] = f"{type(e).__name__}: {e}"
                rows.append(row)
                if (idx + 1) % 1000 == 0:
                    print(f"{split}: {idx + 1}/{len(dataset)}")
        stats = pd.DataFrame(rows)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats.to_csv(stats_path, index=False)
        print("Saved statistics to", stats_path)

    stats["aspect_ratio"] = stats["width"] / stats["height"]
    stats["short_side"] = stats[["width", "height"]].min(axis=1)
    stats["long_side"] = stats[["width", "height"]].max(axis=1)
    stats["species"] = stats["label"].map(lambda label: class_names[label])
    return stats


def analyze_corrupted(stats):
    section("CORRUPTED FILES")
    failed = stats[stats["error"].notna()]
    print("Images:", stats["split"].value_counts().to_dict())
    print("Failed to decode:", len(failed))
    if len(failed):
        print(failed[["split", "idx", "species", "error"]].to_string(index=False))


def analyze_sizes(stats, datasets, output_dir):
    section("IMAGE SIZES AND ASPECT RATIOS")
    columns = ["width", "height", "aspect_ratio", "short_side"]
    print(stats.groupby("split")[columns].describe().T.round(2).to_string())

    print(f"\nLongest side is 500 px: {(stats['long_side'] == 500).mean() * 100:.1f}%")
    print(f"Landscape (aspect > 1.05): {(stats['aspect_ratio'] > 1.05).mean() * 100:.1f}%")
    print(f"Portrait (aspect < 0.95): {(stats['aspect_ratio'] < 0.95).mean() * 100:.1f}%")
    for threshold in [150, 224, 256]:
        small = stats[stats["short_side"] < threshold]["split"].value_counts().to_dict()
        print(f"Short side < {threshold} px: {small}")

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
    for ax, column in zip(axes[:3], ["width", "height", "aspect_ratio"]):
        hist_by_split(ax, stats, column, column.replace("_", " ").capitalize())
        ax.set_xlabel("Pixels" if column != "aspect_ratio" else "Width / height")
    train = stats[stats["split"] == "train"]
    axes[3].scatter(train["width"], train["height"], s=5, alpha=0.3, color=SPLIT_COLORS["train"], linewidths=0)
    axes[3].set_title("Width vs height (train)")
    axes[3].set_xlabel("Width (pixels)")
    axes[3].set_ylabel("Height (pixels)")
    save(fig, output_dir, "03_image_sizes")

    # Smallest images get upscaled and may look blurry; a center crop may cut extreme shapes
    show_images(datasets, stats.nsmallest(12, "short_side"), output_dir, "04_smallest_images",
                title=lambda r: f"{r.split} #{r.idx}\n{r.width:.0f}x{r.height:.0f}",
                suptitle="Smallest images")
    extreme = pd.concat([stats.nsmallest(6, "aspect_ratio"), stats.nlargest(6, "aspect_ratio")])
    show_images(datasets, extreme, output_dir, "05_extreme_aspect_ratios",
                title=lambda r: f"{r.split} #{r.idx}\naspect {r.aspect_ratio:.2f}",
                suptitle="Most extreme aspect ratios")


def analyze_color_modes(stats, datasets, output_dir):
    section("COLOR MODES AND GRAYSCALE IMAGES")
    print(pd.crosstab(stats["mode"], stats["split"]).to_string())

    grayscale = stats[stats["channel_diff"] < GRAYSCALE_DIFF]
    print("\nGrayscale photos (any mode):", grayscale["split"].value_counts().to_dict())
    show_images(datasets, grayscale, output_dir, "06_grayscale_images",
                title=lambda r: f"{r.split} #{r.idx} ({r.mode})\n{r.species}",
                suptitle="Grayscale photos")

    low_saturation = stats[(stats["saturation"] < LOW_SATURATION) & (stats["channel_diff"] >= GRAYSCALE_DIFF)]
    print("Color photos with very low saturation:", low_saturation["split"].value_counts().to_dict())
    show_images(datasets, low_saturation.nsmallest(12, "saturation"), output_dir, "07_low_saturation_images",
                title=lambda r: f"{r.split} #{r.idx}\nsat {r.saturation:.3f}",
                suptitle="Lowest-saturation color photos")


def analyze_quality(stats, datasets, output_dir):
    section("IMAGE QUALITY (scores only flag candidates: look at the grids)")
    print(stats[["sharpness", "brightness", "contrast", "saturation"]].describe().round(2).to_string())

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
    hist_by_split(axes[0], stats, "sharpness", "Sharpness", values=np.log10(stats["sharpness"]))
    axes[0].set_xlabel("log10 of Laplacian variance")
    hist_by_split(axes[1], stats, "brightness", "Brightness")
    axes[1].set_xlabel("Mean gray level (0-255)")
    hist_by_split(axes[2], stats, "contrast", "Contrast")
    axes[2].set_xlabel("Std of gray levels")
    hist_by_split(axes[3], stats, "saturation", "Saturation")
    axes[3].set_xlabel("Mean HSV saturation (0-1)")
    save(fig, output_dir, "08_image_quality")

    show_images(datasets, stats.nsmallest(12, "sharpness"), output_dir, "09_lowest_sharpness",
                title=lambda r: f"{r.split} #{r.idx}\nsharp {r.sharpness:.0f}",
                suptitle="Lowest sharpness (possible blur)")
    show_images(datasets, pd.concat([stats.nsmallest(6, "brightness"), stats.nlargest(6, "brightness")]),
                output_dir, "10_darkest_and_brightest",
                title=lambda r: f"{r.split} #{r.idx}\nbright {r.brightness:.0f}",
                suptitle="Darkest and brightest")
    show_images(datasets, stats.nsmallest(6, "contrast"), output_dir, "11_lowest_contrast",
                title=lambda r: f"{r.split} #{r.idx}\ncontrast {r.contrast:.1f}",
                suptitle="Lowest contrast")


def analyze_duplicates(stats, datasets, output_dir):
    section("DUPLICATE IMAGES (same perceptual hash)")
    duplicates = stats[stats.duplicated("dhash", keep=False)].sort_values(["dhash", "split", "idx"])
    groups = duplicates.groupby("dhash")["split"].agg(list)

    print("Duplicate groups:", len(groups))
    print("Groups spanning train and test (leakage):", groups.map(lambda s: len(set(s)) > 1).sum())
    print(duplicates[["dhash", "split", "idx", "species", "width", "height"]].to_string(index=False))
    show_images(datasets, duplicates, output_dir, "12_duplicate_images",
                title=lambda r: f"{r.split} #{r.idx}\n{r.species}", ncols=4,
                suptitle="Duplicate groups (consecutive images share a hash)")


def analyze_channels(stats, output_dir):
    section("CHANNEL MEAN AND STANDARD DEVIATION")
    train = stats[stats["split"] == "train"]

    # Every image was resized to the same size, so each one has the same weight
    mean = np.array([train[f"mean_{c}"].mean() for c in "rgb"])
    std = np.sqrt(np.array([train[f"sqmean_{c}"].mean() for c in "rgb"]) - mean ** 2)

    print(pd.DataFrame({
        "CUB train mean": mean.round(3),
        "ImageNet mean": IMAGENET_MEAN,
        "CUB train std": std.round(3),
        "ImageNet std": IMAGENET_STD,
    }, index=["R", "G", "B"]).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    positions = np.arange(3)
    for ax, cub, imagenet, title in [(axes[0], mean, IMAGENET_MEAN, "Channel mean"),
                                     (axes[1], std, IMAGENET_STD, "Channel standard deviation")]:
        ax.bar(positions - 0.19, cub, width=0.34, color=MAIN_COLOR, label="CUB train")
        ax.bar(positions + 0.19, imagenet, width=0.34, color=SECOND_COLOR, label="ImageNet")
        ax.set_xticks(positions, ["Red", "Green", "Blue"])
        ax.grid(axis="x", visible=False)
        ax.set_title(title)
        ax.set_ylabel("Pixel value (0-1)")
        ax.margins(y=0.2)  # room for the legend above the bars
        ax.legend(loc="upper right", ncols=2)
    save(fig, output_dir, "13_channel_statistics")


# ---------------------------------------------------------------- captions

def tokenize_raw(caption):
    """Lowercase and keep letters only: no cleaning rules, so the statistics show the raw data."""
    return re.sub(r"[^a-z\s]", " ", caption.lower()).split()


def build_caption_df(dataset, split):
    rows = []
    for idx, (description, label) in enumerate(zip(dataset["description"], dataset["label"])):
        for caption_no, caption in enumerate(split_captions(description)):
            rows.append({
                "split": split,
                "index": idx,
                "label": label,
                "caption_no": caption_no,
                "caption": caption,
                "n_words": len(tokenize_raw(caption)),
                "n_chars": len(caption),
            })
    return pd.DataFrame(rows)


def edits1(word):
    """Every string one delete, swap, replacement or insertion away from `word`."""
    pairs = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    return (
        {a + b[1:] for a, b in pairs if b}
        | {a + b[1] + b[0] + b[2:] for a, b in pairs if len(b) > 1}
        | {a + c + b[1:] for a, b in pairs if b for c in ascii_lowercase}
        | {a + c + b for a, b in pairs for c in ascii_lowercase}
    )


def find_suspected_typos(word_counts, max_count=MIN_FREQ - 1, reference_freq=TYPO_REFERENCE_FREQ):
    """Rare words that look like a misspelling or two merged words of a common word.

    A heuristic, so it is an estimate: real rare words can still match ("entirety" -> "entirely"),
    and short typos or typos with two or more mistakes are missed. To keep real words out, the first
    letter must match or be swapped with the second ("ehite" -> "white", but not "come" -> "some"),
    and plain word endings do not count ("hairs" -> "hair").
    """
    common = {w for w, c in word_counts.items() if c >= reference_freq}

    def is_likely_typo_of(word, target):
        same_start = word[0] == target[0] or word[:2] == target[1::-1]
        is_inflection = word.startswith(target) and word[len(target):] in WORD_ENDINGS
        return same_start and not is_inflection

    rows = []
    for word, count in word_counts.items():
        if count > max_count or len(word) < 5:
            continue
        close = [w for w in edits1(word) if w in common and is_likely_typo_of(word, w)]
        # Parts need 3+ letters ("a" is the exception): stray tokens like the "s" of "bird's" are common
        merged = [(word[:i], word[i:]) for i in range(1, len(word) - 2)
                  if (i >= 3 or word[:i] == "a") and word[:i] in common and word[i:] in common]
        if close:
            best = max(close, key=lambda w: word_counts[w])
            rows.append({"word": word, "count": count, "kind": "misspelling", "likely": best})
        elif merged:
            a, b = max(merged, key=lambda p: min(word_counts[p[0]], word_counts[p[1]]))
            rows.append({"word": word, "count": count, "kind": "merged words", "likely": f"{a} {b}"})
    columns = ["word", "count", "kind", "likely"]
    return pd.DataFrame(rows, columns=columns).sort_values(["count", "word"], ascending=[False, True])


def analyze_caption_format(datasets, train_caps, test_caps, class_names, output_dir):
    section("CAPTIONS: FORMAT, CHARACTERS AND PUNCTUATION")
    train = datasets["train"]
    print(f"Raw record 0 ({class_names[train[0]['label']]}):")
    print(repr(train["description"][0]))

    print("\nTraining captions:", len(train_caps))
    print("Test captions:", len(test_caps))
    for name, caps, dataset in [("Training", train_caps, train), ("Test", test_caps, datasets["test"])]:
        per_image = caps.groupby("index").size().reindex(range(len(dataset)), fill_value=0)
        print(f"{name}: images by caption count {per_image.value_counts().sort_index().to_dict()}")

    # Every character that is not a lowercase letter or a space
    special_chars = Counter(ch for ch in "".join(train_caps["caption"]) if not ("a" <= ch <= "z" or ch == " "))
    print("\nSpecial characters:")
    print(pd.DataFrame(special_chars.most_common(), columns=["character", "count"]).to_string(index=False))

    captions = train_caps["caption"]
    checks = {
        "contains uppercase": captions.str.contains(r"[A-Z]"),
        "contains digits": captions.str.contains(r"\d"),
        "contains comma": captions.str.contains(","),
        "ends with a period": captions.str.endswith("."),
        "contains non-ASCII": captions.str.contains(r"[^\x00-\x7F]"),
    }
    print()
    print(pd.DataFrame({
        "captions": {k: int(v.sum()) for k, v in checks.items()},
        "percent": {k: round(v.mean() * 100, 2) for k, v in checks.items()},
    }).to_string())

    print("\nExamples with digits or non-ASCII characters:")
    for caption in captions[checks["contains digits"] | checks["contains non-ASCII"]].head(10):
        print(" -", caption)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    top_chars = special_chars.most_common(15)
    hbars(axes[0], [repr(ch) for ch, _ in top_chars], [c for _, c in top_chars],
          "Most common special characters (train)", "Occurrences (log scale)")
    axes[0].set_xscale("log")
    ordered = sorted(checks, key=lambda k: -checks[k].mean())
    hbars(axes[1], ordered, [checks[k].mean() * 100 for k in ordered],
          "Caption format checks (train)", "Share of captions (%)")
    save(fig, output_dir, "14_caption_characters")


def analyze_caption_length(train_caps, output_dir):
    section("CAPTIONS: LENGTH")
    lengths = train_caps["n_words"]
    print(lengths.describe().round(2).to_string())
    for q in [0.90, 0.95, 0.99]:
        print(f"{int(q * 100)}th percentile: {lengths.quantile(q):.0f} words")

    print("\nCaptions that fit a maximum length:")
    for n in [15, 20, 25, 30, 40]:
        print(f"  {n} words: {(lengths <= n).mean() * 100:.2f}%")

    print("\nShortest captions:")
    for _, row in train_caps.nsmallest(10, "n_words").iterrows():
        print(f"  [{row['n_words']} words] {row['caption']!r}")
    print("\nLongest captions:")
    for _, row in train_caps.nlargest(5, "n_words").iterrows():
        print(f"  [{row['n_words']} words] {row['caption']!r}")

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.hist(lengths, bins=range(0, lengths.max() + 2), color=MAIN_COLOR, edgecolor="#fcfcfb", linewidth=0.8)
    ax.grid(axis="x", visible=False)
    top = ax.get_ylim()[1]
    # The two lines are close: one label goes on the left of its line, the other on the right
    for value, label, offset, align in [(lengths.quantile(0.95), "95th percentile", -0.4, "right"),
                                        (MAX_WORDS, f"limit used: {MAX_WORDS}", 0.4, "left")]:
        ax.axvline(value, color=MUTED_INK, linewidth=1)
        ax.text(value + offset, top * 0.92, label, color=MUTED_INK, fontsize=9, ha=align)
    ax.set_title("Caption length (train)")
    ax.set_xlabel("Words per caption")
    ax.set_ylabel("Captions")
    save(fig, output_dir, "15_caption_length")


def analyze_vocabulary(train_caps, test_caps, output_dir):
    section("CAPTIONS: VOCABULARY")
    word_counts = Counter(word for caption in train_caps["caption"] for word in tokenize_raw(caption))
    total = sum(word_counts.values())
    print("Total words (tokens):", total)
    print("Unique words (vocabulary):", len(word_counts))

    # Effect of a minimum-frequency threshold
    thresholds = [1, 2, 3, 5, 10]
    sizes, coverage = [], []
    for min_freq in thresholds:
        kept = [c for c in word_counts.values() if c >= min_freq]
        sizes.append(len(kept))
        coverage.append(sum(kept) / total * 100)
    print("\nmin_freq  vocab_size  tokens_covered_%")
    for row in zip(thresholds, sizes, coverage):
        print(f"{row[0]:>8}  {row[1]:>10}  {row[2]:>16.3f}")

    rare_words = [w for w, c in word_counts.items() if c == 1]
    rng = np.random.default_rng(42)
    print("\nWords that appear only once:", len(rare_words))
    print("Random sample of 50:", sorted(rng.choice(rare_words, size=min(50, len(rare_words)), replace=False).tolist()))

    # How many test words would become <unk> with a vocabulary from training captions only
    vocab = {w for w, c in word_counts.items() if c >= MIN_FREQ}
    test_counts = Counter(word for caption in test_caps["caption"] for word in tokenize_raw(caption))
    unknown = Counter({w: c for w, c in test_counts.items() if w not in vocab})
    print(f"\nTest tokens that would be <unk> (min_freq={MIN_FREQ}): "
          f"{sum(unknown.values()) / sum(test_counts.values()) * 100:.2f}%")
    print("Most common unknown test words:", unknown.most_common(20))

    fig, axes = plt.subplots(1, 3, figsize=(18, 7.5))
    top_all = word_counts.most_common(30)
    hbars(axes[0], [w for w, _ in top_all], [c for _, c in top_all], "Top 30 words")
    top_content = [(w, c) for w, c in word_counts.most_common() if w not in ENGLISH_STOP_WORDS][:30]
    hbars(axes[1], [w for w, _ in top_content], [c for _, c in top_content], "Top 30 words (stop words removed)")
    labels = [str(t) for t in thresholds]
    axes[2].bar(labels, sizes, color=MAIN_COLOR, width=0.6)
    for label, size, covered in zip(labels, sizes, coverage):
        axes[2].annotate(f"{covered:.2f}% of\ntokens covered", (label, size), xytext=(0, 4),
                         textcoords="offset points", ha="center", fontsize=8, color=MUTED_INK)
    axes[2].margins(y=0.12)
    axes[2].grid(axis="x", visible=False)
    axes[2].set_title("Vocabulary size by minimum word frequency")
    axes[2].set_xlabel("Minimum word frequency")
    axes[2].set_ylabel("Words kept")
    save(fig, output_dir, "16_vocabulary")

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = sorted(COLOR_WORDS, key=lambda c: -word_counts.get(c, 0))
    hbars(ax, colors, [word_counts.get(c, 0) for c in colors], "Color words in training captions")
    save(fig, output_dir, "17_color_words")
    print(f"\nSpelling variants: grey {word_counts['grey']} vs gray {word_counts['gray']}, "
          f"greyish {word_counts['greyish']} vs grayish {word_counts['grayish']}, "
          f"colored {word_counts['colored']} vs coloured {word_counts['coloured']}")

    return word_counts


def analyze_typos(word_counts, output_dir):
    section("CAPTIONS: TYPOS (heuristic estimate)")
    typos = find_suspected_typos(word_counts)
    rare = sum(1 for c in word_counts.values() if c < MIN_FREQ)
    print(f"Words rarer than min_freq={MIN_FREQ}: {rare}")
    print(f"Suspected typos among them: {len(typos)} words, {typos['count'].sum()} occurrences "
          f"({typos['count'].sum() / sum(word_counts.values()) * 100:.3f}% of all tokens)")
    print(typos["kind"].value_counts().to_string())
    print("\nRandom sample of 40 (check how many are really typos):")
    print(typos.sample(min(40, len(typos)), random_state=42).sort_values("word").to_string(index=False))

    # Rare words by how often they appear: typos should pile up among the rarest
    occurrences = np.arange(1, MIN_FREQ)
    rare_counts = pd.Series([c for c in word_counts.values() if c < MIN_FREQ]).value_counts()
    suspected = typos["count"].value_counts().reindex(occurrences, fill_value=0)
    other = rare_counts.reindex(occurrences, fill_value=0) - suspected

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(occurrences - 0.19, other, width=0.34, color=MAIN_COLOR, label="other rare words")
    ax.bar(occurrences + 0.19, suspected, width=0.34, color=SECOND_COLOR, label="suspected typos")
    ax.set_xticks(occurrences)
    ax.grid(axis="x", visible=False)
    ax.set_title("Rare words in training captions")
    ax.set_xlabel("Times the word appears")
    ax.set_ylabel("Distinct words")
    ax.legend()
    save(fig, output_dir, "18_suspected_typos")


def analyze_noisy_captions(train_caps, output_dir):
    section("CAPTIONS: EMPTY, DUPLICATE AND REPEATED")
    print("Empty captions:", (train_caps["n_words"] == 0).sum())
    print("Very short captions (3 words or fewer):", (train_caps["n_words"] <= 3).sum())

    normalized = train_caps["caption"].map(lambda c: " ".join(tokenize_raw(c)))
    print("Duplicate captions within the same image:",
          train_caps.assign(norm=normalized).duplicated(["index", "norm"]).sum())

    repeats = normalized.value_counts()
    print(f"Captions whose text appears more than once: {(normalized.map(repeats) > 1).sum()} "
          f"({(normalized.map(repeats) > 1).mean() * 100:.1f}%)")
    top = repeats.head(10)
    print("\nMost repeated captions:")
    for caption, count in top.items():
        species = train_caps.loc[normalized == caption, "label"].nunique()
        print(f"  {count:>4}x across {species:>3} species: {caption}")

    fig, ax = plt.subplots(figsize=(11, 5))
    hbars(ax, top.index.tolist(), top.tolist(), "Most repeated captions (train)", "Occurrences")
    save(fig, output_dir, "19_repeated_captions")


def main():
    parser = argparse.ArgumentParser(description="Exploratory data analysis of CUB-200")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="where plots and statistics are saved")
    parser.add_argument("--recompute", action="store_true", help="ignore the cached image statistics")
    args = parser.parse_args()

    # Captions contain U+FFFD, which the default Windows console encoding cannot print
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    plt.switch_backend("Agg")  # plots go to files, no window
    plt.rcParams.update(STYLE)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset, test_dataset = load_cub_dataset()
    datasets = {"train": train_dataset, "test": test_dataset}
    class_names = train_dataset.features["label"].names

    analyze_classes(datasets, class_names, args.output_dir)

    all_stats = load_image_stats(datasets, class_names, args.output_dir / STATS_FILE, args.recompute)
    analyze_corrupted(all_stats)
    stats = all_stats[all_stats["error"].isna()]  # everything below needs a readable image
    analyze_sizes(stats, datasets, args.output_dir)
    analyze_color_modes(stats, datasets, args.output_dir)
    analyze_quality(stats, datasets, args.output_dir)
    analyze_duplicates(stats, datasets, args.output_dir)
    analyze_channels(stats, args.output_dir)

    train_caps = build_caption_df(train_dataset, "train")
    test_caps = build_caption_df(test_dataset, "test")
    analyze_caption_format(datasets, train_caps, test_caps, class_names, args.output_dir)
    analyze_caption_length(train_caps, args.output_dir)
    word_counts = analyze_vocabulary(train_caps, test_caps, args.output_dir)
    analyze_typos(word_counts, args.output_dir)
    analyze_noisy_captions(train_caps, args.output_dir)

    print(f"\nPlots saved to {args.output_dir}")


if __name__ == "__main__":
    main()
