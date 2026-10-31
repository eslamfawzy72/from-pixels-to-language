# Data Loading and Exploratory Data Analysis (EDA)

This README documents the data-loading and exploratory analysis completed so far for the CUB image-to-language project in `notebooks/preprocessing.ipynb`.

## 1. Data Loading

The dataset is loaded using the Hugging Face `datasets` library through the project's data-loader module:

```python
from src.data.data_loader import load_cub_dataset

train_dataset, test_dataset = load_cub_dataset()
```

The function loads these two dataset repositories:

- `Multimodal-Fatima/CUB_train`, split `train`
- `Multimodal-Fatima/CUB_test`, split `test`

Hugging Face Datasets caches downloaded and prepared data locally by default. Later calls can reuse cached data when available, avoiding unnecessary downloads.

### Dataset fields

The training dataset contains these columns:

| Column | Description |
|---|---|
| `image` | Decoded image represented as a PIL image object |
| `description` | Text descriptions associated with the image |
| `label` | Numeric class label, mapped to a species name by the `ClassLabel` feature |
| `file_name` | Original image filename |
| `id` | Dataset record identifier |

The `label` feature defines **200 bird species**.

### Description field observation

The `description` field contains multiple captions joined in one string and separated by newline characters (`\n`). Some captions appear noisy or potentially inaccurate. Caption statistics are in the caption EDA section of `notebooks/eda.ipynb`; the cleaning rules are implemented in `src/data/captions.py` (see `README_preprocessing.md`).

## 2. Class Distribution Analysis

A table was created to compare the number of examples per species in the training and test splits. A horizontal bar chart was plotted for the training split.

### Training split

| Statistic | Result |
|---|---:|
| Minimum images per species | 29 |
| Maximum images per species | 30 |
| Mean images per species | 29.97 |
| Median images per species | 30 |
| Maximum-to-minimum ratio | 1.03 |

**Finding:** The training split is very well balanced across the 200 species. Class weighting or oversampling does not appear necessary solely to correct class imbalance.

### Test split

| Statistic | Result |
|---|---:|
| Minimum images per species | 11 |
| Maximum images per species | 30 |
| Mean images per species | 28.97 |
| Median images per species | 30 |
| Maximum-to-minimum ratio | 2.73 |

**Finding:** The test split is less uniform than the training split. The test data should not be rebalanced by duplicating or deleting records just to make class counts equal. Per-class results and macro-averaged evaluation metrics may help show performance across species. The reason for the uneven distribution can be investigated if needed.

## 3. Image Exploration

The dataset's `image` field returns PIL image objects. Image properties are being inspected before choosing a resizing or preprocessing strategy.

The analysis records:

- Image width and height
- Resolution (`width * height`)
- Aspect ratio (`width / height`)
- Image mode, such as RGB
- Image format, when available
- Whether the image can be accessed and has positive dimensions

Histograms were prepared for width, height, and aspect ratio. A grid of sample images was also used for visual inspection.

## 4. Preliminary Image Quality Analysis

OpenCV (`opencv-python`, imported as `cv2`) and NumPy were added to the workflow to calculate simple image-quality indicators:

- **Sharpness:** variance of the Laplacian. Lower values can indicate blur, but do not prove an image is blurry.
- **Brightness:** mean grayscale intensity, from 0 (black) to 255 (white).
- **Contrast:** standard deviation of grayscale intensity.
- **Resolution:** image width multiplied by image height.

Images are converted to RGB and then grayscale before calculating these metrics.

### Quality-check status

The training split contains **5,994 records**. An initial run reported all 5,994 quality checks as failed, but a subsequent output showed records marked `valid=True`. Therefore, the initial failure count should not be interpreted as evidence that all images are corrupted.

A displayed table also appeared to have misaligned columns. The DataFrame columns and values should be verified before interpreting summary statistics. The lowest sharpness scores should be reviewed visually before deciding whether any images need special handling.

**No images have been removed based on the quality scores.** Successful decoding and valid dimensions do not necessarily mean an image is visually high quality.



## 5. Work Completed

- Loaded the training and test splits through `src/data/data_loader.py`.
- Inspected the dataset columns, feature types, and sample records.
- Identified the 200 species represented by the class labels.
- Observed that each `description` field contains multiple newline-separated captions.
- Calculated class counts for the training and test splits.
- Plotted the training class distribution.
- Started exploring image dimensions, aspect ratios, formats, and modes.
- Implemented preliminary sharpness, brightness, contrast, and resolution measurements.


## 6. Next EDA Steps

1. Verify the quality DataFrame's column names and sample rows.
2. Review image-dimension and quality plots and visually inspect low-sharpness candidates.
3. Analyze caption counts, caption lengths, vocabulary, and common words.
4. Check for empty, duplicate, misspelled, noisy, or potentially mismatched captions.
5. Investigate duplicate image IDs, filenames, or image content.
6. Decide on conservative caption-cleaning rules.
7. Confirm the splitting strategy and create a validation set from training data if required.
8. Plan image preprocessing and training-only augmentations.

## 7. Current Status

The work so far is exploratory. No image records have been removed, no caption-cleaning rules have been applied, and no augmentation pipeline has been finalized. Those decisions should be based on the remaining EDA findings.
