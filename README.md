# From Pixels to Language

## Objectives

This project aims to build an end-to-end multimodal deep learning system that performs three tasks:

- **Image Classification:** Train a CNN encoder from scratch and compare its performance with transfer learning.
- **Image Colorization:** Reconstruct color images from grayscale inputs.
- **Image Captioning:** Generate natural-language descriptions of images using a visual encoder and a sequence decoder.

The same visual encoder serves as the backbone for all three tasks.

## Dataset

The project uses the **CUB-200-2011 dataset with captions**, which contains images of 200 bird species, with ten human-written descriptions for each image.

### Loading the Dataset

Install the required library:

```bash
pip install datasets
```

```bash
Load the training and test splits using Hugging Face:

from datasets import load_datasettrain_dataset = load_dataset( "Multimodal-Fatima/CUB_train",split="train")
test_dataset = load_dataset("Multimodal-Fatima/CUB_test",split="test")
print(train_dataset[0])


```
The dataset includes the following fields:
- image: The bird photograph.
- label: The bird species.
- description: Ten captions separated by newline characters.
- file_name: The original image filename.
- id: The record identifier.
For image colorization, grayscale images will be generated from the original color images, which will serve as the target outputs.

**References**
- CUB-200-2011 Official Dataset Website
- CUB Training Dataset — Hugging Face
- CUB Test Dataset — Hugging Face
