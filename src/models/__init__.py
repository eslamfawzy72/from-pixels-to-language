"""Models: encoders (image -> feature map) and classifiers (encoder + head).

    encoder.py     build_encoder(name): from-scratch or ImageNet-pretrained encoders
    classifier.py  create_model(name, num_classes): encoder + classifier head
    alexnet.py, vgg.py, googlenet.py, resnet.py, blocks.py: the from-scratch architectures
"""

from .classifier import (ImageClassifier, available_models, count_parameters, create_model, pretrained_models,
                         upgrade_state_dict)
from .encoder import available_encoders, build_encoder
from .googlenet import GoogLeNetOutputs

__all__ = [
    "GoogLeNetOutputs",
    "ImageClassifier",
    "available_encoders",
    "available_models",
    "build_encoder",
    "count_parameters",
    "create_model",
    "pretrained_models",
    "upgrade_state_dict",
]
