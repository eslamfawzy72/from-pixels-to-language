"""Image encoders: image (B, 3, 224, 224) -> feature vector (B, feature_dim).

Usage:

    from src.models.encoder import ScratchCNN, pretrained_resnet50

    encoder = ScratchCNN()              # trained from scratch
    encoder = pretrained_resnet50()     # transfer learning
    features = encoder(images)          # (B, encoder.feature_dim)
"""
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50


def conv_block(in_channels, out_channels):
    """Conv -> BatchNorm -> ReLU -> MaxPool (halves the image size)."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class ScratchCNN(nn.Module):
    """CNN trained from scratch. Image (B, 3, 224, 224) -> features (B, 512)."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(  # 224 -> 112 -> 56 -> 28 -> 14 -> 7
            conv_block(3, 32),
            conv_block(32, 64),
            conv_block(64, 128),
            conv_block(128, 256),
            conv_block(256, 512),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)  # (B, 512, 7, 7) -> (B, 512, 1, 1)
        self.feature_dim = 512

    def forward(self, x):
        return self.pool(self.features(x)).flatten(1)


def pretrained_resnet50():
    """ResNet-50 with ImageNet weights, last layer removed. Image -> features (B, 2048)."""
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = nn.Identity()  # drop the 1000-class ImageNet layer, keep the features
    model.feature_dim = 2048
    return model


if __name__ == "__main__":
    # Quick check: python -m src.models.encoder
    import torch

    x = torch.randn(2, 3, 224, 224)
    for name, encoder in [("scratch", ScratchCNN()), ("resnet50", pretrained_resnet50())]:
        params = sum(p.numel() for p in encoder.parameters())
        print(f"{name}: output {tuple(encoder(x).shape)}, {params / 1e6:.1f}M parameters")
