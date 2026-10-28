"""Image classifier: encoder + linear head. Image (B, 3, 224, 224) -> 200 species scores (B, 200).

Usage:

    from src.models.classifier import build_model

    model = build_model("scratch")      # or "resnet50"
    logits = model(images)              # (B, 200), use with nn.CrossEntropyLoss
    model.encoder                       # the encoder alone, reused in Milestone 2
"""
from torch import nn

from src.models.encoder import ScratchCNN, pretrained_resnet50


class Classifier(nn.Module):
    """Any encoder + dropout + one linear layer. Returns logits (no softmax)."""

    def __init__(self, encoder, num_classes=200, dropout=0.5):
        super().__init__()
        self.encoder = encoder                                   # image -> (B, feature_dim)
        self.head = nn.Sequential(
            nn.Dropout(dropout),                                 # fights overfitting
            nn.Linear(encoder.feature_dim, num_classes),         # (B, feature_dim) -> (B, 200)
        )

    def forward(self, x):
        return self.head(self.encoder(x))                        # 200 scores (logits)


def build_model(name, num_classes=200, dropout=0.5):
    """Build a classifier by name: "scratch" or "resnet50"."""
    if name == "scratch":
        encoder = ScratchCNN()
    elif name == "resnet50":
        encoder = pretrained_resnet50()
    else:
        raise ValueError(f"Unknown model {name!r}. Choose 'scratch' or 'resnet50'.")
    return Classifier(encoder, num_classes, dropout)


if __name__ == "__main__":
    # Quick check: python -m src.models.classifier
    import torch

    x = torch.randn(2, 3, 224, 224)
    for name in ["scratch", "resnet50"]:
        model = build_model(name)
        params = sum(p.numel() for p in model.parameters())
        print(f"{name}: output {tuple(model(x).shape)}, {params / 1e6:.1f}M parameters")
