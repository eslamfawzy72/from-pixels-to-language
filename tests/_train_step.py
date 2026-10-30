"""One forward/loss/backward/optimizer step for a single model.

Run as ``python -m tests._train_step <model_name>``. ``test_gradients`` runs
this in a fresh subprocess per model so the 140M-parameter VGG models do not
accumulate memory across tests, and a native crash fails one subtest instead
of the whole suite. Exits non-zero (with a traceback) if any check fails.
"""

from __future__ import annotations

import json
import sys

import torch
from torch import nn

from src.models import create_model
from src.training.losses import classification_loss

from ._utils import NUM_CLASSES, dummy_batch


def check_training_step(name: str) -> dict[str, float]:
    torch.manual_seed(0)
    model = create_model(name, num_classes=NUM_CLASSES).train()
    images, labels = dummy_batch()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    loss, logits = classification_loss(model(images), labels, nn.CrossEntropyLoss())
    assert loss.dim() == 0 and torch.isfinite(loss), f"bad loss: {loss}"
    assert tuple(logits.shape) == (images.shape[0], NUM_CLASSES), f"bad logits shape {tuple(logits.shape)}"

    optimizer.zero_grad()
    loss.backward()
    missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, f"parameters without gradients: {missing[:5]}"
    assert all(torch.isfinite(p.grad).all() for p in model.parameters()), "non-finite gradient"
    assert any(p.grad.abs().sum() > 0 for p in model.parameters()), "all gradients are zero"

    # Snapshot only the first and last parameters to keep peak memory low.
    params = list(model.parameters())
    watched = {0: params[0].detach().clone(), len(params) - 1: params[-1].detach().clone()}
    optimizer.step()
    for index, before in watched.items():
        assert not torch.equal(before, params[index]), f"parameter {index} unchanged by optimizer step"
    return {"loss": loss.item()}


if __name__ == "__main__":
    print(json.dumps({"model": sys.argv[1], **check_training_step(sys.argv[1])}))
