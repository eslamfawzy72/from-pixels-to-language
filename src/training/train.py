"""Training loop for the classifier.

Usage (from the project root):

    python -m src.training.train --model scratch --epochs 30
    python -m src.training.train --model resnet50 --epochs 15 --lr 1e-4

Or from Python:

    from src.training.train import fit
    history = fit(model, train_loader, val_loader, epochs=30, lr=1e-3, checkpoint_path="best.pt")

The best model (highest validation accuracy) is saved to the checkpoint. The test set is never used here.
"""
import random
import time

import numpy as np
import torch
from torch import nn


def set_seed(seed=42):
    """Same seed -> same weights init and same results, so runs can be compared."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, loss_fn, optimizer, scaler, device):
    """One pass over the training data. Returns (average loss, accuracy)."""
    model.train()  # dropout and batchnorm in training mode
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        # Mixed precision on GPU: faster and uses less memory. Does nothing on CPU.
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            logits = model(images)
            loss = loss_fn(logits, labels)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    """Loss and accuracy on a validation or test set. Returns (average loss, accuracy)."""
    model.eval()  # dropout off, batchnorm uses its running averages
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            logits = model(images)
            loss = loss_fn(logits, labels)
        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


def fit(model, train_loader, val_loader, epochs, lr, checkpoint_path,
        weight_decay=1e-4, patience=None, device=None, on_epoch_end=None):
    """Train for `epochs`, keep the weights with the best validation accuracy.

    patience:      stop early if validation accuracy has not improved for this many epochs (None = never).
    on_epoch_end:  optional function called as on_epoch_end(epoch, metrics), e.g. to log to MLflow.

    Returns the history: one dict of metrics per epoch.
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)  # lr slowly decays to 0
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")

    history = []
    best_acc, best_epoch = 0.0, 0
    for epoch in range(1, epochs + 1):
        start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, loss_fn, optimizer, scaler, device)
        val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)
        metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": time.time() - start,
        }
        scheduler.step()
        history.append(metrics)

        improved = val_acc > best_acc
        if improved:
            best_acc, best_epoch = val_acc, epoch
            torch.save({"state_dict": model.state_dict(), "epoch": epoch, "val_acc": val_acc}, checkpoint_path)

        print(f"epoch {epoch:3d}/{epochs} | train loss {train_loss:.3f} acc {train_acc:.3f} | "
              f"val loss {val_loss:.3f} acc {val_acc:.3f} | {metrics['seconds']:.0f}s"
              + (" | saved best" if improved else ""))

        if on_epoch_end is not None:
            on_epoch_end(epoch, metrics)

        if patience is not None and epoch - best_epoch >= patience:
            print(f"Early stopping: no improvement for {patience} epochs.")
            break

    print(f"Best validation accuracy {best_acc:.3f} at epoch {best_epoch}, saved to {checkpoint_path}")
    return history


def main():
    import argparse
    import json
    from pathlib import Path

    from src.data.dataset import get_dataloaders
    from src.models.classifier import build_model

    parser = argparse.ArgumentParser(description="Train a bird species classifier.")
    parser.add_argument("--model", default="scratch", choices=["scratch", "resnet50"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="outputs/runs")
    args = parser.parse_args()

    set_seed(args.seed)
    run_dir = Path(args.out_dir) / f"{args.model}_lr{args.lr}_{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, _ = get_dataloaders(args.batch_size, args.num_workers, seed=args.seed)
    model = build_model(args.model, dropout=args.dropout)

    history = fit(model, train_loader, val_loader, args.epochs, args.lr,
                  checkpoint_path=run_dir / "best.pt", weight_decay=args.weight_decay,
                  patience=args.patience)

    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=1))
    (run_dir / "history.json").write_text(json.dumps(history, indent=1))
    print(f"Run saved to {run_dir}")


if __name__ == "__main__":  # required on Windows when num_workers > 0
    main()
