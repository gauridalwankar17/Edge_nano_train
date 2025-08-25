"""Train a SimpleCNN on CIFAR-10 and save the best model checkpoint.

Usage:
  python train_cnn.py --data-dir ./datasets --epochs 10 --batch-size 128 --out-dir ./artifacts

The script will save:
- best_model.pt: best validation accuracy model state
- labels.json: class names mapping
- metrics.json: per-epoch metrics for reference

TODO: You may tweak hyper-parameters such as learning rate, weight decay, and epochs.
"""

import argparse
import json
from pathlib import Path
from typing import Dict

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from common.data import build_cifar10_loaders
from common.model import SimpleCNN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SimpleCNN on CIFAR-10")
    parser.add_argument("--data-dir", type=str, default="./datasets", help="Dataset directory")
    parser.add_argument("--out-dir", type=str, default="./artifacts", help="Output directory for checkpoints")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device",
    )
    return parser.parse_args()


def evaluate(
    model: nn.Module, loader: torch.utils.data.DataLoader, device: torch.device
) -> Dict[str, float]:
    model.eval()
    total = 0
    correct = 0
    running_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            logits = model(images)
            loss = criterion(logits, targets)
            running_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += images.size(0)
    avg_loss = running_loss / max(total, 1)
    acc = correct / max(total, 1)
    return {"loss": avg_loss, "acc": acc}


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Data
    train_loader, val_loader, dataset_info = build_cifar10_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    labels_path = out_dir / "labels.json"
    with labels_path.open("w", encoding="utf-8") as f:
        json.dump({"classes": dataset_info.classes}, f, ensure_ascii=False, indent=2)

    device = torch.device(args.device)
    model = SimpleCNN(num_classes=dataset_info.num_classes).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    metrics = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        epoch_loss = 0.0
        seen = 0
        for images, targets in pbar:
            images = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * images.size(0)
            seen += images.size(0)
            pbar.set_postfix(loss=epoch_loss / max(seen, 1))

        scheduler.step()

        # Validation
        val_metrics = evaluate(model, val_loader, device)
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": epoch_loss / max(seen, 1),
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
        }
        metrics.append(epoch_metrics)

        # Save best
        if val_metrics["acc"] > best_acc:
            best_acc = val_metrics["acc"]
            torch.save({"model_state": model.state_dict()}, out_dir / "best_model.pt")

        # Persist metrics every epoch
        with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"Training complete. Best val acc: {best_acc:.4f}")
    print(f"Artifacts saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()