"""Train on MNIST, select by validation accuracy, then evaluate the test set."""

import argparse
import csv
import json
import os
import platform
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST

from data_utils import MEAN, STD, mnist_transform
from model import DigitCNN


def run_epoch(model, loader, criterion, device, optimizer=None):
    """Return sample-weighted mean loss and accuracy for one full pass."""
    model.train(optimizer is not None)
    total_loss, correct, count = 0.0, 0, 0
    with torch.set_grad_enabled(optimizer is not None):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            count += labels.size(0)
    return total_loss / count, correct / count


@torch.inference_mode()
def evaluate_test(model, loader, criterion, device):
    model.eval()
    matrix = torch.zeros(10, 10, dtype=torch.int64)
    examples, mistakes = [], []
    total_loss, count = 0.0, 0
    for images, labels in loader:
        logits = model(images.to(device))
        predictions = logits.argmax(1).cpu()
        total_loss += criterion(logits, labels.to(device)).item() * len(labels)
        count += len(labels)
        matrix += torch.bincount(labels * 10 + predictions, minlength=100).reshape(10, 10)
        for image, label, prediction in zip(images, labels, predictions):
            item = (image.squeeze().numpy() * STD + MEAN, label.item(), prediction.item())
            if len(examples) < 16:
                examples.append(item)
            if label != prediction and len(mistakes) < 16:
                mistakes.append(item)
    return total_loss / count, matrix.diag().sum().item() / count, matrix, examples, mistakes


def save_figures(history, matrix, examples, mistakes, output):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    epochs = [row["epoch"] for row in history]
    for split in ("train", "val"):
        axes[0].plot(epochs, [row[f"{split}_loss"] for row in history], "o-", label=split)
        axes[1].plot(epochs, [row[f"{split}_accuracy"] for row in history], "o-", label=split)
    for axis, title in zip(axes, ("Cross-entropy loss", "Accuracy")):
        axis.set(xlabel="Epoch", title=title, xticks=epochs)
        axis.legend()
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "learning_curves.png", dpi=150)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 6))
    heatmap = axis.imshow(matrix.numpy(), cmap="Blues")
    for i in range(10):
        for j in range(10):
            axis.text(j, i, str(matrix[i, j].item()), ha="center", va="center",
                      color="white" if matrix[i, j] > matrix.max() / 2 else "black", fontsize=8)
    axis.set(xlabel="Predicted digit", ylabel="True digit", title="MNIST test confusion matrix",
             xticks=range(10), yticks=range(10))
    fig.colorbar(heatmap, ax=axis)
    fig.tight_layout()
    fig.savefig(output / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    for items, name, title in ((examples, "predictions", "First 16 test samples"),
                               (mistakes, "mistakes", "First misclassified test samples")):
        fig, axes = plt.subplots(4, 4, figsize=(7, 7))
        for axis in axes.flat:
            axis.axis("off")
        for axis, (image, label, prediction) in zip(axes.flat, items):
            axis.imshow(image, cmap="gray", vmin=0, vmax=1)
            axis.set_title(f"True {label} / Pred {prediction}", fontsize=10,
                           color="green" if label == prediction else "firebrick")
        fig.suptitle(title if items else "No misclassified test samples")
        fig.tight_layout()
        fig.savefig(output / f"{name}.png", dpi=150)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best_model.pt"))
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.threads) < 1 or args.lr <= 0:
        parser.error("epochs, batch-size, threads and lr must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is not available; use --device cpu")
    if args.device == "cuda":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    if args.device == "cuda":
        torch.backends.cudnn.benchmark = False
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)

    dataset = MNIST(args.data_dir, train=True, download=True, transform=mnist_transform())
    train_set, val_set = random_split(dataset, [55000, 5000],
                                      generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(args.seed))
    val_loader = DataLoader(val_set, batch_size=args.batch_size)
    test_set = MNIST(args.data_dir, train=False, download=True, transform=mnist_transform())
    test_loader = DataLoader(test_set, batch_size=args.batch_size)
    model = DigitCNN().to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    history, best_accuracy, best_epoch = [], -1.0, 0
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = run_epoch(model, train_loader, criterion, args.device, optimizer)
        val_loss, val_accuracy = run_epoch(model, val_loader, criterion, args.device)
        history.append(dict(epoch=epoch, train_loss=train_loss, train_accuracy=train_accuracy,
                            val_loss=val_loss, val_accuracy=val_accuracy))
        print(f"Epoch {epoch}/{args.epochs} | train loss={train_loss:.4f} acc={train_accuracy:.2%}"
              f" | val loss={val_loss:.4f} acc={val_accuracy:.2%}", flush=True)
        if val_accuracy > best_accuracy:
            best_accuracy, best_epoch = val_accuracy, epoch
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch,
                        "val_accuracy": val_accuracy, "seed": args.seed}, args.checkpoint)

    training_seconds = time.perf_counter() - started
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_accuracy, matrix, examples, mistakes = evaluate_test(
        model, test_loader, criterion, args.device)
    with (args.output_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    metrics = {
        "dataset": "MNIST", "train_samples": len(train_set), "val_samples": len(val_set),
        "test_samples": len(test_set), "epochs": args.epochs, "best_epoch": best_epoch,
        "batch_size": args.batch_size, "learning_rate": args.lr, "seed": args.seed,
        "device": args.device, "threads": args.threads,
        "parameters": sum(p.numel() for p in model.parameters()),
        "best_val_accuracy": best_accuracy, "test_loss": test_loss, "test_accuracy": test_accuracy,
        "training_seconds": round(training_seconds, 2), "python": platform.python_version(),
        "torch": str(torch.__version__), "torchvision": str(torchvision.__version__),
        "platform": platform.system(), "confusion_matrix": matrix.tolist(),
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    save_figures(history, matrix, examples, mistakes, args.output_dir)
    print(f"Best epoch: {best_epoch} | Test accuracy: {test_accuracy:.2%} | "
          f"Training: {training_seconds:.1f}s", flush=True)
    print(f"Checkpoint: {args.checkpoint} | Results: {args.output_dir}")


if __name__ == "__main__":
    main()
