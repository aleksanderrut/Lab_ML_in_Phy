#!/usr/bin/env python3

import argparse
import copy
import csv
import json
import os
import random
import time
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")  # no display required on SSH
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


# ---------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------

class Perceptron(nn.Module):
    """Single fully-connected layer: 784 -> 10."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(28 * 28, 10)

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        return F.log_softmax(self.fc(x), dim=1)


class Deep(nn.Module):
    """Fully-connected network: 784 -> 20 -> 10."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 20)
        self.fc2 = nn.Linear(20, 10)

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class CNN(nn.Module):
    """
    Convolutional network tuned for MNIST.
    Deliberately contains NO Dropout and uses NO weight decay.
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))       # 28 x 28
        x = F.relu(self.conv2(x))       # 28 x 28
        x = F.max_pool2d(x, 2)          # 14 x 14
        x = F.relu(self.conv3(x))       # 14 x 14
        x = F.max_pool2d(x, 2)          # 7 x 7
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        # No Dropout, as required by the task.
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------

def make_loaders(data_dir, batch_size, num_workers, seed, device):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    print(f"MNIST data directory: {data_dir}", flush=True)
    train_full = datasets.MNIST(
        root=str(data_dir), train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root=str(data_dir), train=False, download=True, transform=transform
    )

    generator = torch.Generator().manual_seed(seed)
    train_dataset, validation_dataset = random_split(
        train_full, [50000, 10000], generator=generator
    )

    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": (device.type == "cuda"),
    }
    if num_workers > 0:
        common["persistent_workers"] = True

    train_generator = torch.Generator().manual_seed(seed + 1)
    train_loader = DataLoader(
        train_dataset, shuffle=True, generator=train_generator, **common
    )
    validation_loader = DataLoader(
        validation_dataset, shuffle=False, **common
    )
    test_loader = DataLoader(
        test_dataset, shuffle=False, **common
    )

    print(
        f"Train: {len(train_dataset)} | "
        f"Validation: {len(validation_dataset)} | "
        f"Test: {len(test_dataset)}",
        flush=True
    )
    return train_loader, validation_loader, test_loader


# ---------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------

def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.inference_mode():
        for data, target in loader:
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            output = model(data)
            total_loss += F.nll_loss(
                output, target, reduction="sum"
            ).item()
            total_correct += output.argmax(dim=1).eq(target).sum().item()
            total_samples += target.size(0)

    return total_loss / total_samples, total_correct / total_samples


def train_one_epoch(model, loader, optimizer, device, epoch, model_name):
    model.train()
    total_loss = 0.0
    total_samples = 0

    for batch_idx, (data, target) in enumerate(loader, start=1):
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        batch_size = target.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        if batch_idx == 1 or batch_idx % 100 == 0:
            print(
                f"{model_name} | epoch {epoch} | "
                f"batch {batch_idx}/{len(loader)} | "
                f"loss {loss.item():.5f}",
                flush=True
            )

    return total_loss / total_samples


def train_model(
    model, model_name, optimizer, train_loader, validation_loader,
    device, epochs, output_dir, patience=None, scheduler=None
):
    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "learning_rate": [],
    }

    best_val_loss = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0
    checkpoint_path = output_dir / f"{model_name.lower()}_best.pt"

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, epoch, model_name
        )
        val_loss, val_accuracy = evaluate(model, validation_loader, device)

        current_lr = optimizer.param_groups[0]["lr"]
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)
        history["learning_rate"].append(current_lr)

        print(
            f"{model_name} | epoch {epoch}/{epochs} | "
            f"train loss={train_loss:.5f} | "
            f"val loss={val_loss:.5f} | "
            f"val accuracy={100 * val_accuracy:.3f}% | "
            f"lr={current_lr:.2e} | "
            f"time={time.time() - epoch_start:.1f}s",
            flush=True
        )

        if scheduler is not None:
            scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_name": model_name,
                    "epoch": best_epoch,
                    "best_val_loss": best_val_loss,
                    "model_state_dict": best_state,
                },
                checkpoint_path
            )
        else:
            epochs_without_improvement += 1

        if patience is not None and epochs_without_improvement >= patience:
            print(
                f"{model_name}: early stopping after {epoch} epochs "
                f"(best epoch: {best_epoch}).",
                flush=True
            )
            break

    model.load_state_dict(best_state)
    elapsed = time.time() - start_time

    save_history_csv(history, output_dir / f"{model_name.lower()}_history.csv")
    plot_history(history, model_name, output_dir / f"{model_name.lower()}_learning_curves.png")

    print(
        f"{model_name}: finished in {elapsed / 60:.1f} min; "
        f"best epoch={best_epoch}, best validation loss={best_val_loss:.5f}",
        flush=True
    )
    return history, best_epoch, elapsed, checkpoint_path


# ---------------------------------------------------------------------
# Saving results and plots
# ---------------------------------------------------------------------

def save_history_csv(history, path):
    keys = list(history.keys())
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        writer.writerows(zip(*(history[key] for key in keys)))


def plot_history(history, model_name, path):
    epochs = history["epoch"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(epochs, history["train_loss"], "o-", label="Training")
    axes[0].plot(epochs, history["val_loss"], "o-", label="Validation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("NLL loss")
    axes[0].set_title("Loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(
        epochs, 100 * np.asarray(history["val_accuracy"]),
        "o-", label="Validation accuracy"
    )
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy [%]")
    axes[1].set_title("Validation accuracy")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.suptitle(f"{model_name}: training history")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def get_predictions(model, loader, device):
    model.eval()
    all_targets = []
    all_predictions = []

    with torch.inference_mode():
        for data, target in loader:
            output = model(data.to(device, non_blocking=True))
            all_predictions.append(output.argmax(dim=1).cpu().numpy())
            all_targets.append(target.numpy())

    return np.concatenate(all_targets), np.concatenate(all_predictions)


def plot_confusion_matrix(model, loader, device, model_name, output_dir):
    targets, predictions = get_predictions(model, loader, device)

    cm = np.zeros((10, 10), dtype=np.int64)
    np.add.at(cm, (targets, predictions), 1)

    # Save raw matrix as CSV too.
    np.savetxt(
        output_dir / f"{model_name.lower()}_confusion_matrix.csv",
        cm, delimiter=",", fmt="%d"
    )

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, label="Number of images")

    ax.set(
        xticks=np.arange(10),
        yticks=np.arange(10),
        xticklabels=np.arange(10),
        yticklabels=np.arange(10),
        xlabel="Predicted digit",
        ylabel="True digit",
        title=f"{model_name}: confusion matrix (test set)"
    )

    threshold = cm.max() / 2
    for i in range(10):
        for j in range(10):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > threshold else "black",
                fontsize=8
            )

    fig.tight_layout()
    fig.savefig(
        output_dir / f"{model_name.lower()}_confusion_matrix.png",
        dpi=200, bbox_inches="tight"
    )
    plt.close(fig)

    off_diag = cm.copy()
    np.fill_diagonal(off_diag, 0)
    most_confused = np.argsort(off_diag.ravel())[::-1]

    print(f"\n{model_name}: most common confusions (true -> predicted):")
    printed = 0
    for flat_index in most_confused:
        true_digit, predicted_digit = np.unravel_index(flat_index, off_diag.shape)
        count = off_diag[true_digit, predicted_digit]
        if count == 0 or printed >= 10:
            break
        print(f"  {true_digit} -> {predicted_digit}: {count}")
        printed += 1

    return cm


def plot_model_comparison(results, output_dir):
    names = list(results.keys())
    accuracies = [100.0 * results[name]["test_accuracy"] for name in names]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, accuracies)
    ax.axhline(99.0, color="red", linestyle="--", label="99% target")
    ax.set_ylabel("Test accuracy [%]")
    ax.set_ylim(85, 100)
    ax.set_title("MNIST: comparison of models")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    for bar, value in zip(bars, accuracies):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.08,
            f"{value:.3f}%",
            ha="center",
            va="bottom"
        )

    fig.tight_layout()
    fig.savefig(output_dir / "model_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train MNIST models and save all results for SSH use."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("./data/MNIST/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("./mnist_results"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=min(20, os.cpu_count() or 1))
    parser.add_argument("--seed", type=int, default=123)

    parser.add_argument("--epochs-perceptron", type=int, default=20)
    parser.add_argument("--epochs-deep", type=int, default=30)
    parser.add_argument("--epochs-cnn", type=int, default=18)

    parser.add_argument("--deep-weight-decay", type=float, default=1e-3)
    parser.add_argument("--cnn-lr", type=float, default=1e-3)
    parser.add_argument("--cnn-patience", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.data_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.set_num_threads(args.threads)
    if hasattr(torch, "set_num_interop_threads"):
        try:
            torch.set_num_interop_threads(max(1, min(4, args.threads)))
        except RuntimeError:
            pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 72, flush=True)
    print("MNIST experiments", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"PyTorch: {torch.__version__}", flush=True)
    print(f"CPU threads: {args.threads}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"Output directory: {args.output_dir.resolve()}", flush=True)
    print("=" * 72, flush=True)

    with open(args.output_dir / "configuration.json", "w") as f:
        json.dump(vars(args) | {"device": str(device)}, f, indent=2, default=str)

    train_loader, validation_loader, test_loader = make_loaders(
        args.data_dir, args.batch_size, args.num_workers, args.seed, device
    )

    results = {}

    # 1. Perceptron: baseline model.
    torch.manual_seed(args.seed)
    perceptron = Perceptron().to(device)
    perceptron_optimizer = torch.optim.SGD(
        perceptron.parameters(), lr=0.05, momentum=0.5
    )
    _, best_epoch, elapsed, checkpoint = train_model(
        perceptron, "Perceptron", perceptron_optimizer,
        train_loader, validation_loader, device,
        args.epochs_perceptron, args.output_dir, patience=5
    )
    test_loss, test_accuracy = evaluate(perceptron, test_loader, device)
    plot_confusion_matrix(
        perceptron, test_loader, device, "Perceptron", args.output_dir
    )
    results["Perceptron"] = {
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "best_epoch": best_epoch,
        "training_seconds": elapsed,
        "checkpoint": str(checkpoint.name),
    }
    print(
        f"Perceptron TEST: loss={test_loss:.5f}, "
        f"accuracy={100 * test_accuracy:.3f}%",
        flush=True
    )

    # 2. Deep: L2 regularization via nonzero weight_decay.
    torch.manual_seed(args.seed)
    deep = Deep().to(device)
    deep_optimizer = torch.optim.SGD(
        deep.parameters(),
        lr=0.05,
        momentum=0.5,
        weight_decay=args.deep_weight_decay
    )
    _, best_epoch, elapsed, checkpoint = train_model(
        deep, "Deep_L2", deep_optimizer,
        train_loader, validation_loader, device,
        args.epochs_deep, args.output_dir, patience=7
    )
    test_loss, test_accuracy = evaluate(deep, test_loader, device)
    plot_confusion_matrix(
        deep, test_loader, device, "Deep_L2", args.output_dir
    )
    results["Deep_L2"] = {
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "best_epoch": best_epoch,
        "training_seconds": elapsed,
        "checkpoint": str(checkpoint.name),
    }
    print(
        f"Deep L2 TEST: loss={test_loss:.5f}, "
        f"accuracy={100 * test_accuracy:.3f}%",
        flush=True
    )

    # 3. CNN: no Dropout and no weight decay.
    torch.manual_seed(args.seed)
    cnn = CNN().to(device)
    cnn_optimizer = torch.optim.Adam(
        cnn.parameters(), lr=args.cnn_lr, weight_decay=0.0
    )
    cnn_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        cnn_optimizer, mode="min", factor=0.5, patience=2
    )
    _, best_epoch, elapsed, checkpoint = train_model(
        cnn, "CNN_no_dropout", cnn_optimizer,
        train_loader, validation_loader, device,
        args.epochs_cnn, args.output_dir,
        patience=args.cnn_patience, scheduler=cnn_scheduler
    )
    test_loss, test_accuracy = evaluate(cnn, test_loader, device)
    plot_confusion_matrix(
        cnn, test_loader, device, "CNN_no_dropout", args.output_dir
    )
    results["CNN_no_dropout"] = {
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "best_epoch": best_epoch,
        "training_seconds": elapsed,
        "checkpoint": str(checkpoint.name),
    }
    print(
        f"CNN TEST: loss={test_loss:.5f}, "
        f"accuracy={100 * test_accuracy:.3f}%",
        flush=True
    )

    # Summary.
    plot_model_comparison(results, args.output_dir)
    summary_path = args.output_dir / "summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model", "test_loss", "test_accuracy",
            "best_epoch", "training_seconds", "checkpoint"
        ])
        for name, values in results.items():
            writer.writerow([
                name,
                values["test_loss"],
                values["test_accuracy"],
                values["best_epoch"],
                values["training_seconds"],
                values["checkpoint"],
            ])

    with open(args.output_dir / "summary.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 72, flush=True)
    print("FINAL TEST RESULTS", flush=True)
    for name, values in results.items():
        print(
            f"{name:20s}: {100 * values['test_accuracy']:.3f}% "
            f"(best epoch {values['best_epoch']})",
            flush=True
        )

    cnn_accuracy = results["CNN_no_dropout"]["test_accuracy"]
    if cnn_accuracy > 0.99:
        print("Target achieved: CNN test accuracy > 99%.", flush=True)
    else:
        print(
            "CNN did not exceed 99% in this run. "
            "Try --epochs-cnn 25 or 30; use validation results to assess tuning.",
            flush=True
        )

    print(f"\nAll results saved in: {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
