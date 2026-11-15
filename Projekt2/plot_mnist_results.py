#!/usr/bin/env python3
"""
Generate plots locally from the CSV/JSON files produced by CNNs_ssh.py.

Usage:
    python plot_mnist_results.py --input-dir ./mnist_results
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def load_history(path):
    """Load one model's training history CSV."""
    history = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"No data in history file: {path}")

    for key in reader.fieldnames:
        values = [row[key] for row in rows]
        if key == "epoch":
            history[key] = [int(value) for value in values]
        else:
            history[key] = [float(value) for value in values]
    return history


def plot_history(history, model_name, output_path):
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
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(cm, model_name, output_path):
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
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_model_comparison(results, output_path):
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
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Generate MNIST plots from results saved by the SSH training script."
    )
    parser.add_argument(
        "--input-dir", type=Path, default=Path("./mnist_results"),
        help="Directory containing history CSVs, confusion-matrix CSVs and summary.json."
    )
    args = parser.parse_args()

    input_dir = args.input_dir
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Results directory does not exist: {input_dir}")

    model_files = [
        ("Perceptron", "perceptron"),
        ("Deep_L2", "deep_l2"),
        ("CNN_no_dropout", "cnn_no_dropout"),
    ]

    for model_name, file_stem in model_files:
        history_path = input_dir / f"{file_stem}_history.csv"
        confusion_path = input_dir / f"{file_stem}_confusion_matrix.csv"

        if history_path.exists():
            history = load_history(history_path)
            plot_history(
                history, model_name,
                input_dir / f"{file_stem}_learning_curves.png"
            )
            print(f"Saved: {file_stem}_learning_curves.png")
        else:
            print(f"Warning: missing {history_path.name}; skipping learning curves.")

        if confusion_path.exists():
            cm = np.loadtxt(confusion_path, delimiter=",", dtype=np.int64)
            plot_confusion_matrix(
                cm, model_name,
                input_dir / f"{file_stem}_confusion_matrix.png"
            )
            print(f"Saved: {file_stem}_confusion_matrix.png")
        else:
            print(f"Warning: missing {confusion_path.name}; skipping confusion matrix.")

    summary_path = input_dir / "summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            results = json.load(f)
        plot_model_comparison(results, input_dir / "model_comparison.png")
        print("Saved: model_comparison.png")
    else:
        print(f"Warning: missing {summary_path.name}; skipping model comparison.")

    print(f"\\nPlots saved in: {input_dir.resolve()}")


if __name__ == "__main__":
    main()
