
#!/usr/bin/env python3

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. LOADING DATA
# ============================================================

def load_history(path):

    history = {}

    with open(path, "r", newline="") as f:

        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"Empty history file: {path}")

    for key in reader.fieldnames:

        values = [row[key] for row in rows]

        if key == "epoch":
            history[key] = [int(x) for x in values]
        else:
            history[key] = [float(x) for x in values]

    return history


def load_json(path):

    with open(path, "r") as f:
        return json.load(f)


# ============================================================
# 2. TRAINING HISTORY
# ============================================================

def plot_history(history, experiment_name, output_dir):

    epochs = history["epoch"]

    fig, axes = plt.subplots(
        1, 2,
        figsize=(13, 5)
    )

    # Loss.
    axes[0].plot(
        epochs,
        history["train_loss"],
        "o-",
        label="Training"
    )

    axes[0].plot(
        epochs,
        history["val_loss"],
        "o-",
        label="Validation"
    )

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("NLL loss")
    axes[0].set_title("Loss")

    axes[0].grid(alpha=0.3)
    axes[0].legend()

    # Accuracy.
    axes[1].plot(
        epochs,
        100 * np.asarray(history["val_accuracy"]),
        "o-",
        label="Validation accuracy"
    )

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy [%]")
    axes[1].set_title("Validation accuracy")

    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.suptitle(
        f"{experiment_name}: training history"
    )

    fig.tight_layout()

    fig.savefig(
        output_dir / "learning_curves.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# 3. CONFUSION MATRIX
# ============================================================

def plot_confusion_matrix(
    cm,
    class_names,
    experiment_name,
    output_dir
):

    num_classes = len(class_names)

    fig, ax = plt.subplots(
        figsize=(
            max(8, num_classes * 0.8),
            max(7, num_classes * 0.7)
        )
    )

    im = ax.imshow(
        cm,
        interpolation="nearest",
        cmap="Blues"
    )

    fig.colorbar(
        im,
        ax=ax,
        label="Number of images"
    )

    ax.set(
        xticks=np.arange(num_classes),
        yticks=np.arange(num_classes),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted class",
        ylabel="True class",
        title=f"{experiment_name}: confusion matrix"
    )

    plt.setp(
        ax.get_xticklabels(),
        rotation=45,
        ha="right",
        rotation_mode="anchor"
    )

    threshold = cm.max() / 2 if cm.max() > 0 else 0

    for i in range(num_classes):

        for j in range(num_classes):

            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color=(
                    "white"
                    if cm[i, j] > threshold
                    else "black"
                ),
                fontsize=8
            )

    fig.tight_layout()

    fig.savefig(
        output_dir / "confusion_matrix.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# 4. LEARNING RATE
# ============================================================

def plot_learning_rate(history, experiment_name, output_dir):

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        history["epoch"],
        history["learning_rate"],
        "o-"
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning rate")
    ax.set_title(f"{experiment_name}: learning rate")

    ax.set_yscale("log")
    ax.grid(alpha=0.3)

    fig.tight_layout()

    fig.savefig(
        output_dir / "learning_rate.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# 5. CLASS ACCURACY
# ============================================================

def plot_class_accuracy(
    cm,
    class_names,
    experiment_name,
    output_dir
):

    correct = np.diag(cm)

    total = cm.sum(axis=1)

    accuracy = np.divide(
        correct,
        total,
        out=np.zeros_like(correct, dtype=float),
        where=total != 0
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    bars = ax.bar(
        class_names,
        100 * accuracy
    )

    ax.set_ylabel("Accuracy [%]")
    ax.set_xlabel("Crystal net type")
    ax.set_title(f"{experiment_name}: accuracy per class")

    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)

    for bar, value in zip(bars, accuracy):

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            100 * value + 1,
            f"{100 * value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8
        )

    plt.xticks(rotation=45, ha="right")

    fig.tight_layout()

    fig.savefig(
        output_dir / "class_accuracy.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# 6. DIRECT VS FOURIER COMPARISON
# ============================================================

def plot_comparison(results, output_dir):

    names = ["direct", "fourier"]

    accuracies = [
        100 * results[name]["test_accuracy"]
        for name in names
    ]

    losses = [
        results[name]["test_loss"]
        for name in names
    ]

    fig, axes = plt.subplots(
        1, 2,
        figsize=(12, 5)
    )

    # Test accuracy.
    bars = axes[0].bar(
        names,
        accuracies
    )

    axes[0].set_ylabel("Accuracy [%]")
    axes[0].set_title("Test accuracy comparison")
    axes[0].set_ylim(0, 105)
    axes[0].grid(axis="y", alpha=0.3)

    for bar, value in zip(bars, accuracies):

        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            value + 1,
            f"{value:.2f}%",
            ha="center",
            va="bottom"
        )

    # Test loss.
    bars = axes[1].bar(
        names,
        losses
    )

    axes[1].set_ylabel("NLL loss")
    axes[1].set_title("Test loss comparison")
    axes[1].grid(axis="y", alpha=0.3)

    for bar, value in zip(bars, losses):

        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.4f}",
            ha="center",
            va="bottom"
        )

    fig.tight_layout()

    fig.savefig(
        output_dir / "direct_vs_fourier.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# 7. MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Generate plots for crystal CNN experiments."
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("./crystal_results")
    )

    args = parser.parse_args()

    input_dir = args.input_dir

    if not input_dir.is_dir():
        raise FileNotFoundError(
            f"Directory does not exist: {input_dir}"
        )

    for experiment_name in ["direct", "fourier"]:

        print(f"\nProcessing {experiment_name}...")

        experiment_dir = input_dir / experiment_name

        if not experiment_dir.is_dir():
            print(
                f"Warning: missing directory {experiment_dir}"
            )
            continue

        history_path = experiment_dir / "history.csv"

        confusion_path = (
            experiment_dir / "confusion_matrix.csv"
        )

        class_names_path = (
            experiment_dir / "class_names.json"
        )

        if history_path.exists():

            history = load_history(history_path)

            plot_history(
                history,
                experiment_name,
                experiment_dir
            )

            plot_learning_rate(
                history,
                experiment_name,
                experiment_dir
            )

            print("  Learning curves saved.")

        if (
            confusion_path.exists()
            and class_names_path.exists()
        ):

            cm = np.loadtxt(
                confusion_path,
                delimiter=",",
                dtype=np.int64,
                ndmin=2
            )

            class_names = load_json(class_names_path)

            plot_confusion_matrix(
                cm,
                class_names,
                experiment_name,
                experiment_dir
            )

            plot_class_accuracy(
                cm,
                class_names,
                experiment_name,
                experiment_dir
            )

            print("  Confusion matrix saved.")
            print("  Class accuracy saved.")

    comparison_path = input_dir / "comparison.json"

    if comparison_path.exists():

        results = load_json(comparison_path)

        if all(
            name in results
            for name in ["direct", "fourier"]
        ):

            plot_comparison(
                results,
                input_dir
            )

            print("\nDirect vs Fourier comparison saved.")

    print(
        f"\nAll plots saved in: {input_dir.resolve()}"
    )


if __name__ == "__main__":
    main()
