#!/usr/bin/env python3
"""Plot saved MNIST training results; no model training is performed."""

from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# SETTINGS - CHANGE ONLY HERE
# ============================================================

PROJECT_DIR = Path(r"C:\Users\aleks\Desktop\PWR\uczenie maszynowe w fizyce\Lab_2\Projekt2")
RESULTS_DIR = PROJECT_DIR / "result_1"  # Input .data files
FIGS_DIR = PROJECT_DIR / "figs"          # Generated plots

# Which plot types should be generated?
PLOT_CONFUSION = False
PLOT_LOSS_AND_ACCURACY = False
PLOT_MODEL_COMPARISON = True
PLOT_PAIR_COMPARISON = False  # NEW: two models on the same loss/accuracy figure

# Individual confusion matrices and learning curves.
MODELS_TO_PLOT = ["CNN_Large_Dropout"]

# Confusion matrix: "best" (lowest validation loss) or "last" (final epoch).
CONFUSION_EPOCH = "last"
# "counts", "percent", "both".
CONFUSION_FORMAT = "both"

# Pairs to compare on one figure with TWO subplots:
# left: training + validation loss for both models (4 curves)
# right: test accuracy for both models (2 curves)
# Comment out a pair to skip its comparison.
MODEL_PAIRS_TO_COMPARE = [
    #("Deep", "Deep_L2"),
    #("CNN_Small", "CNN_Small_Dropout"),
    #("CNN_Large", "CNN_Large_Dropout"),
]

# Models for the optional bar chart.
MODELS_TO_COMPARE = [
    "Perceptron",
    "Deep",
    "Deep_L2",
    "CNN_Small",
    "CNN_Small_Dropout",
    "CNN_Medium",
    "CNN_Large",
    "CNN_Large_Dropout",
]

# Bar chart mode: "max" or "last".
COMPARISON_ACCURACY_MODE = "max"
REFERENCE_ACCURACY = 99.0  # Horizontal reference line in the bar chart [%]
CONFUSION_CMAP = "Purples"

SAVE_DPI = 200
SHOW_PLOTS = True  # False: save PNG only


# ============================================================
# READING DATA
# ============================================================

def history_for(model_name):
    """Return columns: epoch, train_loss, val_loss, test_accuracy_percent."""
    path = RESULTS_DIR / f"{model_name.lower()}_history.data"
    if not path.is_file():
        raise FileNotFoundError(f"Missing history file: {path}")

    history = np.loadtxt(path, comments="#", ndmin=2)
    if history.shape[1] != 4 or history.shape[0] == 0:
        raise ValueError(
            f"Invalid file {path.name}: expected 4 columns "
            "(epoch, train_loss, val_loss, test_accuracy_percent)."
        )
    return history


def confusion_for(model_name, epoch_type, value_type):
    """Read a confusion matrix and its epoch number from the file header."""
    filename = f"{model_name.lower()}_confusion_{epoch_type}_{value_type}.data"
    path = RESULTS_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(f"Missing confusion matrix file: {path}")

    with path.open("r", encoding="utf-8") as f:
        header = f.readline().strip()
    match = re.search(r"\bepoch\s*=\s*(\d+)\b", header)
    if not match:
        raise ValueError(f"Missing epoch number in header of {filename}: {header!r}")
    epoch_number = int(match.group(1))

    matrix = np.loadtxt(path, comments="#")
    if matrix.shape != (10, 10):
        raise ValueError(f"Matrix {filename} has shape {matrix.shape} instead of (10, 10).")
    return matrix, epoch_number


# ============================================================
# PLOT 1 - CONFUSION MATRIX
# ============================================================

def plot_confusion(model_name, epoch_type, value_type):
    cm, epoch_number = confusion_for(model_name, epoch_type, value_type)
    is_percent = value_type == "percent"

    # For counts, use the largest OFF-DIAGONAL value as the color-scale maximum.
    if is_percent:
        color_max = 100.0
    else:
        off_diagonal = cm[~np.eye(cm.shape[0], dtype=bool)]
        color_max = max(float(off_diagonal.max()), 1.0)

    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(cm, cmap=CONFUSION_CMAP, interpolation="nearest",
                   vmin=0, vmax=color_max)

    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xlabel("Predicted digit")
    ax.set_ylabel("True digit")
    ax.set_title(
        f"{model_name} | Confusion matrix "
        f"({epoch_type}, epoch {epoch_number}, "
        f"{'percentages' if is_percent else 'counts'})"
    )
    fig.colorbar(im, ax=ax, label="Share of true class [%]" if is_percent else "Number of images")

    threshold = 0.5 * color_max
    for row in range(10):
        for col in range(10):
            value = f"{cm[row, col]:.1f}%" if is_percent else f"{int(cm[row, col])}"
            ax.text(col, row, value, ha="center", va="center", fontsize=8,
                    color="white" if cm[row, col] > threshold else "black")

    fig.tight_layout()
    save_figure(fig, f"{model_name.lower()}_confusion_{epoch_type}_{value_type}.png")


# ============================================================
# PLOT 2 - LOSS + ACCURACY FOR ONE MODEL
# ============================================================

def plot_learning_curves(model_name):
    history = history_for(model_name)
    epochs = history[:, 0]
    train_loss = history[:, 1]
    val_loss = history[:, 2]
    test_accuracy = history[:, 3]

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(13, 5))

    ax_loss.plot(epochs, train_loss, "o-", markersize=3, label="Training")
    ax_loss.plot(epochs, val_loss, "o-", markersize=3, label="Validation")
    ax_loss.set(xlabel="Epoch", ylabel="NLL loss", title="Training and validation loss")
    ax_loss.legend()
    ax_loss.grid(alpha=0.3)

    ax_acc.plot(epochs, test_accuracy, "o-", color="tab:orange",
                markersize=3, label="Test accuracy")
    ax_acc.set(xlabel="Epoch", ylabel="Accuracy [%]", title="Test accuracy")
    ax_acc.grid(alpha=0.3)
    ax_acc.legend()

    fig.suptitle(model_name)
    fig.tight_layout()
    save_figure(fig, f"{model_name.lower()}_loss_accuracy.png")


# ============================================================
# PLOT 3 - BAR CHART OF MODEL ACCURACIES
# ============================================================

def plot_comparison(model_names):
    if not model_names:
        raise ValueError("MODELS_TO_COMPARE is empty.")

    accuracies = []
    selected_epochs = []
    for model_name in model_names:
        history = history_for(model_name)
        if COMPARISON_ACCURACY_MODE == "max":
            selected_idx = int(np.argmax(history[:, 3]))
        else:
            selected_idx = int(np.argmax(history[:, 0]))

        accuracies.append(float(history[selected_idx, 3]))
        selected_epochs.append(int(history[selected_idx, 0]))

    fig, ax = plt.subplots(figsize=(max(9, 1.45 * len(model_names)), 6))
    x = np.arange(len(model_names))
    bars = ax.bar(x, accuracies)
    ax.axhline(REFERENCE_ACCURACY, linestyle="--", color="red", linewidth=1.5,
               label=f"Reference: {REFERENCE_ACCURACY:g}%")

    if COMPARISON_ACCURACY_MODE == "last":
        ax.set_ylabel("Final test accuracy [%]")
        ax.set_title("MNIST: test accuracy at the last epoch")
        output_filename = "comparison_last_test_accuracy.png"
    else:
        ax.set_ylabel("Maximum test accuracy [%]")
        ax.set_title("MNIST: highest test accuracy across epochs")
        output_filename = "comparison_max_test_accuracy.png"

    ax.set_xticks(x, model_names, rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend()

    lower_bound = max(0, min(85, 5 * np.floor(min(accuracies) / 5)))
    ax.set_ylim(lower_bound, max(100.5, max(accuracies) + 1.0))
    for bar, accuracy, epoch in zip(bars, accuracies, selected_epochs):
        ax.annotate(f"{accuracy:.2f}%\n(epoch {epoch})",
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    save_figure(fig, output_filename)


# ============================================================
# PLOT 4 - OVERLAY TWO MODELS: LOSS + ACCURACY
# ============================================================

def plot_pair_comparison(model_a, model_b):
    """Plot both models on one figure: 4 loss curves + 2 accuracy curves.

    Consistent visual encoding:
    - training loss: blue
    - validation loss: orange
    - accuracy of the first model: red
    - accuracy of the second model: green
    - baseline model: solid line
    - regularized variant (_L2 or _Dropout): dashed line
    """
    history_a = history_for(model_a)
    history_b = history_for(model_b)

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(14, 5.3))

    for index, (name, history) in enumerate(((model_a, history_a), (model_b, history_b))):
        epochs = history[:, 0]
        is_regularized = name.lower().endswith(("_dropout", "_l2"))
        linestyle = "--" if is_regularized else "-"
        accuracy_color = "tab:red" if index == 0 else "tab:green"

        ax_loss.plot(
            epochs, history[:, 1], color="tab:blue", linestyle=linestyle,
            label=f"{name} — training"
        )
        ax_loss.plot(
            epochs, history[:, 2], color="tab:orange", linestyle=linestyle,
            label=f"{name} — validation"
        )
        ax_acc.plot(
            epochs, history[:, 3], color=accuracy_color, linestyle=linestyle,
            label=name
        )

    ax_loss.set(xlabel="Epoch", ylabel="NLL loss", title="Training and validation loss")
    ax_loss.grid(alpha=0.3)
    ax_loss.legend(fontsize=9)

    ax_acc.set(xlabel="Epoch", ylabel="Test accuracy [%]", title="Test accuracy")
    ax_acc.grid(alpha=0.3)
    ax_acc.legend(fontsize=9)

    fig.suptitle(f"{model_a} vs {model_b}")
    fig.tight_layout()
    filename = f"comparison_{model_a.lower()}_vs_{model_b.lower()}_loss_accuracy.png"
    save_figure(fig, filename)


# ============================================================
# SAVING AND RUNNING
# ============================================================

def save_figure(fig, filename):
    path = FIGS_DIR / filename
    fig.savefig(path, dpi=SAVE_DPI, bbox_inches="tight")
    print(f"Saved: {path}")
    if not SHOW_PLOTS:
        plt.close(fig)


def main():
    if CONFUSION_EPOCH not in ("best", "last"):
        raise ValueError('CONFUSION_EPOCH must be "best" or "last".')
    if CONFUSION_FORMAT not in ("counts", "percent", "both"):
        raise ValueError('CONFUSION_FORMAT must be "counts", "percent", or "both".')
    if COMPARISON_ACCURACY_MODE not in ("max", "last"):
        raise ValueError('COMPARISON_ACCURACY_MODE must be "max" or "last".')
    if not RESULTS_DIR.is_dir():
        raise FileNotFoundError(f"Results folder not found: {RESULTS_DIR}")

    FIGS_DIR.mkdir(parents=True, exist_ok=True)

    for model_name in MODELS_TO_PLOT:
        if PLOT_CONFUSION:
            types = ("counts", "percent") if CONFUSION_FORMAT == "both" else (CONFUSION_FORMAT,)
            for value_type in types:
                plot_confusion(model_name, CONFUSION_EPOCH, value_type)
        if PLOT_LOSS_AND_ACCURACY:
            plot_learning_curves(model_name)

    if PLOT_MODEL_COMPARISON:
        plot_comparison(MODELS_TO_COMPARE)

    if PLOT_PAIR_COMPARISON:
        for model_a, model_b in MODEL_PAIRS_TO_COMPARE:
            plot_pair_comparison(model_a, model_b)

    if SHOW_PLOTS:
        plt.show()
    print("Done.")


if __name__ == "__main__":
    main()
