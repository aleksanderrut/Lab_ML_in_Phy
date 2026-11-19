#!/usr/bin/env python3

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
PLOT_MODEL_COMPARISON = False
PLOT_PAIR_COMPARISON = False  # NEW: two models on the same loss/accuracy figure

# Plot all learned convolution kernels of CNN_Small in two separate figures.
PLOT_CNN_SMALL_FILTERS = True
# None: automatically prefer cnn_small_last.pth, then cnn_small_best.pth.
# You can also provide a filename or a full path, e.g. "cnn_small_best.pth".
CNN_SMALL_CHECKPOINT = r"C:\Users\aleks\Desktop\PWR\uczenie maszynowe w fizyce\Lab_2\Projekt2\result_1\cnn_small_last.pt"

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
# PLOT 5 - LEARNED CONVOLUTION FILTERS FOR CNN_SMALL
# ============================================================

def resolve_cnn_small_checkpoint():
    """Find the CNN_Small checkpoint in RESULTS_DIR."""
    if CNN_SMALL_CHECKPOINT is not None:
        path = Path(CNN_SMALL_CHECKPOINT)
        if not path.is_absolute():
            path = RESULTS_DIR / path
        if not path.is_file():
            raise FileNotFoundError(f"CNN_Small checkpoint not found: {path}")
        return path

    # Prefer the final checkpoint. If it is unavailable, use the best one.
    preferred_paths = [
        RESULTS_DIR / "cnn_small_last.pth",
        RESULTS_DIR / "cnn_small_best.pth",
    ]
    for path in preferred_paths:
        if path.is_file():
            return path

    # If there is exactly one other CNN_Small checkpoint, use it.
    candidates = sorted(RESULTS_DIR.glob("cnn_small*.pth"))
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        names = ", ".join(path.name for path in candidates)
        raise FileNotFoundError(
            "Several CNN_Small checkpoints were found: "
            f"{names}. Set CNN_SMALL_CHECKPOINT explicitly."
        )

    raise FileNotFoundError(
        f"No CNN_Small checkpoint found in {RESULTS_DIR}. "
        "Expected cnn_small_last.pth or cnn_small_best.pth."
    )


def load_checkpoint_state_dict(checkpoint_path):
    """Load a state_dict from either a raw state_dict or a checkpoint dictionary."""
    import torch

    # Prefer the safer weights-only loading mode when it is available.
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True
        )
    except TypeError:
        # Compatibility with older PyTorch versions.
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    except Exception:
        # Fallback for checkpoints that contain a serialized nn.Module.
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )

    if isinstance(checkpoint, torch.nn.Module):
        return checkpoint.state_dict()

    if isinstance(checkpoint, dict):
        # Common formats used when saving training checkpoints.
        for key in ("model_state_dict", "state_dict", "model"):
            if key not in checkpoint:
                continue

            candidate = checkpoint[key]
            if isinstance(candidate, torch.nn.Module):
                return candidate.state_dict()
            if isinstance(candidate, dict):
                return candidate

        # The file may contain the state_dict directly.
        return checkpoint

    raise TypeError(
        "Unsupported checkpoint format. Expected a PyTorch model or state_dict."
    )

def get_convolution_weights(state_dict, layer_name, expected_shape):
    """Extract and validate one convolution layer's weight tensor."""
    exact_key = f"{layer_name}.weight"

    if exact_key in state_dict:
        weights = state_dict[exact_key]
    else:
        # Also support prefixes such as 'module.' or 'model.'.
        matching_keys = [
            key for key in state_dict
            if key.endswith(f"{layer_name}.weight")
        ]
        if not matching_keys:
            raise KeyError(
                f"Could not find {layer_name}.weight in the checkpoint. "
                f"Available keys: {list(state_dict.keys())}"
            )
        if len(matching_keys) > 1:
            raise KeyError(
                f"More than one weight tensor matches {layer_name}.weight: "
                f"{matching_keys}"
            )
        weights = state_dict[matching_keys[0]]

    if hasattr(weights, "detach"):
        weights = weights.detach().cpu().numpy()
    else:
        weights = np.asarray(weights)

    if weights.shape != expected_shape:
        raise ValueError(
            f"Unexpected shape of {layer_name}.weight: {weights.shape}. "
            f"For CNN_Small, expected {expected_shape}. "
            "Check whether the selected checkpoint belongs to CNN_Small."
        )

    return weights


def plot_cnn_small_filters():
    """Plot all learned kernels of CNN_Small in two clearly arranged figures."""
    checkpoint_path = resolve_cnn_small_checkpoint()
    state_dict = load_checkpoint_state_dict(checkpoint_path)

    # CNN_Small:
    # Conv1: 2 output filters, 1 input channel, kernel size 5x5.
    # Conv2: 4 output filters, 2 input channels, kernel size 5x5.
    conv1_weights = get_convolution_weights(
        state_dict, "conv1", expected_shape=(2, 1, 5, 5)
    )
    conv2_weights = get_convolution_weights(
        state_dict, "conv2", expected_shape=(4, 2, 5, 5)
    )

    print(f"Loading CNN_Small filters from: {checkpoint_path}")

    # --------------------------------------------------------
    # Figure 1: first convolutional layer (2 kernels).
    # --------------------------------------------------------
    vmax1 = float(np.max(np.abs(conv1_weights)))
    if vmax1 == 0:
        vmax1 = 1.0

    fig1 = plt.figure(figsize=(9, 4.8), constrained_layout=True)
    grid1 = fig1.add_gridspec(
        1, 3, width_ratios=[1, 1, 0.07]
    )
    axes1 = [
        fig1.add_subplot(grid1[0, 0]),
        fig1.add_subplot(grid1[0, 1]),
    ]
    cax1 = fig1.add_subplot(grid1[0, 2])

    for output_channel, ax in enumerate(axes1):
        image1 = ax.imshow(
            conv1_weights[output_channel, 0],
            cmap="gray",
            interpolation="nearest",
            vmin=-vmax1,
            vmax=vmax1,
        )
        ax.set_title(f"Filter {output_channel + 1}", fontsize=11, pad=9)
        ax.set_xlabel("Kernel column")
        ax.set_ylabel("Kernel row")
        ax.set_xticks(range(5))
        ax.set_yticks(range(5))
        ax.tick_params(labelsize=9)
        ax.set_aspect("equal")

    fig1.suptitle(
        "CNN_Small — First convolutional layer (Conv1)\n"
        "1 input channel | 2 output filters | 5 × 5 kernels",
        fontsize=13,
        fontweight="bold",
    )
    fig1.colorbar(image1, cax=cax1, label="Weight value")
    save_figure(fig1, "cnn_small_conv1_filters.png")

    # --------------------------------------------------------
    # Figure 2: second convolutional layer (8 kernel slices).
    # Rows correspond to output filters; columns to input channels.
    # --------------------------------------------------------
    vmax2 = float(np.max(np.abs(conv2_weights)))
    if vmax2 == 0:
        vmax2 = 1.0

    fig2 = plt.figure(figsize=(10, 12), constrained_layout=True)
    grid2 = fig2.add_gridspec(
        4, 3,
        width_ratios=[1, 1, 0.07],
        height_ratios=[1, 1, 1, 1],
    )
    axes2 = [
        [fig2.add_subplot(grid2[row, col]) for col in range(2)]
        for row in range(4)
    ]
    cax2 = fig2.add_subplot(grid2[:, 2])

    for output_channel in range(4):
        for input_channel in range(2):
            ax = axes2[output_channel][input_channel]
            image2 = ax.imshow(
                conv2_weights[output_channel, input_channel],
                cmap="gray",
                interpolation="nearest",
                vmin=-vmax2,
                vmax=vmax2,
            )

            # Label the two input-channel columns once, at the top.
            if output_channel == 0:
                ax.set_title(
                    f"Input channel {input_channel + 1}",
                    fontsize=11,
                    pad=10,
                )

            # Identify each output-filter row on the left.
            if input_channel == 0:
                ax.set_ylabel(
                    f"Output filter {output_channel + 1}\nKernel row",
                    fontsize=9,
                    labelpad=9,
                )
            else:
                ax.set_ylabel("Kernel row", fontsize=9)

            if output_channel == 3:
                ax.set_xlabel("Kernel column")

            ax.set_xticks(range(5))
            ax.set_yticks(range(5))
            ax.tick_params(labelsize=8)
            ax.set_aspect("equal")

    fig2.suptitle(
        "CNN_Small — Second convolutional layer (Conv2)\n"
        "4 output filters × 2 input channels | 5 × 5 kernels",
        fontsize=13,
        fontweight="bold",
    )
    fig2.colorbar(image2, cax=cax2, label="Weight value")
    save_figure(fig2, "cnn_small_conv2_filters.png")


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

    if PLOT_CNN_SMALL_FILTERS:
        plot_cnn_small_filters()

    if SHOW_PLOTS:
        plt.show()
    print("Done.")


if __name__ == "__main__":
    main()
