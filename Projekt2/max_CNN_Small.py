#!/usr/bin/env python3
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.colors import Normalize

# ============================================================
CHECKPOINT = Path(
    r"C:\Users\aleks\Desktop\PWR\uczenie maszynowe w fizyce"
    r"\Lab_2\Projekt2\result_1\cnn_small_last.pt"
)
OUTPUT_DIR = Path(
    r"C:\Users\aleks\Desktop\PWR\uczenie maszynowe w fizyce"
    r"\Lab_2\Projekt2\figs\activation_maximization"
)

STEPS = 500
LEARNING_RATE = 0.05

# The model input remains in the [0, 1] range.
INPUT_MIN = 0.0
INPUT_MAX = 1.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
class CNN_Small(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 2, kernel_size=5)
        self.conv2 = nn.Conv2d(2, 4, kernel_size=5)
        self.fc1 = nn.Linear(64, 20)
        self.fc2 = nn.Linear(20, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(F.max_pool2d(x, 2))

        x = self.conv2(x)
        x = F.relu(F.max_pool2d(x, 2))

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))

        logits = self.fc2(x)
        return logits

# ============================================================
def extract_state_dict(checkpoint):
    """Support a raw state_dict and common checkpoint dictionaries."""
    if isinstance(checkpoint, nn.Module):
        return checkpoint.state_dict()

    if not isinstance(checkpoint, dict):
        raise TypeError("Unsupported checkpoint format.")

    for key in ("model_state_dict", "state_dict", "model"):
        if key in checkpoint:
            value = checkpoint[key]
            if isinstance(value, nn.Module):
                return value.state_dict()
            if isinstance(value, dict):
                return value

    return checkpoint

def load_model():
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"Checkpoint not found:\n{CHECKPOINT}")

    try:
        checkpoint = torch.load(
            CHECKPOINT, map_location=DEVICE, weights_only=True
        )
    except TypeError:
        checkpoint = torch.load(CHECKPOINT, map_location=DEVICE)
    except Exception:
        checkpoint = torch.load(
            CHECKPOINT, map_location=DEVICE, weights_only=False
        )

    state_dict = extract_state_dict(checkpoint)

    if any(key.startswith("module.") for key in state_dict):
        state_dict = {
            key.removeprefix("module."): value
            for key, value in state_dict.items()
        }

    model = CNN_Small().to(DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    return model

# ============================================================
def checkpoint_label(checkpoint_path):
    """Return 'best', 'last', or the checkpoint stem."""
    stem = checkpoint_path.stem.lower()
    if "best" in stem:
        return "best"
    if "last" in stem:
        return "last"
    return checkpoint_path.stem


# ============================================================
def maximize_activation(model, target_function, seed):
    generator = torch.Generator(device=DEVICE)
    generator.manual_seed(seed)

    # obraz początkowy losowy
    initial_image = torch.rand((1, 1, 28, 28), generator=generator, device=DEVICE,) * 0.2

    image = nn.Parameter(initial_image)
    optimizer = torch.optim.Adam([image], lr=LEARNING_RATE) # używam zwykłego adama do aktualizacji cyfr na obrazku

    for _ in range(STEPS):
        optimizer.zero_grad()

        activation = target_function(image)
        loss = -activation

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            image.clamp_(INPUT_MIN, INPUT_MAX) # trzymanie zakresu od 0 do 1 

    return image.detach().cpu().squeeze().numpy()

# Zapisy rysunki
# ============================================================
def save_class_grid(images, titles, filename, suptitle):
    fig, axes = plt.subplots(
        2,
        5,
        figsize=(15, 8),
        squeeze=False,
    )

    for index, ax in enumerate(axes.flat):
        if index < len(images):
            # gray_r: value 0 is white, value 1 is black.
            ax.imshow(
                images[index],
                cmap="gray_r",
                vmin=INPUT_MIN,
                vmax=INPUT_MAX,
            )
            ax.set_title(titles[index], fontsize=12, pad=10)

        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        suptitle,
        fontsize=15,
        fontweight="bold",
    )

    # Reserve space on the right for one shared colorbar.
    fig.subplots_adjust(
        left=0.04,
        right=0.89,
        bottom=0.08,
        top=0.88,
        wspace=0.28,
        hspace=0.50,
    )

    # Shared grayscale scale for all ten images:
    # 0 = white, 1 = black.
    colorbar_ax = fig.add_axes([0.92, 0.20, 0.018, 0.60])
    scalar_map = plt.cm.ScalarMappable(
        norm=Normalize(vmin=INPUT_MIN, vmax=INPUT_MAX),
        cmap="gray_r",
    )
    scalar_map.set_array([])

    colorbar = fig.colorbar(
        scalar_map,
        cax=colorbar_ax,
        ticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    )
    colorbar.set_label(
        "Pixel value",
        fontsize=10,
        labelpad=10,
    )
    colorbar.ax.tick_params(labelsize=9)

    output_path = OUTPUT_DIR / filename
    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
        pad_inches=0.2,
    )
    plt.close(fig)
    print(f"Saved: {output_path}")

# ============================================================
def generate_output_layer_images(model):
    images = []
    titles = []

    for class_index in range(10):
        def target(image, class_index=class_index):
            normalized_image = (image - 0.1307) / 0.3081
            logits = model(normalized_image)
            return logits[0, class_index]

        generated = maximize_activation(model,target,seed=300 + class_index,)
        images.append(generated)
        titles.append(f"Class {class_index}")

    network_name = "CNN_Small"
    ckpt_label = checkpoint_label(CHECKPOINT)
    figure_title = (
        f"{network_name} — output layer activation maximization ({ckpt_label})"
    )

    save_class_grid(
        images,
        titles,
        f"activation_maximization_{network_name.lower()}_{ckpt_label}.png",
        figure_title,
    )

# ============================================================
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Device: {DEVICE}")
    print(f"Loading model: {CHECKPOINT}")
    model = load_model()
    print("Generating output-layer visualizations...")
    generate_output_layer_images(model)
    print("Done.")

if __name__ == "__main__":
    main()
