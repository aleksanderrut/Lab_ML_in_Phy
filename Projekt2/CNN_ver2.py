#!/usr/bin/env python3
"""Porownanie Perceptron, Deep, Deep+L2 i CNN z/bez dropout na MNIST."""

from pathlib import Path
import os
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


# ============================================================
# PARAMETRY EKSPERYMENTU - zmieniaj tylko tutaj
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "MNIST" / "raw"
OUTPUT_DIR = BASE_DIR / "cnn_results"

EPOCHS_BY_MODEL = {
    "Perceptron": 50,
    "Deep": 50,
    "Deep_L2": 50,
    "CNN_Small": 30,
    "CNN_Medium": 30,
    "CNN_Large": 30,
    "CNN_Large_Dropout": 30,
}

BATCH_SIZE = 128
LEARNING_RATE = 0.001
SEED = 123
NUM_WORKERS = 0

# Regularyzacja: 0.0 = wylaczona; przykladowe wartosci do eksperymentu.
DEEP_WEIGHT_DECAY = 0.001      # L2 dla Deep_L2 (optimizer Adam)
CNN_DROPOUT_P = 0.5            # 0.0 = dropout wylaczony (CNN_Large_Dropout)

# Kazdy wariant to osobny trening i osobne pliki wynikowe.
# Aby wykonac tylko porownania, zostaw np.
# ("Deep", "Deep_L2", "CNN_Large", "CNN_Large_Dropout").
MODELS_TO_TRAIN = (
    "Perceptron",
    "Deep",
    "Deep_L2",
    "CNN_Small",
    "CNN_Medium",
    "CNN_Large",
    "CNN_Large_Dropout",
)


# ============================================================
# MODELE
# ============================================================

class Perceptron(nn.Module):
    """Jedna warstwa w pelni polaczona: 784 -> 10."""

    def __init__(self):
        super(Perceptron, self).__init__()
        # fully connected layer
        self.fc = nn.Linear(28*28, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.fc(x)
        return F.log_softmax(x, dim=1) # note that dim=0 is the number of samples in batch


class Deep(nn.Module):
    """Dwie warstwy w pelni polaczone: 784 -> 20 -> 10."""

    # this defines the structure of the Perceptron model
    def __init__(self):
        super(Deep, self).__init__()
        # fully connected layers
        self.fc1 = nn.Linear(28*28, 20)
        self.fc2 = nn.Linear(20, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.fc1(x)
        x = F.relu(x)
        # hidden layer
        x = self.fc2(x)
        return F.log_softmax(x, dim=1) # note that dim=0 is the number of samples in batch


class CNN_Small(nn.Module):
    """Model z instrukcji: 2 i 4 filtry 5x5, bez dropout."""

    def __init__(self):
        super(CNN_Small, self).__init__()
        # convolutional layer with 2 kernels of size 5x5
        self.conv1 = nn.Conv2d(1, 2, kernel_size=5)
        # 4 kernels of size 5x5
        self.conv2 = nn.Conv2d(2, 4, kernel_size=5)
        # fully connected layers
        self.fc1 = nn.Linear(64, 20)
        self.fc2 = nn.Linear(20, 10)

    def forward(self, x):
        # 1st layers group
        x = self.conv1(x) # resulting in 2 feature maps each of size 24x24
        x = F.max_pool2d(x, 2) # downsizing each map to 12x12
        x = F.relu(x) # standard (in CNNs) ReLU activation
        # 2nd group
        x = self.conv2(x) # resulting in 4 feature maps each of size 8x8
        x = F.max_pool2d(x, 2) # downsizing each map to 4x4
        x = F.relu(x)
        # fully connected layers
        x = x.view(-1, 64) # 4 maps of 4x4 size gives 64 numbers
        x = self.fc1(x) # 64 -> 20
        x = F.relu(x)
        x = self.fc2(x) # 20 -> 10
        # softmax (multinomial classification) gives probabilities of each class
        return F.log_softmax(x, dim=1) # note that dim=0 is the number of samples in batch



class CNN_Medium(nn.Module):
    """Model pośredni: 8, 16, 32 filtrów 3x3."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))                    # 8 x 28 x 28
        x = F.relu(self.conv2(x))                    # 16 x 28 x 28
        x = F.max_pool2d(x, 2)                      # 16 x 14 x 14
        x = F.relu(self.conv3(x))                    # 32 x 14 x 14
        x = F.max_pool2d(x, 2)                      # 32 x 7 x 7
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return F.log_softmax(self.fc2(x), dim=1)


class CNN_Large(nn.Module):
    """Duzy CNN: 32, 64, 128 filtrow; dropout sterowany parametrem p."""

    def __init__(self, dropout_p=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 7 * 7, 128)
        self.dropout = nn.Dropout(p=dropout_p)  # w model.eval() automatycznie nie dziala
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return F.log_softmax(self.fc2(x), dim=1)


# ============================================================
# TRENING I OCENA
# ============================================================
def train_epoch(model, train_loader, optimizer, device):
    model.train()
    loss_sum = 0.0
    n_samples = 0

    for images, labels in train_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        prediction = model(images)
        loss = F.nll_loss(prediction, labels)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * labels.size(0)
        n_samples += labels.size(0)

    return loss_sum / n_samples


@torch.inference_mode()
def evaluate(model, loader, device, make_confusion=False):
    model.eval()
    loss_sum = 0.0
    n_correct = 0
    n_samples = 0
    cm = np.zeros((10, 10), dtype=np.int64) if make_confusion else None

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        output = model(images)
        loss_sum += F.nll_loss(output, labels, reduction="sum").item()
        predicted = output.argmax(dim=1)
        n_correct += (predicted == labels).sum().item()
        n_samples += labels.size(0)

        if make_confusion:
            np.add.at(cm, (labels.cpu().numpy(), predicted.cpu().numpy()), 1)

    return loss_sum / n_samples, 100.0 * n_correct / n_samples, cm


def save_confusion(cm, model_name, variant, epoch):
    """Zapis macierzy testowej dla best lub last; wiersz=prawda, kolumna=predykcja."""
    prefix = model_name.lower()
    np.savetxt(
        OUTPUT_DIR / f"{prefix}_confusion_{variant}_counts.data",
        cm,
        fmt="%d",
        header=f"epoch={epoch}; rows=true_digit(0..9), columns=predicted_digit(0..9)",
    )
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_percent = np.divide(
        cm * 100.0,
        row_sums,
        out=np.zeros_like(cm, dtype=float),
        where=row_sums != 0,
    )
    np.savetxt(
        OUTPUT_DIR / f"{prefix}_confusion_{variant}_percent.data",
        cm_percent,
        fmt="%.4f",
        header=f"epoch={epoch}; percent per true class (rows sum to 100); rows=true, columns=predicted",
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # Dostosowanie do #SBATCH -c 4; nie wykorzystujemy wiecej rdzeni.
    cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))
    torch.set_num_threads(max(1, cpus))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}; results: {OUTPUT_DIR}", flush=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    full_train = datasets.MNIST(str(DATA_DIR), train=True, download=True, transform=transform)
    test_set = datasets.MNIST(str(DATA_DIR), train=False, download=True, transform=transform)
    train_set, val_set = random_split(
        full_train, [50000, 10000], generator=torch.Generator().manual_seed(SEED)
    )

    loader_kwargs = {
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "pin_memory": device.type == "cuda",
    }
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_set, shuffle=False, **loader_kwargs)

    model_classes = {
        "Perceptron": Perceptron,
        "Deep": Deep,
        "Deep_L2": Deep,                        # ta sama architektura, inne L2
        "CNN_Small": CNN_Small,
        "CNN_Medium": CNN_Medium,
        "CNN_Large": CNN_Large,                 # dropout_p=0.0
        "CNN_Large_Dropout": lambda: CNN_Large(dropout_p=CNN_DROPOUT_P),
    }

    for name in MODELS_TO_TRAIN:
        epochs = EPOCHS_BY_MODEL[name]
        # Osobny seed i identyczna kolejnosc tasowania danych dla kazdego modelu.
        torch.manual_seed(SEED)
        model = model_classes[name]().to(device)
        weight_decay = DEEP_WEIGHT_DECAY if name == "Deep_L2" else 0.0
        dropout_p = CNN_DROPOUT_P if name == "CNN_Large_Dropout" else 0.0
        optimizer = torch.optim.Adam(
            model.parameters(), lr=LEARNING_RATE, weight_decay=weight_decay
        )
        train_loader = DataLoader(
            train_set, shuffle=True,
            generator=torch.Generator().manual_seed(SEED), **loader_kwargs
        )

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(
            f"\n{name}: {n_params} trainable parameters; "
            f"epochs={epochs}; weight_decay={weight_decay}; dropout_p={dropout_p}",
            flush=True,
        )
        history_path = OUTPUT_DIR / f"{name.lower()}_history.data"
        best_path = OUTPUT_DIR / f"{name.lower()}_best.pt"
        last_path = OUTPUT_DIR / f"{name.lower()}_last.pt"
        best_val_loss = float("inf")

        with open(history_path, "w", encoding="utf-8") as f:
            f.write("# epoch train_loss val_loss test_accuracy_percent\n")
            for epoch in range(1, epochs + 1):
                train_loss = train_epoch(model, train_loader, optimizer, device)
                val_loss, _, _ = evaluate(model, val_loader, device)
                _, test_accuracy, cm = evaluate(
                    model, test_loader, device, make_confusion=True
                )

                # Zapis po KAZDEJ epoce (wyniki nie gina po przerwaniu joba).
                f.write(
                    f"{epoch} {train_loss:.8f} {val_loss:.8f} "
                    f"{test_accuracy:.4f}\n"
                )
                f.flush()
                # Macierze "last" sa aktualizowane po kazdej ukonczonej epoce.
                save_confusion(cm, name, "last", epoch)

                # Wagi ostatniej epoki odpowiadaja zapisanej macierzy pomylek.
                checkpoint = {
                    "model_name": name,
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "weight_decay": weight_decay,
                    "dropout_p": dropout_p,
                    "model_state_dict": model.state_dict(),
                }
                torch.save(checkpoint, last_path)

                # Najlepszy model wybieramy WYŁĄCZNIE na podstawie validation loss.
                # Trening i tak trwa przez pelna liczbe epok dla tego modelu.
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(checkpoint, best_path)
                    # Macierze "best" sa z tej samej epoki co wagi "best.pt".
                    save_confusion(cm, name, "best", epoch)

                print(
                    f"{name} | epoch {epoch:>2}/{epochs} | "
                    f"train loss={train_loss:.5f} | "
                    f"val loss={val_loss:.5f} | "
                    f"test accuracy={test_accuracy:.2f}%",
                    flush=True,
                )

    print(f"\nZakonczono. Wyniki: {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
