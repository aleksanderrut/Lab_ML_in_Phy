#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CNN + macierz pomyłek MNIST — skrypt na serwer (bez Jupytera).

Model CNN, train(), test() oraz parametry domyślne pochodzą z CNNs(1).ipynb.
Skrypt: pobiera MNIST, trenuje oryginalną CNN, testuje i zapisuje wyniki.

Przykłady:
    python cnn_confusion_matrix.py
    python cnn_confusion_matrix.py --epochs 10 --output-dir wyniki_test
    python cnn_confusion_matrix.py --load-model wyniki_cnn/cnn_model.pt

Zależności: torch, torchvision, numpy, matplotlib.
"""

import argparse
from pathlib import Path
import random

import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import matplotlib
matplotlib.use("Agg")  # zapis wykresów bez ekranu / serwera X
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 14})

# Te same hiperparametry co w oryginalnym notebooku.
model_args = {
    "seed": 123,
    "batch_size": 128,
    "lr": 0.05,
    "momentum": 0.5,
    "epochs": 50,
    "log_interval": 100,
}

# ---------------------------------------------------------------------------
# Model CNN — przeniesiony bez zmiany architektury i forward() z notebooka.
# ---------------------------------------------------------------------------
class CNN(nn.Module):
    # this defines the structure of the CNN model
    def __init__(self):
        super(CNN, self).__init__()
        # convolutional layer with 2 kernels of size 5x5
        self.conv1 = nn.Conv2d(1, 2, kernel_size=5)
        # 4 kernels of size 5x5
        self.conv2 = nn.Conv2d(2, 4, kernel_size=5)
        # 2D dropout
        self.conv2_drop = nn.Dropout2d()
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
        # x = self.conv2_drop(x)
        x = F.max_pool2d(x, 2) # downsizing each map to 4x4
        x = F.relu(x)
        # fully connected layers
        x = x.view(-1, 64) # 4 maps of 4x4 size gives 64 numbers
        x = self.fc1(x) # 64 -> 20
        x = F.relu(x)
        x = F.dropout(x, training=self.training) # dropout is a type of regularization
        x = self.fc2(x) # 20 -> 10
        # softmax (multinomial classification) gives probabilities of each class
        return F.log_softmax(x, dim=1) # note that dim=0 is the number of samples in batch

# ---------------------------------------------------------------------------
# Funkcje train() i test() — przeniesione z oryginalnego notebooka.
# ---------------------------------------------------------------------------
def train(model, device, train_loader, optimizer, epoch_number):
    model.train()
    train_loss = 0.
    # get subsequent batches over the data in a given epoch
    for batch_idx, (data, target) in enumerate(train_loader):
        # send data tensors to GPU (or CPU)
        data, target = data.to(device), target.to(device)
        # this will zero out the gradients for this batch
        optimizer.zero_grad()
        # this will execute the forward() function
        output = model(data)
        # calculate the negative-log-likelihood loss
        loss = F.nll_loss(output, target, reduction='mean')
        # backpropagate the loss
        loss.backward()
        # update the model weights (with assumed learning rate)
        optimizer.step()
        if batch_idx % model_args['log_interval'] == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch_number, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
        train_loss += loss.item()
    train_loss /= len(train_loader)
    print('\nTrain set: Average loss: {:.4f}'.format(train_loss))
    return train_loss

def test(model, device, test_loader, message):
    model.eval()
    test_loss = 0.
    correct = 0
    # this is just inference, we don't need to calculate gradients
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # calculate and sum up batch loss
            test_loss += F.nll_loss(output, target, reduction='mean')
            # get the index of class with the max log-probability
            prediction = output.argmax(dim=1)
            # item() returns value of the given tensor
            correct += prediction.eq(target).sum().item()
    test_loss /= len(test_loader)
    print('{}: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        message, test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_loss.cpu()

# ============================================================
# CONFUSION MATRIX - OBLICZENIA
# ============================================================
def calculate_confusion_matrix(model, device, test_loader):
    model.eval()
    matrix = torch.zeros((10, 10), dtype=torch.int64)

    with torch.no_grad():
        for data, target in test_loader:
            prediction = model(data.to(device)).argmax(dim=1).cpu()
            pairs = target.cpu() * 10 + prediction
            matrix += torch.bincount(pairs, minlength=100).reshape(10, 10)

    return matrix.numpy()


# ============================================================
# CONFUSION MATRIX - ZAPIS DANYCH
# ============================================================
def save_confusion_matrix(matrix, output_dir):
    counts = matrix.astype(np.int64)
    row_totals = counts.sum(axis=1, keepdims=True)
    percentages = np.divide(counts * 100.0, row_totals,
                            out=np.zeros_like(counts, dtype=float),
                            where=row_totals != 0)

    # Wiersze: prawdziwa cyfra, kolumny: przewidziana cyfra
    np.savetxt(output_dir / "confusion_matrix_counts.txt", counts, fmt="%d",
               header="rows=true_digit_0..9 columns=predicted_digit_0..9")
    np.savetxt(output_dir / "confusion_matrix_percent.txt", percentages, fmt="%.6f",
               header="row_percentages rows=true_digit_0..9 columns=predicted_digit_0..9")

    accuracy = 100.0 * np.trace(counts) / counts.sum()
    print(f"Confusion matrix accuracy: {accuracy:.4f}%")


# ============================================================
# MAIN
# ============================================================
def main():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="CNN MNIST - confusion matrix")
    parser.add_argument("--epochs", type=int, default=model_args["epochs"])
    parser.add_argument("--batch-size", type=int, default=model_args["batch_size"])
    parser.add_argument("--lr", type=float, default=model_args["lr"])
    parser.add_argument("--momentum", type=float, default=model_args["momentum"])
    parser.add_argument("--seed", type=int, default=model_args["seed"])
    parser.add_argument("--log-interval", type=int, default=model_args["log_interval"])
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--data-dir", type=Path, default=script_dir.parent / "data")
    parser.add_argument("--output-dir", type=Path, default=script_dir / "wyniki_cnn")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs musi być >= 1")

    model_args.update({
        "seed": args.seed, "batch_size": args.batch_size, "lr": args.lr,
        "momentum": args.momentum, "epochs": args.epochs,
        "log_interval": args.log_interval
    })

    # Folder wynikowy i urządzenie
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Device: {device}, Output: {output_dir}")

    # Dane MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    mnist_train = datasets.MNIST(str(args.data_dir), train=True,
                                 download=True, transform=transform)
    train_subset, validation_subset = torch.utils.data.random_split(
        mnist_train, [50000, 10000],
        generator=torch.Generator().manual_seed(args.seed))
    test_subset = datasets.MNIST(str(args.data_dir), train=False,
                                 download=True, transform=transform)

    # DataLoadery
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda"
    }
    train_loader = torch.utils.data.DataLoader(train_subset, shuffle=True, **loader_kwargs)
    validation_loader = torch.utils.data.DataLoader(validation_subset, shuffle=False, **loader_kwargs)
    test_loader = torch.utils.data.DataLoader(test_subset, shuffle=False, **loader_kwargs)
    print(f"MNIST: train={len(train_subset)}, validation={len(validation_subset)}, test={len(test_subset)}")

    # Zapis konfiguracji eksperymentu
    config = {
        "seed": args.seed, "epochs": args.epochs,
        "batch_size": args.batch_size, "learning_rate": args.lr,
        "momentum": args.momentum, "log_interval": args.log_interval,
        "num_workers": args.num_workers, "device": str(device),
        "train_samples": len(train_subset),
        "validation_samples": len(validation_subset),
        "test_samples": len(test_subset),
        "normalization_mean": 0.1307, "normalization_std": 0.3081
    }
    (output_dir / "experiment_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Model CNN i optymalizator
    model = CNN().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    train_loss, validation_loss = [], []

    # Trening
    for epoch_number in range(1, args.epochs + 1):
        train_loss.append(train(model, device, train_loader, optimizer, epoch_number))
        validation_loss.append(float(test(model, device, validation_loader, "Validation set")))

    # Zapis historii strat: epoka, train loss, validation loss
    np.savetxt(output_dir / "loss_history.txt",
               np.column_stack((np.arange(1, args.epochs + 1), train_loss, validation_loss)),
               fmt=["%d", "%.8f", "%.8f"],
               header="epoch train_loss validation_loss")

    # Test i zapis macierzy pomyłek
    test(model, device, test_loader, "Test set")
    matrix = calculate_confusion_matrix(model, device, test_loader)
    save_confusion_matrix(matrix, output_dir)

    print(f"\nWyniki zapisano w: {output_dir}")


if __name__ == "__main__":
    main()
