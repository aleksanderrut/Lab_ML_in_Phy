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
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================
# MODEL
class CNN_small(nn.Module):

    def __init__(self, num_classes, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4)) # pooluje średnią aż bedzie rozmiar 4na4
        self.fc1 = nn.Linear(64 * 4 * 4, 64)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x))) # pytorch zaogragla w doł wieć 62 na 62
        x = self.adaptive_pool(x)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.softmax(x, dim=1)


class CNN(nn.Module):

    def __init__(self, num_classes, dropout=0.3):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        self.fc1 = nn.Linear(128 * 7 * 7, 128)
        self.dropout = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# ============================================================
# DATASET
IMAGE_EXTENSIONS = [".png"]

def find_image(data_dir, stem):

    for extension in IMAGE_EXTENSIONS:

        path = data_dir / f"{stem}{extension}"

        if path.exists():
            return path


def load_labels(data_dir):

    labels_path = data_dir / "labels.json"

    with open(labels_path, "r") as f:
        raw_labels = json.load(f)

    file_ids = []
    label_names = []

    for item in raw_labels:

        file_ids.append(str(item["file_id"]))
        label_names.append(item["label"])

    classes = sorted(set(label_names))

    class_to_idx = {
        name: idx
        for idx, name in enumerate(classes)
    }

    labels = {
        file_id: class_to_idx[label]
        for file_id, label in zip(file_ids, label_names)
    }

    print(f"Number of images: {len(file_ids)}")
    print(f"Number of classes: {len(classes)}")
    print(f"Classes: {classes}")

    return file_ids, labels, class_to_idx


class CrystalDataset(Dataset):

    def __init__(
        self,
        data_dir,
        file_ids,
        labels,
        image_size,
        fourier=False
    ):

        self.data_dir = data_dir
        self.file_ids = file_ids
        self.labels = labels
        self.image_size = image_size
        self.fourier = fourier

        self.image_paths = []

        for file_id in file_ids:

            if fourier:
                stem = f"{file_id}_f"
            else:
                stem = file_id

            self.image_paths.append(
                find_image(data_dir, stem)
            )

    def __len__(self):
        return len(self.file_ids)

    #-------------------------------------
    #metoda z resamplingiem LANCZOS
    def __getitem__(self, idx):

        path = self.image_paths[idx]
        file_id = self.file_ids[idx]

        image = Image.open(path).convert("L")

        image = image.resize(
            (self.image_size, self.image_size),
            Image.Resampling.LANCZOS
        )

        image = np.asarray(image, dtype=np.float32) / 255.0

        # Normalize pixel values from [0, 1] to [-1, 1].
        image = (image - 0.5) / 0.5

        image = torch.from_numpy(image).unsqueeze(0)

        label = self.labels[file_id]

        return image, label

    #-------------------------------------
    #metoda z wycinkiem centralnego regionu 500x500
    # def __getitem__(self, idx):
    #     path = self.image_paths[idx]
    #     file_id = self.file_ids[idx]

    #     image = Image.open(path).convert("L")

    #     # Crop the central 500 x 500 region.
    #     width, height = image.size
    #     crop_size = 500

    #     left = (width - crop_size) // 2
    #     top = (height - crop_size) // 2

    #     image = image.crop((
    #         left,
    #         top,
    #         left + crop_size,
    #         top + crop_size
    #     ))

    #     # Convert image to a NumPy array.
    #     image = np.asarray(image, dtype=np.float32) / 255.0

    #     # Normalize pixel values to [-1, 1].
    #     image = (image - 0.5) / 0.5

    #     # Convert to a PyTorch tensor.
    #     image = torch.from_numpy(image).unsqueeze(0)

    #     label = self.labels[file_id]

    #     return image, label


# ============================================================
# 3. STRATIFIED DATA SPLIT
# ============================================================

def stratified_split(file_ids, labels, seed):

    rng = np.random.default_rng(seed)

    class_to_ids = {}

    for file_id in file_ids:

        label = labels[file_id]

        class_to_ids.setdefault(label, []).append(file_id)

    train_ids = []
    val_ids = []
    test_ids = []

    for label, ids in class_to_ids.items():

        ids = np.array(ids, dtype=object)

        rng.shuffle(ids)

        n = len(ids)

        n_test = max(1, round(0.1 * n))
        n_val = max(1, round(0.1 * n))

        test_ids.extend(ids[:n_test].tolist())

        val_ids.extend(ids[n_test:n_test + n_val].tolist())

        train_ids.extend(ids[n_test + n_val:].tolist())

    rng.shuffle(train_ids)
    rng.shuffle(val_ids)
    rng.shuffle(test_ids)

    print("\nDataset split:")
    print(f"Train: {len(train_ids)}")
    print(f"Validation: {len(val_ids)}")
    print(f"Test: {len(test_ids)}")

    return train_ids, val_ids, test_ids


def save_split(train_ids, val_ids, test_ids, output_dir):

    split = {
        "train": train_ids,
        "validation": val_ids,
        "test": test_ids
    }

    with open(output_dir / "data_split.json", "w") as f:
        json.dump(split, f, indent=2)


# ============================================================
# 4. DATA LOADERS
# ============================================================

def make_loaders(
    data_dir,
    train_ids,
    val_ids,
    test_ids,
    labels,
    image_size,
    batch_size,
    num_workers,
    device,
    fourier
):

    train_dataset = CrystalDataset(
        data_dir, train_ids, labels,
        image_size, fourier
    )

    val_dataset = CrystalDataset(
        data_dir, val_ids, labels,
        image_size, fourier
    )

    test_dataset = CrystalDataset(
        data_dir, test_ids, labels,
        image_size, fourier
    )

    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda"
    }

    if num_workers > 0:
        common["persistent_workers"] = True

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **common
    )

    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **common
    )

    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        **common
    )

    return train_loader, val_loader, test_loader


# ============================================================
# 5. EVALUATION
# ============================================================

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

            loss = F.nll_loss(
                output,
                target,
                reduction="sum"
            )

            total_loss += loss.item()

            total_correct += (
                output.argmax(dim=1) == target
            ).sum().item()

            total_samples += target.size(0)

    return (
        total_loss / total_samples,
        total_correct / total_samples
    )


# ============================================================
# 6. TRAINING
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    epoch,
    experiment_name
):

    model.train()

    total_loss = 0.0
    total_samples = 0
<<<<<<< HEAD
    for batch_idx, (data, target) in enumerate(loader, start=1):
=======

    for batch_idx, (data, target) in enumerate(
        loader, start=1
    ):
>>>>>>> 21638592919cc95f7aac91e90470055de3751a8a

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
                f"{experiment_name} | "
                f"epoch {epoch} | "
                f"batch {batch_idx}/{len(loader)} | "
                f"loss={loss.item():.5f}",
                flush=True
            )
    return total_loss / total_samples


def save_history_csv(history, path):

    keys = list(history.keys())

    with open(path, "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow(keys)

        writer.writerows(
            zip(*(history[key] for key in keys))
        )


def train_model(
    model,
    experiment_name,
    optimizer,
    train_loader,
    val_loader,
    device,
    epochs,
    output_dir,
    patience,
    scheduler,
    class_to_idx
):

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "learning_rate": []
    }

    best_val_loss = float("inf")
    best_epoch = 0

    best_state = copy.deepcopy(model.state_dict())

    epochs_without_improvement = 0

    checkpoint_path = (
        output_dir / "best_model.pt"
    )

    start_time = time.time()

    for epoch in range(1, epochs + 1):

        epoch_start = time.time()

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            epoch,
            experiment_name
        )

        val_loss, val_accuracy = evaluate(
            model,
            val_loader,
            device
        )

        current_lr = optimizer.param_groups[0]["lr"]

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)
        history["learning_rate"].append(current_lr)

        print(
            f"\n{experiment_name} | "
            f"epoch {epoch}/{epochs} | "
            f"train loss={train_loss:.5f} | "
            f"val loss={val_loss:.5f} | "
            f"val accuracy={100 * val_accuracy:.3f}% | "
            f"lr={current_lr:.2e} | "
            f"time={time.time() - epoch_start:.1f}s",
            flush=True
        )

        scheduler.step(val_loss)

        if val_loss < best_val_loss:

            best_val_loss = val_loss
            best_epoch = epoch

            best_state = copy.deepcopy(model.state_dict())

            epochs_without_improvement = 0

            torch.save(
                {
                    "experiment": experiment_name,
                    "epoch": best_epoch,
                    "best_val_loss": best_val_loss,
                    "model_state_dict": best_state,
                    "class_to_idx": class_to_idx
                },
                checkpoint_path
            )

        else:

            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:

            print(
                f"{experiment_name}: early stopping "
                f"after epoch {epoch}. "
                f"Best epoch: {best_epoch}",
                flush=True
            )

            break

    model.load_state_dict(best_state)

    elapsed = time.time() - start_time

    save_history_csv(
        history,
        output_dir / "history.csv"
    )

    print(
        f"\n{experiment_name} finished."
        f"\nBest epoch: {best_epoch}"
        f"\nBest validation loss: {best_val_loss:.5f}"
        f"\nTraining time: {elapsed / 60:.2f} min",
        flush=True
    )

    return history, best_epoch, elapsed, checkpoint_path


# ============================================================
# 7. CONFUSION MATRIX
# ============================================================

def get_predictions(model, loader, device):

    model.eval()

    all_targets = []
    all_predictions = []

    with torch.inference_mode():

        for data, target in loader:

            data = data.to(device, non_blocking=True)

            output = model(data)

            predictions = output.argmax(dim=1)

            all_predictions.append(
                predictions.cpu().numpy()
            )

            all_targets.append(
                target.numpy()
            )

    return (
        np.concatenate(all_targets),
        np.concatenate(all_predictions)
    )


def save_confusion_matrix(
    model,
    loader,
    device,
    num_classes,
    class_names,
    output_dir
):

    targets, predictions = get_predictions(
        model, loader, device
    )

    cm = np.zeros(
        (num_classes, num_classes),
        dtype=np.int64
    )

    np.add.at(cm, (targets, predictions), 1)

    np.savetxt(
        output_dir / "confusion_matrix.csv",
        cm,
        delimiter=",",
        fmt="%d"
    )

    with open(output_dir / "class_names.json", "w") as f:
        json.dump(class_names, f, indent=2)

    print("\nMost common confusions:")

    off_diag = cm.copy()
    np.fill_diagonal(off_diag, 0)

    indices = np.argsort(
        off_diag.ravel()
    )[::-1]

    printed = 0

    for index in indices:

        true_class, predicted_class = np.unravel_index(
            index, off_diag.shape
        )

        count = off_diag[true_class, predicted_class]

        if count == 0 or printed >= 10:
            break

        print(
            f"{class_names[true_class]} -> "
            f"{class_names[predicted_class]}: {count}"
        )

        printed += 1

    return cm

# ============================================================
# 8. SINGLE EXPERIMENT
# ============================================================

def run_experiment(
    experiment_name,
    fourier,
    args,
    data_dir,
    train_ids,
    val_ids,
    test_ids,
    labels,
    class_to_idx,
    device
):

    print("\n" + "=" * 72)
    print(f"EXPERIMENT: {experiment_name}")
    print("=" * 72)

    output_dir = args.output_dir / experiment_name

    output_dir.mkdir(parents=True, exist_ok=True)

    # Same seed for both experiments.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_loader, val_loader, test_loader = make_loaders(
        data_dir,
        train_ids,
        val_ids,
        test_ids,
        labels,
        args.image_size,
        args.batch_size,
        args.num_workers,
        device,
        fourier
    )

    num_classes = len(class_to_idx)

    model = CNN(
        num_classes=num_classes,
        dropout=args.dropout
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.0
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=6
    )

    history, best_epoch, elapsed, checkpoint = train_model(
        model,
        experiment_name,
        optimizer,
        train_loader,
        val_loader,
        device,
        args.epochs,
        output_dir,
        args.patience,
        scheduler,
        class_to_idx
    )

    # Evaluate the best validation checkpoint.
    test_loss, test_accuracy = evaluate(
        model,
        test_loader,
        device
    )

    cm = save_confusion_matrix(
        model,
        test_loader,
        device,
        num_classes,
        list(class_to_idx.keys()),
        output_dir
    )

    results = {
        "experiment": experiment_name,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "best_epoch": best_epoch,
        "training_seconds": elapsed,
        "num_classes": num_classes,
        "num_train": len(train_ids),
        "num_validation": len(val_ids),
        "num_test": len(test_ids),
        "checkpoint": checkpoint.name
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nTEST RESULTS:")
    print(f"Test loss: {test_loss:.5f}")
    print(f"Test accuracy: {100 * test_accuracy:.3f}%")

    return results


# ============================================================
# 9. ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="CNN classification of crystal net types."
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("./crystal_data")
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./crystal_results")
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=128
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=4
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=min(20, os.cpu_count() or 1)
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=123
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=100
    )

    return parser.parse_args()


# ============================================================
# 10. MAIN
# ============================================================

def main():

    args = parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.set_num_threads(args.threads)

    if hasattr(torch, "set_num_interop_threads"):
        try:
            torch.set_num_interop_threads(
                max(1, min(4, args.threads))
            )
        except RuntimeError:
            pass

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 72)
    print("CRYSTAL NET CLASSIFICATION")
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Image size: {args.image_size}x{args.image_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Dropout: {args.dropout}")
    print("=" * 72)

    data_dir = args.data_dir

    file_ids, labels, class_to_idx = load_labels(data_dir)

    # Check that all images exist before starting training.
    print("\nChecking image files...")

    for file_id in file_ids:

        find_image(data_dir, file_id)
        find_image(data_dir, f"{file_id}_f")

    print("All images found.")

    # One common stratified split for both experiments.
    train_ids, val_ids, test_ids = stratified_split(
        file_ids,
        labels,
        args.seed
    )

    save_split(
        train_ids,
        val_ids,
        test_ids,
        args.output_dir
    )

    with open(args.output_dir / "class_mapping.json", "w") as f:
        json.dump(class_to_idx, f, indent=2)

    configuration = vars(args).copy()
    configuration["device"] = str(device)
    configuration["class_to_idx"] = class_to_idx

    with open(args.output_dir / "configuration.json", "w") as f:
        json.dump(
            configuration,
            f,
            indent=2,
            default=str
        )

    results = {}

    # Experiment 1: original images.
    results["direct"] = run_experiment(
        experiment_name="direct",
        fourier=False,
        args=args,
        data_dir=data_dir,
        train_ids=train_ids,
        val_ids=val_ids,
        test_ids=test_ids,
        labels=labels,
        class_to_idx=class_to_idx,
        device=device
    )

    # Experiment 2: Fourier images.
    results["fourier"] = run_experiment(
        experiment_name="fourier",
        fourier=True,
        args=args,
        data_dir=data_dir,
        train_ids=train_ids,
        val_ids=val_ids,
        test_ids=test_ids,
        labels=labels,
        class_to_idx=class_to_idx,
        device=device
    )

    # Combined summary.
    with open(args.output_dir / "comparison.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(args.output_dir / "comparison.csv", "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            "experiment",
            "test_loss",
            "test_accuracy",
            "best_epoch",
            "training_seconds"
        ])

        for name, values in results.items():

            writer.writerow([
                name,
                values["test_loss"],
                values["test_accuracy"],
                values["best_epoch"],
                values["training_seconds"]
            ])

    print("\n" + "=" * 72)
    print("FINAL COMPARISON")
    print("=" * 72)

    for name, values in results.items():

        print(
            f"{name:10s} | "
            f"accuracy={100 * values['test_accuracy']:.3f}% | "
            f"loss={values['test_loss']:.5f} | "
            f"best epoch={values['best_epoch']}"
        )

    print("\nAll results saved in:")
    print(args.output_dir.resolve())


if __name__ == "__main__":
    main()
