"""Challenge 1 - Data generation and models benchmarking.
Author: Milena Napiorkowska.
"""
import os
import gc
import random
import time
from pathlib import Path
from typing import Self

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models

# region constants
NUM_CLASSES = 8
NUM_SAMPLES = 1000
IMG_SIZE = 56
BATCH_SIZE = 32
EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = Path(__file__).resolve().parent / "outputs"
RESULTS_DIR.mkdir(exist_ok=True)
# endregion


class Arrow:
    """Random arrow image generator."""
    def __init__(self: Self, image_size: int) -> None:
        """Initialize arrow generator with image size."""
        self.image_size = image_size

    def generate(self: Self, angle: int) -> np.ndarray:
        """Generate an arrow image with angle."""
        image = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        center, length = self._find_center(angle)
        start, end = self._calculate_endpoints(center, angle, length)
        return self._draw(image, start, end)

    def _find_center(self: Self, angle: int, min_length: int = 10) -> tuple[tuple[int, int], int]:
        """Find a valid center ensuring arrow fits in the image."""
        while True:
            center = tuple(int(v) for v in np.random.randint(0, self.image_size, size=2))
            max_length = self._max_length(center, angle)
            if max_length >= min_length:
                lenght = int(
                    np.clip(self.image_size // np.random.randint(2, 5), min_length, max_length)
                    )
                return center, lenght
    def _max_length(self: Self, center: tuple[int, ...], angle: int) -> int:
        """Calculate max arrow length that fits in the image."""
        x, y = center
        rad = np.deg2rad(angle)
        max_x = (self.image_size - 1 - abs(2 * x - self.image_size)) / abs(np.cos(rad))
        max_y = (self.image_size - 1 - abs(2 * y - self.image_size)) / abs(np.sin(rad))
        return int(min(max_x, max_y))

    def _calculate_endpoints(
            self: Self,
            center: tuple[int, ...],
            angle: int,
            length: int
            ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Calculate start and end points of the arrow."""
        rad = np.deg2rad(angle)
        delta = np.array([length * np.cos(rad), length * np.sin(rad)])
        start = tuple(map(int, (max(0, min(self.image_size - 1, c - d)) for c, d in zip(center, delta))))
        end = tuple(map(int, (max(0, min(self.image_size - 1, c + d)) for c, d in zip(center, delta))))
        return start, end

    def _draw(
            self: Self,
            img: np.ndarray,
            start: tuple[int, ...],
            end: tuple[int, ...]
            ) -> np.ndarray:
        """Draw an arrow with random properties."""
        color = tuple(map(int, np.random.randint(10, 255, size=3)))
        thickness = np.random.randint(1, 5)
        tip_length = np.random.uniform(0.1, 0.5)
        return cv2.arrowedLine(img, start, end, color, thickness, tipLength=tip_length)


class ArrowsDataset(Dataset):
    """Arrows dataset generator."""
    def __init__(
            self: Self,
            num_samples: int,
            image_size: int,
            num_classes: int,
            visualize : bool = False
            ) -> None:
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_classes = num_classes
        self.angle_step = 360 // num_classes
        self.data, self.labels = self._create_dataset(visualize)

    def _create_dataset(self: Self, visualize: bool):
        angles = (np.random.randint(-self.angle_step // 2, self.angle_step // 2, self.num_samples) + \
                 np.arange(self.num_samples) % self.num_classes * self.angle_step).tolist()

        arrows = [Arrow(self.image_size).generate(int(angle)) for angle in angles]

        if visualize:
            plot_arrows(arrows, num_cols=10, num_rows=5)

        data = torch.stack([torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0 for img in arrows])
        labels = torch.LongTensor(np.arange(self.num_samples) % self.num_classes)

        return data, labels

    def __len__(self: Self) -> int:
        return self.num_samples

    def __getitem__(self: Self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.labels[idx]


class SimpleCNN(nn.Module):
    """Simple CNN model."""
    def __init__(self: Self, num_classes, image_size=56, batch_norm=False, dropout_rate=0.0):
        super().__init__()
        layers = [
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
        ]
        if batch_norm:
            layers.append(nn.BatchNorm2d(64))
        layers += [nn.ReLU(), nn.MaxPool2d(2)]

        self.features = nn.Sequential(*layers)

        with torch.no_grad():
            dummy = torch.zeros(1, 3, image_size, image_size)
            self.feature_size = self.features(dummy).view(1, -1).size(1)

        self.classifier = nn.Sequential(
            nn.Linear(self.feature_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self: Self, x):
        """Forward pass of the model."""
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

class ModelBenchmark:
    """Model benchmarking class."""
    def __init__(self: Self, model: nn.Module, name: str) -> None:
        self.model = model.to(DEVICE)
        self.name = name
        self.train_time = 0.0
        self.best_accuracy = 0.0
        self.loss_history: list[float] = []
        self.optimizer = optim.Adam(model.parameters())
        self.criterion = nn.CrossEntropyLoss()

    def train(self: Self, train_loader: DataLoader, test_loader: DataLoader) -> None:
        """Train model on training set and validate on test set."""
        best_loss = float('inf')
        model_path = RESULTS_DIR/f"{self.name}_{NUM_CLASSES}.pth"
        start_time = time.time()

        for epoch in range(EPOCHS):
            train_loss = self._run_epoch(train_loader, training=True)
            val_loss = self._run_epoch(test_loader, training=False)
            self.loss_history.append(val_loss)

            print(f"Epoch {epoch+1}/{EPOCHS} | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(self.model.state_dict(), model_path)
                print(f"Saved {self.name} to {model_path}")
                self.best_accuracy = self.evaluate(test_loader)

        self.train_time = time.time() - start_time


    def _run_epoch(self: Self, loader: DataLoader, training: bool = True) -> float:
        self.model.train(training)
        total_loss = 0.0

        with torch.set_grad_enabled(training):
            for inputs, labels in loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                if training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                total_loss += loss.item() * inputs.size(0)

        return total_loss / len(loader.dataset)

    def evaluate(self: Self, test_loader: DataLoader) -> float:
        """Evaluate model on test set."""
        all_preds: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []
        self.model.eval()
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = self.model(inputs.to(DEVICE))
                all_preds.extend(torch.argmax(outputs, 1).cpu().numpy())
                all_labels.extend(labels.numpy())
        return accuracy_score(all_labels, all_preds)

def initialize_models() -> dict[str, nn.Module]:
    """Initialize models for benchmarking."""
    return {
        "SimpleCNN": SimpleCNN(NUM_CLASSES, IMG_SIZE),
        "SimpleCNN+BN": SimpleCNN(NUM_CLASSES, IMG_SIZE, batch_norm=True),
        "ResNet18": models.resnet18(num_classes=NUM_CLASSES),
        "EfficientNet-B0": models.efficientnet_b0(num_classes=NUM_CLASSES),
        "MobileNetV2": models.mobilenet_v2(num_classes=NUM_CLASSES),
    }

def plot_results(benchmarks) -> None:
    plt.figure(figsize=(18, 5))

    colors = plt.cm.Set1(np.linspace(0, 1, len(benchmarks)))

    # Training Time
    plt.subplot(131)
    plt.bar([b.name for b in benchmarks], [b.train_time for b in benchmarks], color=colors)
    plt.title("Training Time (s)")

    # Accuracy
    plt.subplot(132)
    plt.bar([b.name for b in benchmarks], [b.best_accuracy for b in benchmarks], color=colors)
    plt.ylim(0, 1)
    plt.title("Test Accuracy")

    # Loss History
    plt.subplot(133)
    for b, c in zip(benchmarks, colors):
        plt.plot(b.loss_history, label=b.name, color=c)
    plt.title("Validation Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(RESULTS_DIR/"model_benchmarking_results.png")
    plt.show()

def plot_arrows(imgs: list[np.ndarray], num_rows: int = 1, num_cols: int = 5) -> None:
    """
    Plot arrows in a grid layout with specified rows and columns
    """
    imgs = random.choices(imgs, k=num_cols*num_rows)
    _, axes = plt.subplots(num_rows, num_cols,
                            figsize=(3*num_cols, 3*num_rows))

    # Handle single row case
    if num_rows == 1:
        axes = axes.reshape(1, -1)

    # Plot images
    for idx in range(len(imgs)):
        img = imgs[idx]
        row = idx // num_cols
        col = idx % num_cols
        axes[row, col].imshow(img)
        axes[row, col].axis("off")

    # Turn off empty subplots
    for idx in range(len(imgs), num_rows*num_cols):
        row = idx // num_cols
        col = idx % num_cols
        axes[row, col].axis("off")

    plt.tight_layout()
    plt.show()

def main():
    """Main function to generate dataset and train models."""
    train_set = ArrowsDataset(NUM_SAMPLES, IMG_SIZE, NUM_CLASSES, visualize=True)
    test_set = ArrowsDataset(NUM_SAMPLES//5, IMG_SIZE, NUM_CLASSES)

    benchmarks = []
    for name, model in initialize_models().items():
        gc.collect()
        torch.cuda.empty_cache()
        print(f"\nTraining {name}")
        benchmark = ModelBenchmark(model, name)
        benchmark.train(
            DataLoader(train_set, BATCH_SIZE, shuffle=True),
            DataLoader(test_set, BATCH_SIZE)
        )
        benchmarks.append(benchmark)
        print(f"{name} Accuracy: {benchmark.best_accuracy:.2%}")
    plot_results(benchmarks)

if __name__ == "__main__":
    main()

