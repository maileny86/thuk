"""Challenge 1 - Data generation and models benchmarking.
Author: Milena Napiorkowska.
"""
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from pathlib import Path
import numpy as np
import gc
import random
import cv2

# region constants
NUM_CLASSES = 8
NUM_SAMPLES = 1000
IMG_SIZE = 56
BATCH_SIZE = 32
EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
# endregion

class Arrow:
    def __init__(self, image_size: int) -> None:
        """Initialize arrow generator with image size."""
        self.image_size = image_size
    
    def generate(self, angle: float):
        """Generate an arrow image."""
        img = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        
        center, max_length = self._find_center(angle)
        length = self._determine_length(max_length)
        start, end = self._calculate_endpoints(center, angle, length)
        return self._draw(img, start, end)

    def _find_center(self, angle: float, min_len: int = 5) -> tuple:
        """Find random center that allows arrow to fit within image"""
        while True:
            center = (np.random.randint(self.image_size), 
                     np.random.randint(self.image_size))
            max_len = self._calculate_max_lenght(center, angle)
            if max_len >= min_len:
                return center, max_len

    def _calculate_max_lenght(self, center: tuple, angle: float) -> float:
        """Calculate maximum possible arrow length for given center and angle"""
        x, y = center
        rad = np.deg2rad(angle)
        cos_θ, sin_θ = np.cos(rad), np.sin(rad)
        max_x = min(x, self.image_size-1-x) / abs(cos_θ)
        max_y = min(y, self.image_size-1-y) / abs(sin_θ)
        return min(max_x, max_y)

    def _determine_length(self, max_length: float) -> int:
        """Generate reasonable arrow length within constraints"""
        base_length = self.image_size // np.random.randint(2, 5)
        return np.clip(base_length, 10, int(max_length))

    def _calculate_endpoints(self, center: tuple, 
                                 angle: float, length: int) -> tuple:
        """Calculate and validate arrow endpoints"""
        x, y = center
        rad = np.deg2rad(angle)
        dx = length * np.cos(rad)
        dy = length * np.sin(rad)
        
        start = self._clamp_coordinates((x - dx, y - dy))
        end = self._clamp_coordinates((x + dx, y + dy))
        return start, end

    def _clamp_coordinates(self, point: tuple) -> tuple:
        """Ensure coordinates stay within image boundaries"""
        x, y = point
        return (
            int(np.clip(round(x), 0, self.image_size-1)),
            int(np.clip(round(y), 0, self.image_size-1))
        )

    def _draw(self, img: np.ndarray, start: tuple, end: tuple):
        """Draw arrow on image with random visual properties"""
        color = tuple(np.random.randint(30, 255, 3).tolist())
        thickness = np.random.randint(1, 5)
        tip_length = np.random.uniform(0.1, 0.5)
        return cv2.arrowedLine(img, start, end, color, thickness, tipLength=tip_length)

class ArrowsDataset(Dataset):
    def __init__(self, num_samples: int, image_size: int, num_classes: int) -> None:
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_classes = num_classes
        self.angle_step = 360 // num_classes
        self.data, self.labels = self._create_dataset()
    
    def _create_dataset(self):
        data = []
        labels = []
        arrows = []
        for _ in range(self.num_samples):
            angle_idx = np.random.randint(self.num_classes)
            angle = angle_idx * self.angle_step + np.random.randint(-self.angle_step//2, self.angle_step//2)
            arrow = Arrow(self.image_size).generate(angle)
            arrows.append(arrow)
            img_tensor = torch.tensor(arrow, dtype=torch.float32).permute(2, 0, 1) / 255.0
            data.append(img_tensor)
            labels.append(angle_idx)
        return torch.stack(data), torch.LongTensor(labels)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class SimpleCNN(nn.Module):
    def __init__(self, num_classes, image_size=56, batch_norm=False, dropout_rate=0.0):
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

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

class ModelBenchmark:
    def __init__(self, model, name):
        self.model = model.to(DEVICE)
        self.name = name
        self.train_time = 0.0
        self.accuracy = 0.0
        self.loss_history = []
        self.optimizer = optim.Adam(model.parameters())
        self.criterion = nn.CrossEntropyLoss()

    def train(self, train_loader, test_loader):
        best_loss = float('inf')
        model_path = RESULTS_DIR/f"{self.name}.pth"
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
        
        self.train_time = time.time() - start_time
        self.accuracy = self.evaluate(test_loader)
    
    def _run_epoch(self, loader, training=True):
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

    def evaluate(self, test_loader):
        all_preds, all_labels = [], []
        self.model.eval()
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = self.model(inputs.to(DEVICE))
                all_preds.extend(torch.argmax(outputs, 1).cpu().numpy())
                all_labels.extend(labels.numpy())
        return accuracy_score(all_labels, all_preds)

def initialize_models():
    return {
        "SimpleCNN": SimpleCNN(NUM_CLASSES, IMG_SIZE),
        "SimpleCNN+BN": SimpleCNN(NUM_CLASSES, IMG_SIZE, batch_norm=True),
        "SimpleCNN+DO": SimpleCNN(NUM_CLASSES, IMG_SIZE, dropout_rate=0.3),
        "ResNet18": models.resnet18(num_classes=NUM_CLASSES),
        "EfficientNet-B0": models.efficientnet_b0(num_classes=NUM_CLASSES),
        "MobileNetV2": models.mobilenet_v2(num_classes=NUM_CLASSES),
    }

def plot_results(benchmarks):
    plt.figure(figsize=(18, 5))

    colors = plt.cm.Set1(np.linspace(0, 1, len(benchmarks)))
    
    # Training Time
    plt.subplot(131)
    plt.bar([b.name for b in benchmarks], [b.train_time for b in benchmarks], color=colors)
    plt.title("Training Time (s)")
    
    # Accuracy
    plt.subplot(132)
    plt.bar([b.name for b in benchmarks], [b.accuracy for b in benchmarks], color=colors)
    plt.ylim(0, 1)
    plt.title("Test Accuracy")
    
    # Loss History
    plt.subplot(133)
    for b, c in zip(benchmarks, colors):
        plt.plot(b.loss_history, label=b.name, color=c)
    plt.title("Validation Loss")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR/"results.png")
    plt.show()

def main():
    train_set = ArrowsDataset(NUM_SAMPLES, IMG_SIZE, NUM_CLASSES)
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
        print(f"{name} Accuracy: {benchmark.accuracy:.2%}")
    
    plot_results(benchmarks)

if __name__ == "__main__":
    main()