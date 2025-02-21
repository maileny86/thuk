"""Challenge 1 - Data generation and models benchmarking.
Author: Milena Napiorkowska.
"""
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from pathlib import Path
import psutil
import numpy as np
import gc
import pandas as pd
import torch.nn.functional as F
import cv2

# region constants
NUM_CLASSES: int = 8
NUM_SAMPLES: int = 1000
IMG_SIZE: int = 56
BATCH_SIZE: int = 32
EPOCHS: int = 10
DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR: Path = Path("results")
TRANSFORM: transforms.Compose = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
# endregion
    
class ArrowDataset(Dataset):
    def __init__(self, num_samples, image_size, num_classes, transform=None):
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_classes = num_classes
        self.transform = transform
        self.angle_step = 360 // num_classes
        self.data, self.labels = self._generate_dataset()
    
    def _generate_arrow(self, angle):
        img = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        center = (self.image_size // 2, self.image_size // 2)
        lenght = self.image_size // np.random.randint(1, 5)
        rad = np.deg2rad(angle)
        start = (center[0] - int(lenght * np.cos(rad)), center[1] - int(lenght * np.sin(rad)))
        end = (center[0] + int(lenght * np.cos(rad)), center[1] + int(lenght * np.sin(rad)))
        arrow_color = tuple(np.random.randint(30, 255, 3).tolist())
        tip_length = np.random.uniform(0.1, 0.5)
        arrow_thickness = np.random.randint(1, 5)
        cv2.arrowedLine(img, start, end, arrow_color, arrow_thickness, tipLength=tip_length)
        return torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
    
    def _generate_dataset(self):
        data = torch.zeros((self.num_samples, 3, self.image_size, self.image_size))
        labels = torch.zeros(self.num_samples, dtype=torch.long)
        for i in range(self.num_samples):
            angle_index = np.random.randint(0, self.num_classes)
            angle = (angle_index * self.angle_step) + np.random.randint(-(self.angle_step//2), self.angle_step//2)
            data[i] = self._generate_arrow(angle)
            labels[i] = angle_index
        return data, labels
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, index):
        image, label = self.data[index], self.labels[index]
        if self.transform:
            image = self.transform(image)   
        return image, label


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10, batch_norm=False):
        super(SimpleCNN, self).__init__()
        self.batch_norm = batch_norm
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64) if batch_norm else nn.Identity()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  
        x = self.pool(F.relu(self.conv2(x)))  
        x = self.pool(F.relu(self.conv3(x)))  
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Model Benchmarking Class
class ModelBenchmark:
    def __init__(self, model, name):
        self.model = model.to(DEVICE)
        self.name = name
        self.train_time = 0
        self.accuracy = 0
        self.loss_history = []
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()

    def train(self, train_loader, test_loader):
        self.model.train()
        start_time = time.time()
        
        for epoch in range(EPOCHS):
            epoch_loss = 0
            for inputs, labels in train_loader:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(train_loader)
            self.loss_history.append(avg_loss)
            print(f"{self.name} - Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}")

        self.train_time = time.time() - start_time
        
        # Evaluate after training
        self.accuracy = self.evaluate(test_loader)
        
    def evaluate(self, test_loader):
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)
                
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        return accuracy_score(all_labels, all_preds)
    

def initialize_models():
    models_dict = {
        "SimpleCNNBN": SimpleCNN(num_classes=NUM_CLASSES, batch_norm=True),
        "SimpleCNN": SimpleCNN(num_classes=NUM_CLASSES, batch_norm=False),
        "ResNet18": models.resnet18(num_classes=NUM_CLASSES),
        #"EfficientNet-B0": models.efficientnet_b0(num_classes=NUM_CLASSES),
        "MobileNetV2": models.mobilenet_v2(num_classes=NUM_CLASSES),
    }
    return models_dict

def plot_results(benchmarks):
    _, ax = plt.subplots(1, 4, figsize=(18, 5))
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(benchmarks)))

    ax[0].bar([b.name for b in benchmarks], [b.train_time for b in benchmarks], color=colors)
    ax[0].set_title("Training Time")
    ax[0].set_ylabel("Seconds")
    
    # Accuracy
    ax[1].bar([b.name for b in benchmarks], [b.accuracy for b in benchmarks], color=colors)
    ax[1].set_title("Test Accuracy")
    ax[1].set_ylim(0, 1)

    
    # Loss Curves
    for b, c in zip(benchmarks, colors):
        ax[2].plot(b.loss_history, label=b.name, color=c)
    ax[2].set_title("Training Loss Curves")
    ax[2].legend()
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "benchmark_results.png")
    plt.show()

# Main Benchmarking Workflow
def main():
    # Setup directories
    RESULTS_DIR.mkdir(exist_ok=True)
    
    # Generate data
    train_dataset = ArrowDataset(NUM_SAMPLES, IMG_SIZE, NUM_CLASSES)
    test_dataset = ArrowDataset(NUM_SAMPLES // 5, IMG_SIZE, NUM_CLASSES)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # Initialize models
    models_dict = initialize_models()
    
    # Run benchmarks
    benchmarks: list[ModelBenchmark] = []
    for name, model in models_dict.items():
        print(f"\nBenchmarking {name}...")
        
        # Force garbage collection
        gc.collect()
        torch.cuda.empty_cache()
        
        # Run benchmark
        benchmark = ModelBenchmark(model, name)
        benchmark.train(train_loader, test_loader)
        benchmarks.append(benchmark)
        
        # Print results
        print(f"{name} Results:")
        print(f"Training Time: {benchmark.train_time:.2f}s")
        print(f"Test Accuracy: {benchmark.accuracy:.4f}\n")
    
    # Save and plot results
    plot_results(benchmarks)
    
    # Save numerical results
    results = [{
        "Model": b.name,
        "Training Time": b.train_time,
        "Accuracy": b.accuracy
    } for b in benchmarks]
    
    pd.DataFrame(results).to_csv(RESULTS_DIR / "benchmark_results.csv", index=False)

if __name__ == "__main__":
    main()