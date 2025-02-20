"""Challange 2: Object Detector Fine-Tuning
Author: Milena Napiorkowska
"""
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import torch.nn as nn
from numpy.typing import NDArray
import pandas as pd
from sklearn.metrics import classification_report

# region constants
CLASS_MAPPING: dict[str, int] = {"chinchilla": 0, "hamster": 1, "rabbit": 2}
DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# endregion

class RodentsDataset(Dataset):
    """Custom dataset for the rodent images."""
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(annotations_file, sep=" ", header=None)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        filename = self.img_labels.iloc[idx, 0]
        img_path = Path(self.img_dir) / filename
        image = Image.open(img_path).convert('RGB')
        category = filename.split("\\")[0].lower()
        label = CLASS_MAPPING[category]
        if self.transform:
            image = self.transform(image)
        return image, label


def get_resnet(num_classes: int) -> models:
    """Get a ResNet model with a custom head."""
    model = models.resnet18(pretrained=True)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    return model


def train_model(
    model: models,
    dataloader: DataLoader,
    criterion: torch.nn.CrossEntropyLoss,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    patience: int,
    model_path: Path) -> None:
    """Train the model and save the best model based on the loss."""
    model.to(DEVICE)
    best_loss = float('inf')
    patience_counter = 0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE, dtype=torch.long)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Loss: {epoch_loss}")
        
        # Early stopping if the loss does not improve
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

def evaluate_model(model: models, dataloader: DataLoader) -> None:
    """Evaluate the model using the test set."""
    model.eval()
    y_true: list[NDArray] = []
    y_pred: list[NDArray] = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE, dtype=torch.long)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.to(DEVICE).numpy())
            y_pred.extend(predicted.to(DEVICE).numpy())
    
    print(classification_report(y_true, y_pred, target_names=CLASS_MAPPING.keys()))

# Define paths
path = Path("C:\\Users\\Milena\\Downloads\\coding_challenge\\challenge2")
dataset_path = path / "data"
train_annotation_file = path / "train_set.txt"
test_annotation_file = Path(path) / "test_set.txt"

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load dataset
train_dataset = RodentsDataset(train_annotation_file, dataset_path, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_dataset = RodentsDataset(test_annotation_file, dataset_path, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# Initialize model
model = get_resnet(num_classes=3)

# Evaluate the model before fine-tuning
evaluate_model(model, test_loader)

# Define loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)

# Train model
train_model(
    model,
    train_loader,
    criterion,
    optimizer,
    num_epochs=100,
    patience=3,
    model_path=path / "model.pth")

# Evaluate the model
evaluate_model(model, test_loader)
