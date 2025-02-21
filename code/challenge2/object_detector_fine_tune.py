"""Challenge 2: Object Detector Fine-Tuning
Author: Milena Napiorkowska
"""
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import torch.nn as nn
import pandas as pd
import typer
from typing import Self
from sklearn.metrics import classification_report

# region constants
CLASS_MAPPING: dict[str, int] = {"chinchilla": 0, "hamster": 1, "rabbit": 2}
DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRANSFORM: transforms.Compose = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
    ])
BATCH_SIZE: int = 3
# endregion

app = typer.Typer()

class RodentsDataset(Dataset):
    """Custom dataset for the rodent images."""
    def __init__(self, annotations_file, img_dir, transform=None) -> None:
        self.img_labels = pd.read_csv(annotations_file, sep=" ", header=None)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self: Self) -> int:
        return len(self.img_labels)

    def __getitem__(self: Self, idx: int):
        filename = self.img_labels.iloc[idx, 0]
        img_path = Path(self.img_dir) / filename
        image = Image.open(img_path).convert('RGB')
        category = filename.split("\\")[0].lower()
        label = CLASS_MAPPING[category]
        if self.transform:
            image = self.transform(image)
        return image, label

# region private
def _get_resnet(num_classes: int) -> nn.Module: 
    """Get a ResNet model with a custom head."""
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    return model

def _train_model(
    model: torch.nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    criterion: torch.nn.CrossEntropyLoss,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    patience: int,
    model_path: Path
) -> None:
    model.to(DEVICE)
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0
        for inputs, labels in train_dataloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()
        train_loss = running_train_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}, Training Loss: {train_loss}")

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_dataloader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()
        val_loss = running_val_loss / len(val_dataloader)
        print(f"Epoch {epoch+1}, Validation Loss: {val_loss}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
            print(f"Val loss improved. Saving model to: {model_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping!")
                break

def _evaluate_model(model: nn.Module, model_path: Path, dataloader: DataLoader) -> None:
    model.load_state_dict(torch.load(model_path))
    model.to(DEVICE)  
    model.eval()
    y_true: list[int] = [] 
    y_pred: list[int] = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy()) 
            y_pred.extend(predicted.cpu().numpy())
    print(classification_report(y_true, y_pred, target_names=CLASS_MAPPING.keys()))
# endregion

@app.command()
def train(
    dataset_path: Path, 
    train_annotation_file: Path, 
    val_annotation_file: Path, 
    model_path: Path,
    num_epochs: int) -> None:
    """Train a model to classify rodent images."""
    # Load train and val datasets
    train_dataset = RodentsDataset(train_annotation_file, dataset_path, transform=TRANSFORM)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataset = RodentsDataset(val_annotation_file, dataset_path, transform=TRANSFORM)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Load pretrained model
    model = _get_resnet(num_classes=len(CLASS_MAPPING))

    # Define loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001, weight_decay=1e-4)

    _train_model(
        model, 
        train_loader, 
        val_loader, 
        criterion, 
        optimizer, 
        num_epochs=num_epochs, 
        patience=5, 
        model_path=model_path)

@app.command()  
def eval(dataset_path: Path, test_annotation_file: Path, model_path: Path) -> None:
    """Evaluate a trained model on the test dataset."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model file {model_path} not found.")
    
    # Load test dataset
    dataset = RodentsDataset(test_annotation_file, dataset_path, transform=TRANSFORM)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Load model
    model = _get_resnet(num_classes=len(CLASS_MAPPING))
    model.to(DEVICE)  

    # Evaluate model
    _evaluate_model(model, model_path, dataloader)

if __name__ == "__main__":
    app()  