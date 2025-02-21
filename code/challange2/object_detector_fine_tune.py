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
import typer
from sklearn.metrics import classification_report

# region constants
CLASS_MAPPING: dict[str, int] = {"chinchilla": 0, "hamster": 1, "rabbit": 2}
DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRANSFORM: transforms = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
    ])
# endregion

app = typer.Typer()

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

@app.command()
def train_model(
    model: torch.nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    criterion: torch.nn.CrossEntropyLoss,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    patience: int,
    model_path: Path
) -> None:
    """
    Train the model and save the best model based on the validation loss.

    Args:
        model: The model to train.
        train_dataloader: DataLoader for the training data.
        val_dataloader: DataLoader for the validation data.
        criterion: Loss function.
        optimizer: Optimizer for training.
        num_epochs: Number of epochs to train.
        patience: Number of epochs to wait for improvement in validation loss before early stopping.
        model_path: Path to save the best model.
    """
    model.to(DEVICE)
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_train_loss = 0.0
        for inputs, labels in train_dataloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE, dtype=torch.long)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()
        train_loss = running_train_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}, Training Loss: {train_loss}")

        # Validation phase
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_dataloader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE, dtype=torch.long)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()
        val_loss = running_val_loss / len(val_dataloader)
        print(f"Epoch {epoch+1}, Validation Loss: {val_loss}")

        # Early stopping and model saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
            print(f"Validation loss improved. Model saved at {model_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break
@app.command()
def evaluate_model(model: models, model_path: Path, dataloader: DataLoader) -> None:
    """Evaluate the model using the test set."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model file {model_path} not found.")
    
    model.load_state_dict(torch.load(model_path))
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


def main(
    dataset_path: Path,
    train_annotation_file: Path,
    val_annotation_file: Path,
    test_annotation_file: Path,
    model_path: Path) -> None:
    """Main function to train and evaluate the model."""
    

    # Load dataset
    train_dataset = RodentsDataset(train_annotation_file, dataset_path, transform=TRANSFORM)
    train_loader = DataLoader(train_dataset, batch_size=3, shuffle=True)
    val_dataset = RodentsDataset(val_annotation_file, dataset_path, transform=TRANSFORM)
    val_loader = DataLoader(val_dataset, batch_size=3, shuffle=False)

    # Initialize model
    model = get_resnet(num_classes=3)

    # Define loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)

    # Train model
    train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        num_epochs=100,
        patience=3,
        model_path=model_path)

    # Evaluate the model
    test_dataset = RodentsDataset(test_annotation_file, dataset_path, transform=TRANSFORM)
    test_loader = DataLoader(test_dataset, batch_size=3, shuffle=False)
    evaluate_model(model, model_path, test_loader)

if __name__ == "__main__":
     typer.run(main)

