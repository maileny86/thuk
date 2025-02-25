"""Challenge 5: Self-supervised learning and embeddings generation
Author: Milena Napiorkowska
"""
import os
from pathlib import Path
import random
from typing import Self
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# region constatants
CLASS_MAPPING: dict[str, int] = {"chinchilla": 0, "hamster": 1, "rabbit": 2}
EPOCHS = 2
BATCH_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = Path(Path(os.path.realpath(__file__)).parent) / "outputs"
RESULTS_DIR.mkdir(exist_ok=True)
# endregion

def split_images_to_train_and_val(
    data_dir: Path, ratio: float = 0.8, seed: int = 42
) -> tuple[list[Path], list[Path]]:
    """
    Splits images into training and validation sets.

    Args:
        data_dir: Path to the dataset directory.
        ratio: Proportion of images used for training. Defaults to 0.8.
        seed: Random seed for reproducibility. Defaults to 42.

    Returns:
        Lists of training and validation image paths.
    """
    random.seed(seed)
    train_paths, val_paths = [], []

    for class_dir in data_dir.iterdir():
        images = list(class_dir.glob("*.jpg"))
        random.shuffle(images)
        split_idx = int(len(images) * ratio)

        train_paths.extend(images[:split_idx])
        val_paths.extend(images[split_idx:])

    return train_paths, val_paths


class RodentsDataset(Dataset):
    """Custom dataset for the rodent images."""
    def __init__(self: Self, image_paths: list[Path], transform: transforms.Compose) -> None:
        self.image_paths = image_paths
        self.transform = transform
        self.labels = [CLASS_MAPPING[image.parent.name.lower()] for image in image_paths]

    def __len__(self: Self) -> int:
        return len(self.image_paths)

    def __getitem__(self: Self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        image = Image.open(self.image_paths[idx]).convert('RGB')
        view1 = self.transform(image)
        view2 = self.transform(image)
        return view1, view2, self.labels[idx]

def get_indices(dataset: RodentsDataset, split_ratio: float = 0.25) -> tuple[list[int], list[int]]:
    """Split the dataset into training and validation indices."""
    indices = list(range(len(dataset)))
    split = int(np.floor(split_ratio * len(dataset)))
    np.random.shuffle(indices)
    return indices[split:], indices[:split]


class ProjectionHead(nn.Module):
    """Projection head for the SimCLR model."""
    def __init__(self: Self, input_dim: int, output_dim: int = 128) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self: Self, x) -> torch.Tensor:
        """Forward pass."""
        return self.layers(x)


class SimCLR(nn.Module):
    """SimCLR model for self-supervised task."""
    def __init__(self: Self, base_model: nn.Module, projection_dim: int = 128) -> None:
        super().__init__()
        self.encoder = base_model
        in_features = base_model.fc.in_features
        self.encoder.fc = nn.Identity()
        self.projection = ProjectionHead(in_features, projection_dim)

    def forward(self: Self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.projection(self.encoder(x))


class SimCLRTrainer:
    """Trainer for the SimCLR model."""

    def __init__(
            self,
            model: SimCLR,
            train_loader: DataLoader,
            val_loader: DataLoader,
            lr: float = 0.001,
            weight_decay: float = 1e-6) -> None:
        self.model = model.to(DEVICE)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optim.SGD(
            model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=EPOCHS)

    def _nt_xent_loss(
            self,
            z_i: torch.Tensor,
            z_j: torch.Tensor,
            temperature: float = 0.5) -> torch.Tensor:
        """Computes the NT-Xent loss for contrastive learning."""
        z = torch.cat((z_i, z_j), dim=0) # Concatenate augmented views
        z = F.normalize(z, dim=1)  # Normalize embeddings
        similarity_matrix = torch.mm(z, z.T)  # Compute similarity matrix

        batch_size = z.shape[0] // 2
        labels = torch.arange(batch_size, device=DEVICE)
        labels = torch.cat([labels + batch_size, labels])

        # Mask to ignore self-similarities
        mask = torch.eye(batch_size * 2, dtype=torch.bool, device=DEVICE)
        similarity_matrix.masked_fill_(mask, float('-inf'))
        similarity_matrix /= temperature
        return F.cross_entropy(similarity_matrix, labels)

    def _run_epoch(self, loader: DataLoader, training: bool = True) -> float:
        """Runs one epoch of training or validation."""
        self.model.train(training)
        epoch_loss = 0.0

        with torch.set_grad_enabled(training):
            for view1, view2, _ in loader:
                view1, view2 = view1.to(DEVICE), view2.to(DEVICE)
                z_i, z_j = self.model(view1), self.model(view2)  # Forward pass
                loss = self._nt_xent_loss(z_i, z_j)  # Compute loss

                if training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                epoch_loss += loss.item() * view1.size(0)

        return epoch_loss / len(loader.dataset)

    def train(self, model_path: Path) -> None:
        """Train the model and save the best model state."""
        best_val_loss = float('inf')

        for epoch in range(EPOCHS):
            train_loss = self._run_epoch(self.train_loader, training=True)
            val_loss = self._run_epoch(self.val_loader, training=False)

            self.scheduler.step()

            print(f"Epoch {epoch+1}/{EPOCHS}\n"
                  f"Train Loss: {train_loss:.4f}\n"
                  f"Val Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), model_path)
                print(f"Saved model to {model_path}")

class EmbeddingExtractor:
    """Extracts embeddings for the SimCLR model."""
    def __init__(self: Self, model: SimCLR, dataloader: DataLoader) -> None:
        self.model = model.to(DEVICE)
        self.dataloader = dataloader

    def extract(self) -> tuple[np.ndarray, np.ndarray]:
        """Extracts embeddings from the SimCLR model's projection head."""
        self.model.eval()
        embeddings, labels = [], []

        with torch.no_grad():
            for view1, view2, lbls in self.dataloader:  # Both views provided
                images = torch.cat([view1, view2], dim=0).to(DEVICE)  # Concatenate both views
                emb = self.model(images)  # Extract embeddings
                embeddings.append(emb.cpu().numpy())
                labels.extend(lbls)

        return np.concatenate(embeddings), np.array(labels)


class TSNEVisualizer:
    """Visualizes the embeddings using t-SNE."""
    @staticmethod
    def plot(features: np.ndarray, labels: np.ndarray, class_names: list[str]) -> None:
        """Plots 2D t-SNE visualization of the embeddings."""
        tsne = TSNE(n_components=2, perplexity=min(30, len(features) - 1), random_state=42)
        reduced = tsne.fit_transform(features)

        plt.figure(figsize=(10, 8))
        unique_labels = np.unique(labels)

        for label in unique_labels:
            idxs = np.where(labels == label)
            plt.scatter(reduced[idxs, 0], reduced[idxs, 1], label=class_names[label], alpha=0.6)

        plt.legend()
        plt.title("Feature Map Visualization with t-SNE")
        plt.savefig(RESULTS_DIR / "feature_map.png")
        plt.show()

def main(data_dir: Path, model_name: str) -> None:
    """Main function to train the SimCLR model and visualize the embeddings."""

    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([
            transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))
        ], p=0.5),
        transforms.RandomApply([transforms.RandomSolarize(threshold=192)], p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.1), value='random')
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_images, val_images = split_images_to_train_and_val(data_dir)

    train_dataset = RodentsDataset(train_images, transform=train_transforms)
    val_dataset = RodentsDataset(val_images, transform=val_transforms)


    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    model = SimCLR(models.resnet50(pretrained=True)).to(DEVICE)

    trainer = SimCLRTrainer(model, train_loader, val_loader)
    model_path = RESULTS_DIR / f"{model_name}.pt"
    trainer.train(model_path)

    extractor = EmbeddingExtractor(model, val_loader)
    embeddings, labels = extractor.extract()

    TSNEVisualizer.plot(embeddings, labels, list(CLASS_MAPPING.keys()))
    print("Embeddings visualization saved to outputs/feature_map.png")


if __name__ == "__main__":
    input_data_dir = Path("C:\\Users\\Milena\\Downloads\\thales\\challenges\\challenge5\\data")
    main(input_data_dir, model_name="simclr.pth")
