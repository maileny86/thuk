import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os
from PIL import Image
from typing import Self

# region constatants
EPOCHS = 20
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# endregion

class RodentsDataset(Dataset):
    def __init__(self: Self, root_dir: Path, transform: transforms.Compose = None) -> None:
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths, self.labels = self._load_dataset()
        self.class_names = sorted(os.listdir(root_dir))

    def _load_dataset(self: Self) -> tuple[list[str], list[int]]:
        image_paths, labels = [], []
        for label, class_name in enumerate(sorted(os.listdir(self.root_dir))):
            class_dir = os.path.join(self.root_dir, class_name)
            for img_name in os.listdir(class_dir):
                image_paths.append(os.path.join(class_dir, img_name))
                labels.append(label)
        return image_paths, labels

    def __len__(self: Self) -> int:
        return len(self.image_paths)

    def __getitem__(self: Self, idx: int) -> tuple[torch.Tensor, int]:
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]


class ProjectionHead(nn.Module):
    def __init__(self: Self, input_dim: int, output_dim: int = 128) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self: Self, x) -> torch.Tensor:
        return self.layers(x)


class SimCLR(nn.Module):
    def __init__(self: Self, base_model: nn.Module, projection_dim: int = 128) -> None:
        super().__init__()
        self.encoder = base_model
        in_features = base_model.fc.in_features
        self.encoder.fc = nn.Identity()
        self.projection = ProjectionHead(in_features, projection_dim)

    def forward(self: Self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(self.encoder(x))


class SimCLRTrainer:
    def __init__(self: Self, model: SimCLR, dataloader: DataLoader, lr: float = 0.001) -> None:
        self.model = model.to(DEVICE)
        self.dataloader = dataloader
        self.optimizer = optim.Adam(model.parameters(), lr=lr)

    def _nt_xent_loss(self: Self, z_i: torch.Tensor, z_j: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
        z = torch.cat((z_i, z_j), dim=0)
        z = F.normalize(z, dim=1)
        similarity = torch.mm(z, z.T)
        labels = torch.arange(z.shape[0], device=DEVICE)
        labels = (labels + (z.shape[0] // 2)) % z.shape[0]
        return F.cross_entropy(similarity / temperature, labels)

    def train(self: Self) -> None:
        self.model.train()
        for epoch in range(EPOCHS):
            epoch_loss = 0.0
            for images, _ in self.dataloader:
                images = torch.cat([images, images], dim=0).to(DEVICE)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self._nt_xent_loss(outputs[:len(images)//2], outputs[len(images)//2:])
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {epoch_loss/len(self.dataloader):.4f}")
        torch.save(self.model.state_dict(), "simclr_model.pth")
        print("Model saved!")


class FeatureExtractor:
    def __init__(self, model, dataloader):
        self.model = model.to(DEVICE)
        self.dataloader = dataloader

    def extract_features(self):
        self.model.eval()
        features, labels = [], []
        with torch.no_grad():
            for images, lbls in self.dataloader:
                images = images.to(DEVICE)
                feat = self.model.encoder(images)
                features.append(feat.cpu().numpy())
                labels.extend(lbls)
        return np.concatenate(features), np.array(labels)


class TSNEVisualizer:
    @staticmethod
    def plot(features, labels, class_names):
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        reduced = tsne.fit_transform(features)
        plt.figure(figsize=(10, 8))
        for i, label in enumerate(set(labels)):
            idxs = labels == label
            plt.scatter(reduced[idxs, 0], reduced[idxs, 1], label=class_names[label], alpha=0.6)
        plt.legend()
        plt.title("Feature Map Visualization with t-SNE")
        plt.savefig("feature_map.png")
        plt.show()


if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.5, 0.5, 0.5, 0.5)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset = RodentsDataset("C:\\Users\\Milena\\Downloads\\thales\\challenges\\challenge5\\data", transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = SimCLR(models.resnet18(pretrained=True)).to(DEVICE)

    trainer = SimCLRTrainer(model, dataloader)
    trainer.train()

    extractor = FeatureExtractor(model, dataloader)
    features, labels = extractor.extract_features()

    TSNEVisualizer.plot(features, labels, dataset.class_names)
