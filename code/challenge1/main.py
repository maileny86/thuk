import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from dataset_generator import SyntheticDatasetGenerator
from classifier import SimpleClassifier, train_model, evaluate_model

if __name__ == "__main__":
    # Generate synthetic dataset
    generator = SyntheticDatasetGenerator(n_samples=1000, n_features=20, n_classes=3)
    X, y = generator.generate()
    X_train, X_val, X_test, y_train, y_val, y_test = generator.split_data(X, y)

    # Convert data to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_val = torch.tensor(y_val, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # Create DataLoaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)

    # Initialize model, loss, and optimizer
    model = SimpleClassifier(input_dim=20, output_dim=3)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10)

    # Evaluate the model
    test_acc = evaluate_model(model, test_loader)
    print(f"Test Accuracy: {test_acc:.4f}")