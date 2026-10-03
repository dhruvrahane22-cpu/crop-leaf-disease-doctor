import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

import sys
import os

# Append project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.custom_cnn import CustomLeafCNN

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dir = "data/processed/train"
    val_dir = "data/processed/val"

    if not os.path.exists(train_dir):
        print(f"Error: {train_dir} does not exist!")
        return

    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    num_classes = len(train_dataset.classes)
    model = CustomLeafCNN(num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    os.makedirs("logs", exist_ok=True)
    save_path = "logs/custom_cnn.pth"

    epochs = 5  # Increase epochs as needed
    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data)
            total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = correct.double() / total
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

    # Save trained weights
    torch.save(model.state_dict(), save_path)
    print(f"Model saved successfully to {save_path}!")

if __name__ == "__main__":
    train()