import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time

def train_resnet(hyperparameters):
    # Set device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    num_epochs = hyperparameters.get("num_epochs", 1)
    batch_size = hyperparameters.get("batch_size", 64)
    learning_rate = hyperparameters.get("learning_rate", 0.001)
    num_layers = hyperparameters.get("num_layers", 18)  # Default to ResNet-18

    # Load and preprocess CIFAR-10 dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Define ResNet model
    if num_layers == 18:
        resnet = torchvision.models.resnet18().to(device)
    elif num_layers == 34:
        resnet = torchvision.models.resnet34().to(device)
    elif num_layers == 50:
        resnet = torchvision.models.resnet50().to(device)
    elif num_layers == 101:
        resnet = torchvision.models.resnet101().to(device)
    elif num_layers == 152:
        resnet = torchvision.models.resnet152().to(device)
    else:
        raise ValueError("Unsupported number of layers. Supported options: 18, 34, 50, 101, 152")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(resnet.parameters(), lr=learning_rate)

    # Training loop
    start_time = time.time()
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = resnet(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    end_time = time.time()
    runtime_speed = (end_time - start_time) / (num_epochs * len(train_loader))

    # Test the model
    resnet.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = resnet(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total

    return runtime_speed, accuracy

# Example usage:
hyperparameters = {
    "batch_size": 64,
    "learning_rate": 0.001,
    "num_layers": 18  # You can change this to 18, 34, 101, or 152
}

speed, accuracy = train_resnet(hyperparameters)
print(f"Runtime Speed: {speed} seconds per epoch per batch")
print(f"Accuracy: {accuracy * 100:.2f}%")