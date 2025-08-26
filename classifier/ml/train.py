# train.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image, ImageFilter, ImageOps
import random
from cnn_model import CNNModel

# ----------------------------
# Custom Transform for Stroke Thickness
# ----------------------------
class RandomStrokeTransform:
    """
    Simulate variable stroke thickness by dilating or eroding the digit.
    """
    def __init__(self, min_kernel=3, max_kernel=5, p=0.5):
        self.min_kernel = min_kernel
        self.max_kernel = max_kernel
        self.p = p  # probability to apply

    def __call__(self, img):
        if random.random() > self.p:
            return img

        img = img.convert('L')
        img = ImageOps.invert(img)
        img = img.point(lambda x: 0 if x < 128 else 255, '1')

        # Make sure kernel is odd
        kernel_size = random.choice([k for k in range(self.min_kernel, self.max_kernel+1) if k % 2 == 1])

        if random.random() < 0.5:
            img = img.filter(ImageFilter.MaxFilter(kernel_size))  # dilate
        else:
            img = img.filter(ImageFilter.MinFilter(kernel_size))  # erode

        img = ImageOps.invert(img)
        return img


# ----------------------------
# Device setup
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Data transforms
# ----------------------------
train_transforms = transforms.Compose([
    transforms.RandomAffine(
        degrees=15,            # rotate ±15°
        translate=(0.1,0.1),   # shift ±10%
        scale=(0.8,1.2)        # scale 80–120%
    ),
    RandomStrokeTransform(min_kernel=1, max_kernel=2, p=0.7),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ----------------------------
# Datasets & loaders
# ----------------------------
train_dataset = datasets.MNIST(root="./data", train=True, transform=train_transforms, download=True)
test_dataset = datasets.MNIST(root="./data", train=False, transform=test_transforms, download=True)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# ----------------------------
# Model, loss, optimizer
# ----------------------------
model = CNNModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ----------------------------
# Training loop
# ----------------------------
epochs = 5
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Evaluate
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    print(f"Epoch [{epoch+1}/{epochs}] Loss: {running_loss/len(train_loader):.4f}, Test Acc: {acc:.2f}%")

# ----------------------------
# Save model
# ----------------------------
torch.save(model.state_dict(), "mnist_cnn_aug.pth")
print("Model saved to mnist_cnn_aug.pth")
