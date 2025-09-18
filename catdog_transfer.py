# -*- coding: utf-8 -*-
# CatDog Classification with Transfer Learning

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from google.colab import files



# Install Kaggle CLI
!pip install -q kaggle

# Upload kaggle.json
files.upload()

# Move to kaggle folder
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/kaggle.json
!chmod 600 ~/.kaggle/kaggle.json

# Download dataset
!kaggle datasets download -d tongpython/cat-and-dog
!unzip -q /content/cat-and-dog.zip -d /content/data/


img_path = "/content/data/training_set/training_set/cats/cat.1.jpg"
img = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
plt.title("Sample Image")
plt.axis("off")
plt.show()


transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

train_dataset = datasets.ImageFolder(
    root='/content/data/training_set/training_set',
    transform=transform
)

test_dataset = datasets.ImageFolder(
    root='/content/data/test_set/test_set',
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = models.resnet50(pretrained=True)

# Freeze all layers except final layer
for param in model.parameters():
    param.requires_grad = False

# Replace the classifier
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)



def train_one_epoch(epoch):
    model.train()
    running_loss = 0.0

    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if (i + 1) % 10 == 0:
            print(f"[Epoch {epoch+1}] Step {i+1}/{len(train_loader)}, Loss: {loss.item():.4f}")

    return running_loss / len(train_loader)

def evaluate():
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    return acc


num_epochs = 10

for epoch in range(num_epochs):
    loss = train_one_epoch(epoch)
    acc = evaluate()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss:.4f}, Test Accuracy: {acc:.2f}%")


