import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

transform_train = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5)
        )
    ]
)

transform_test = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5)
        )
    ]
)

# load CIFAR-10 datasets
train_datasets = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_datasets = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)


train_loader = DataLoader(dataset=train_datasets, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_datasets, batch_size=64, shuffle=True)

print(f'Training Data Size: {len(train_datasets)}')
print(f'Testing Data Size: {len(test_datasets)}')


class EnhancedCNN(nn.Module):
    def __init__(self):
        super(EnhancedCNN, self).__init__()
        self.conv1 = nn.Conv2d(3,6,5)
        self.bn1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6,16,5)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2,2)
        self.dropout = nn.Dropout(0.5)

        # calculate the size of the output from the convolutional layers dynamically
        self._calculate_conv_output()
        self.fc1 = nn.Linear(self.conv_output_size, 120)
        self.fc2 = nn.Linear(120, 85)
        self.fc3 = nn.Linear(85, 15)

    def _calculate_conv_output(self):
        # dummy input tensor with the same size as the input images
        dummy_input = torch.zeros(1,3,32,32)
        with torch.no_grad():
            output = self.pool(F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(dummy_input)))))))
        self.conv_output_size = output.numel()

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = EnhancedCNN()
print(model)



# define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)
training_loss = []


# training loop
def train_model(
        model, train_loader,
        criterion, optimizer,
        epochs=22):

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        training_loss.append(epoch_loss)
        print(f"Epochs: {epoch+1}, Loss: {epoch_loss:.5f}")



train_model(
    model, train_loader,
    criterion, optimizer,
)


# evaluation loop
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs,1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Testing Accuracy: {100 * correct / total:.4f}%')


evaluate_model(model, test_loader)


plt.plot(training_loss, label='Training Loss')
plt.title('Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()