import torch
import torch.nn as nn
import torch.nn.functional as F
import dask
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ]
)


# load datasets
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='.data', train=False, transform=transform, download=True)

# create data loader
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)
test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=True
)

print(f'Training Data Size: {len(train_dataset)}')
print(f'Test Data Size: {len(test_dataset)}')


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = NeuralNetwork()
print(model)

# define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.002)

# training loop
def train_model(model, train_loader, criterion, optimizer, epochs=5):
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0

        for images, labels in train_loader:
            # zero gradients
            optimizer.zero_grad()

            # forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}')

train_model(model, train_loader, criterion, optimizer)


# evaluation loop
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Test Accuracy: {100 * correct/total:.3f}%')

evaluate_model(model, test_loader)

# save the model
torch.save(model.state_dict(), 'mnist_model.pth')

# reload the model
loaded_model = NeuralNetwork()
loaded_model.load_state_dict(torch.load('mnist_model.pth'))


# verify loaded model performance
evaluate_model(loaded_model, test_loader)


# update optimizer with a new learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
train_model(model, train_loader, criterion, optimizer)
evaluate_model(model, test_loader)