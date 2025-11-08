import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.25
MOMENTUM = 0.9

DATA_PATH = '/kaggle/input/cifar10-python'
PLOT_SAVE_PATH = '/kaggle/working/activation_comparison_plot.png'

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = torchvision.datasets.CIFAR10(
    root=DATA_PATH, train=True, download=False, transform=transform
)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
)

test_dataset = torchvision.datasets.CIFAR10(
    root=DATA_PATH, train=False, download=False, transform=transform
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
)

class LeNet(nn.Module):
    def __init__(self, activation_fn=F.relu):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        self.activation_fn = activation_fn

    def forward(self, x):
        x = self.pool(self.activation_fn(self.conv1(x)))
        x = self.pool(self.activation_fn(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.activation_fn(self.fc1(x))
        x = self.activation_fn(self.fc2(x))
        x = self.fc3(x)
        return x


def train(model, optimizer, epoch):
    model.train()
    running_loss = 0.0
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    return running_loss / len(train_loader)


def test(model):
    model.eval()
    correct = 0
    total = 0
    test_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    avg_loss = test_loss / len(test_loader)
    return avg_loss, accuracy

activation_functions = {
    'ReLU': F.relu,
    'Tanh': F.tanh,
    'Sigmoid': torch.sigmoid
}

all_results = {}

print("--- Starting Activation Function Comparison Experiment ---")

for name, fn in activation_functions.items():
    print(f"\n>>> Training model with {name} activation...")

    model = LeNet(activation_fn=fn).to(DEVICE)

    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

    criterion = nn.CrossEntropyLoss()

    history_test_acc = []
    history_test_loss = []

    for epoch in range(EPOCHS):
        train(model, optimizer, epoch)
        test_loss, test_acc = test(model)

        history_test_acc.append(test_acc)
        history_test_loss.append(test_loss)

        print(f"  Epoch {epoch + 1}/{EPOCHS} - Test Acc: {test_acc:.2f}%, Test Loss: {test_loss:.4f}")

    all_results[name] = {
        'accuracy': history_test_acc,
        'loss': history_test_loss
    }
    print(f"<<< Finished training with {name}. Final Accuracy: {history_test_acc[-1]:.2f}%")

print("\n--- All experiments finished. ---")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

for name, data in all_results.items():
    ax1.plot(range(1, EPOCHS + 1), data['accuracy'], marker='o', linestyle='-', label=name)

ax1.set_title('Test Accuracy Comparison on CIFAR-10', fontsize=16)
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Accuracy (%)', fontsize=12)
ax1.legend(fontsize=12)
ax1.set_xticks(range(1, EPOCHS + 1))
ax1.set_ylim(0, 100)

for name, data in all_results.items():
    ax2.plot(range(1, EPOCHS + 1), data['loss'], marker='o', linestyle='-', label=name)

ax2.set_title('Test Loss Comparison on CIFAR-10', fontsize=16)
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Loss', fontsize=12)
ax2.legend(fontsize=12)
ax2.set_xticks(range(1, EPOCHS + 1))

plt.tight_layout()
plt.savefig(PLOT_SAVE_PATH)
print(f"\nComparison plot saved to {PLOT_SAVE_PATH}")
plt.show()
