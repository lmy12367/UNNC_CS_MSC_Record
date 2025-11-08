import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import time

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 0.01
MOMENTUM = 0.9
DATA_PATH = '/kaggle/input/cifar10-python'
PLOT_SAVE_PATH = '/kaggle/working/resnet_activation_comparison.png'

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = torchvision.datasets.CIFAR10(root=DATA_PATH, train=True, download=False, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
test_dataset = torchvision.datasets.CIFAR10(root=DATA_PATH, train=False, download=False, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, activation_fn=nn.ReLU):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.activation = activation_fn(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.activation(out)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, activation_fn=nn.ReLU):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.activation = activation_fn(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.activation(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.activation(out)


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10, activation_fn=nn.ReLU):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.activation = activation_fn(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], activation_fn=activation_fn)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, activation_fn=activation_fn)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, activation_fn=activation_fn)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, activation_fn=activation_fn)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1, activation_fn=nn.ReLU):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample, activation_fn))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, activation_fn=activation_fn))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def resnet18(num_classes=10, activation_fn=nn.ReLU): return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, activation_fn)


def resnet34(num_classes=10, activation_fn=nn.ReLU): return ResNet(BasicBlock, [3, 4, 6, 3], num_classes, activation_fn)


def resnet50(num_classes=10, activation_fn=nn.ReLU): return ResNet(Bottleneck, [3, 4, 6, 3], num_classes, activation_fn)


def resnet101(num_classes=10, activation_fn=nn.ReLU): return ResNet(Bottleneck, [3, 4, 23, 3], num_classes,
                                                                    activation_fn)


def train(model, optimizer, criterion, epoch):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(train_loader)


def test(model, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / len(test_loader.dataset)
    return test_loss / len(test_loader), accuracy


resnet_models = {
    'ResNet18': resnet18,
    'ResNet34': resnet34,
    'ResNet50': resnet50,
    'ResNet101': resnet101
}

activation_classes = {
    'ReLU': nn.ReLU,
    'Tanh': nn.Tanh,
    'LeakyReLU': nn.LeakyReLU,
    'Sigmoid': nn.Sigmoid
}

all_results = {}
print("--- Starting ResNet & Activation Function Comparison Experiment ---")

for model_name, model_fn in resnet_models.items():
    for act_name, act_class in activation_classes.items():
        print(f"\n>>> Training {model_name} with {act_name}...")
        model = model_fn(num_classes=10, activation_fn=act_class).to(DEVICE)
        optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
        criterion = nn.CrossEntropyLoss()

        history_acc = []
        for epoch in range(EPOCHS):
            train(model, optimizer, criterion, epoch)
            _, test_acc = test(model, criterion)
            history_acc.append(test_acc)
            print(f"  Epoch {epoch + 1}/{EPOCHS} - Test Acc: {test_acc:.2f}%")

        all_results[f'{model_name} ({act_name})'] = history_acc
        print(f"<<< Finished {model_name} ({act_name}). Final Acc: {history_acc[-1]:.2f}%")

print("\n--- All experiments finished. ---")

plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(20, 12))
colors = plt.cm.tab10.colors
linestyles = ['-', '--', '-.', ':']

for i, (label, history) in enumerate(all_results.items()):
    model_part, act_part = label.split(' (')
    act_part = act_part[:-1]
    color_idx = list(resnet_models.keys()).index(model_part)
    linestyle_idx = list(activation_classes.keys()).index(act_part)
    plt.plot(range(1, EPOCHS + 1), history, label=label, color=colors[color_idx % 10],
             linestyle=linestyles[linestyle_idx % 4], linewidth=2.5, marker='o', markersize=4)

plt.title('ResNet Architectures & Activation Functions Comparison on CIFAR-10', fontsize=20)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Test Accuracy (%)', fontsize=14)
plt.xticks(range(1, EPOCHS + 1))
plt.ylim(0, 100)
plt.legend(fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout(rect=[0, 0, 0.8, 1])
plt.savefig(PLOT_SAVE_PATH)
print(f"\nComparison plot saved to {PLOT_SAVE_PATH}")
plt.show()
