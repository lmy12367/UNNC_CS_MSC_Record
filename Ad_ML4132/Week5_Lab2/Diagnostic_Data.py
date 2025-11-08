import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import matplotlib.pyplot as plt
import torch


class Diabetees_Dataset(Dataset):
    def __init__(self, filepath):
        self.data = pd.read_csv(filepath)
        x_raw = self.data.drop(["Outcome"], axis=1).values
        y_raw = self.data["Outcome"].values
        self.scaler = StandardScaler()
        x_scaled = self.scaler.fit_transform(x_raw)
        self.x_data = torch.FloatTensor(x_scaled)
        self.y_data = torch.FloatTensor(y_raw).unsqueeze(1)
        self.len = self.data.shape[0]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


class Diabetees_Model(nn.Module):
    def __init__(self):
        super(Diabetees_Model, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


def evaluate_model(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy


if __name__ == '__main__':
    dataset = Diabetees_Dataset("diabetes.csv")
    train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=2)

    model = Diabetees_Model()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.BCELoss()
    optim = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.5, patience=10, verbose=True)

    epochs = []
    accuracies = []

    for epoch in range(100):
        epochs.append(epoch)
        loss_num = 0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optim.zero_grad()
            y_pred = model(inputs)
            loss = criterion(y_pred, labels)
            loss.backward()
            optim.step()
            loss_num += loss.item()

        avg_loss = loss_num / len(train_loader)
        scheduler.step(avg_loss)

        train_accuracy = evaluate_model(model, train_loader, device)
        accuracies.append(train_accuracy)

        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Average Loss: {avg_loss:.4f}, Accuracy: {train_accuracy:.2f}%')

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, accuracies, color='blue', linewidth=2)
    plt.title('Training Accuracy')
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Epoch')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("train_accuracy.png", dpi=300)
    plt.show()

    final_accuracy = evaluate_model(model, train_loader, device)
    print(f'训练准确率: {final_accuracy:.2f}%')
