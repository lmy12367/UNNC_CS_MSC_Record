import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np


class And(nn.Module):
    def __init__(self):
        super(And, self).__init__()
        self.linear1 = nn.Linear(2, 4)
        self.linear2 = nn.Linear(4, 1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.activation(self.linear1(x))
        x = self.activation(self.linear2(x))
        return x


def train_model(model, X_train, y_train, epochs=1000, lr=0.1):

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    losses = []
    for epoch in range(epochs):
        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

    return losses


def main():
    torch.manual_seed(20828220)
    np.random.seed(20828220)

    X_train = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
    y_train = torch.tensor([[0], [0], [0], [1]], dtype=torch.float32)

    model = And()

    print("开始训练...")
    losses = train_model(model, X_train, y_train, epochs=1000, lr=0.1)

    model.eval()
    with torch.no_grad():
        predictions = model(X_train)

        x_min, x_max = -0.5, 1.5
        y_min, y_max = -0.5, 1.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                             np.linspace(y_min, y_max, 100))

        grid_points = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)

        grid_predictions = model(grid_points)
        grid_predictions = grid_predictions.reshape(xx.shape)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.plot(losses)
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True)

        contour = ax2.contourf(xx, yy, grid_predictions.numpy(), levels=20, alpha=0.8, cmap='RdYlBu')
        ax2.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=100,
                    edgecolors='black', linewidth=2, cmap='RdYlBu', vmin=0, vmax=1)

        for i, (x, y) in enumerate(X_train):
            pred_val = predictions[i].item()
            ax2.annotate(f'{pred_val:.3f}', (x, y), xytext=(5, 5),
                         textcoords='offset points', fontsize=10, fontweight='bold')

        ax2.set_title('AND Gate Decision Boundary')
        ax2.set_xlabel('Input 1')
        ax2.set_ylabel('Input 2')
        ax2.set_xlim(x_min, x_max)
        ax2.set_ylim(y_min, y_max)

        plt.colorbar(contour, ax=ax2)

        plt.tight_layout()
        plt.savefig('./and_model_results.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("\n预测结果:")
        for i, (input_val, target, pred) in enumerate(zip(X_train, y_train, predictions)):
            print(f"输入: {input_val.numpy()}, 目标: {target.item():.0f}, 预测: {pred.item():.3f}")


if __name__ == "__main__":
    main()
