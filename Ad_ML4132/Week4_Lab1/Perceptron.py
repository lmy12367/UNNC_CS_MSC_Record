import random
import numpy as np
import matplotlib.pyplot as plt


class Perceptron:
    def __init__(self, inputs_num, lr):
        self.inputs_num = inputs_num
        self.lr = lr
        self.weights = [random.uniform(-1, 1) for i in range(inputs_num + 1)]

    def forward(self, inputs):
        R = self.weights[0]
        for i in range(len(inputs)):
            R += self.weights[i + 1] * inputs[i]
        return 1 if R > 0 else 0

    def update(self, inputs, y_true, y_pred):
        error = y_true - y_pred
        self.weights[0] += self.lr * error  # 更新偏置项
        for i in range(len(inputs)):
            self.weights[i + 1] += self.lr * error * inputs[i]  # 更新权重


def plot_decision_boundary(ax, weights, xlim, ylim):
    w0, w1, w2 = weights

    if abs(w2) < 1e-6:
        if abs(w1) < 1e-6:
            return
        else:
            x_val = -w0 / w1
            ax.axvline(x=x_val, color='g', linewidth=2, label='Decision Boundary')
    else:
        x_vals = np.linspace(xlim[0], xlim[1], 100)
        y_vals = (-w1 * x_vals - w0) / w2

        mask = (y_vals >= ylim[0]) & (y_vals <= ylim[1])
        if np.any(mask):
            ax.plot(x_vals[mask], y_vals[mask], 'g-', linewidth=2, label='Decision Boundary')


if __name__ == "__main__":
    random.seed(20808220)
    np.random.seed(20828220)

    X_original = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y_original = [0, 0, 0, 1]
    X_new = [[-0.5, 0.5], [-0.2, -0.2], [1.2, 0.2], [1.2, 1.2]]
    y_new = [0, 0, 1, 1]
    X = X_original + X_new
    y = y_original + y_new

    fig, ax = plt.subplots(figsize=(8, 6))
    p = Perceptron(inputs_num=2, lr=0.1)

    class_0_points = np.array([X[i] for i in range(len(y)) if y[i] == 0])
    class_1_points = np.array([X[i] for i in range(len(y)) if y[i] == 1])
    ax.scatter(class_0_points[:, 0], class_0_points[:, 1], color='blue', label='Class 0 (False)', s=100)
    ax.scatter(class_1_points[:, 0], class_1_points[:, 1], color='red', marker='x', s=100, linewidth=2,
               label='Class 1 (True)')

    ax.set_xlabel('Input 1')
    ax.set_ylabel('Input 2')
    ax.set_title('Perceptron Training for AND Gate (Extended Data)')
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(-0.6, 1.6)
    ax.grid(True)

    print("Initial weights:", p.weights)

    for epoch in range(5000):
        total_error = 0
        for inputs, label in zip(X, y):
            prediction = p.forward(inputs)
            error = label - prediction
            total_error += abs(error)
            if error != 0:
                p.update(inputs, label, prediction)

        if total_error == 0:
            print(f"Converged at epoch {epoch}")
            break

    print("\nTraining finished.")
    print(f"Final weights: {p.weights}")

    plot_decision_boundary(ax, p.weights, ax.get_xlim(), ax.get_ylim())

    ax.set_title(f'Final Result - Epoch: {epoch}, Total Error: {total_error}')
    ax.legend()

    print("\nFinal Predictions:")
    for inputs in X:
        result = p.forward(inputs)
        print(f"Input: {inputs} => Output: {result} (Expected: {y[X.index(inputs)]})")

    plt.tight_layout()
    fig.savefig('perceptron_and_gate_final.png', dpi=300, bbox_inches='tight')
    print("\nFinal plot saved as 'perceptron_and_gate_final.png'")

    plt.show()
