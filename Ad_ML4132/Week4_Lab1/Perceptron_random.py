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

        self.weights[0] += self.lr * error

        for i in range(len(inputs)):
            self.weights[i + 1] += self.lr * error * inputs[i]

if __name__ == "__main__":
    random.seed(20808220)
    np.random.seed(20828220)

    num_samples_per_class = 40
    class_0_center = np.array([0.3, 0.3])
    class_0_points = class_0_center + 0.15 * np.random.randn(num_samples_per_class, 2)

    class_1_center = np.array([0.7, 0.7])
    class_1_points = class_1_center + 0.15 * np.random.randn(num_samples_per_class, 2)

    X = np.vstack((class_0_points, class_1_points)).tolist()
    y = [0] * num_samples_per_class + [1] * num_samples_per_class

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(class_0_points[:, 0], class_0_points[:, 1], color='blue', label='Class 0')
    ax.scatter(class_1_points[:, 0], class_1_points[:, 1], color='red', marker='x', s=50, label='Class 1')

    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title('Perceptron Training on Linearly Separable Data')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True)

    decision_boundary, = ax.plot([], [], 'g-', linewidth=2, label='Decision Boundary')
    ax.legend()

    p = Perceptron(2, 0.1)

    for epoch in range(100):
        total_error = 0
        combined = list(zip(X, y))
        random.shuffle(combined)
        X_shuffled, y_shuffled = zip(*combined)

        for inputs, label in zip(X_shuffled, y_shuffled):
            prediction = p.forward(inputs)
            error = label - prediction
            total_error += abs(error)
            if error != 0:
                p.update(inputs, label, prediction)

        if epoch % 1 == 0:
            w0, w1, w2 = p.weights
            if w2 != 0:
                x1_vals = np.array(ax.get_xlim())
                x2_vals = (-w1 * x1_vals - w0) / w2
                decision_boundary.set_data(x1_vals, x2_vals)

            ax.set_title(f'Epoch: {epoch}, Total Error: {total_error}')
            plt.pause(0.05)

            if total_error == 0:
                print(f"Converged at epoch {epoch}!")
                break

    print("\nTraining finished.")
    print(f"Final weights: {p.weights}")

    w0, w1, w2 = p.weights
    if w2 != 0:
        x1_vals = np.array(ax.get_xlim())
        x2_vals = (-w1 * x1_vals - w0) / w2
        decision_boundary.set_data(x1_vals, x2_vals)
    ax.set_title(f'Training Finished (Final Error: {total_error})')

    fig.savefig('perceptron_final_plot.png', dpi=300, bbox_inches='tight')
    print("\nFinal plot saved as 'perceptron_final_plot.png'")

    plt.show()
