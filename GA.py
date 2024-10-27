import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

global_path = "D:\CI\As3\Data\\train7\\"


def load_data(file_path="wdbc.txt"):
    """
    Load and preprocess data
    """
    df = pd.read_csv(file_path, header=None)
    column_names = ["ID", "Diagnosis", *[f"Feature{i}" for i in range(1, 31)]]
    df.columns = column_names
    df["Diagnosis"] = df["Diagnosis"].map({"M": 1, "B": 0})

    x = df.iloc[:, 2:].values
    y = df["Diagnosis"].values
    x = (x - x.mean(axis=0)) / x.std(axis=0)
    return x, y


class MLP:
    """
    Multilayer Perceptron (MLP) with one hidden layer
    """

    def __init__(self, input_size, hidden_size, output_size):
        self.weights1 = np.random.randn(input_size, hidden_size)
        self.bias1 = np.random.randn(hidden_size)
        self.weights2 = np.random.randn(hidden_size, output_size)
        self.bias2 = np.random.randn(output_size)

    def forward(self, x):
        self.hidden = np.tanh(np.dot(x, self.weights1) + self.bias1)
        output = 1 / (1 + np.exp(-np.dot(self.hidden, self.weights2) - self.bias2))
        return output


class GeneticAlgorithm:
    """
    Genetic Algorithm for optimizing MLP weights
    """

    def __init__(self, population_size, input_size, hidden_size, output_size):
        self.population_size = population_size
        self.population = [
            MLP(input_size, hidden_size, output_size) for _ in range(population_size)
        ]

    def evaluate(self, X, y):
        fitness = []
        for model in self.population:
            predictions = np.round([model.forward(x) for x in X]).flatten()
            accuracy = np.mean(predictions == y)
            fitness.append(accuracy)
        return fitness

    def selection(self, fitness):
        idx = np.argsort(fitness)[-self.population_size // 2 :][::-1]
        self.population = [self.population[i] for i in idx]

    def crossover(self):
        new_population = []
        for _ in range(len(self.population) * 2):
            parent1, parent2 = np.random.choice(self.population, 2)
            child = MLP(
                self.population[0].weights1.shape[0],
                self.population[0].weights1.shape[1],
                1,
            )
            child.weights1 = (parent1.weights1 + parent2.weights1) / 2
            child.weights2 = (parent1.weights2 + parent2.weights2) / 2
            child.bias1 = (parent1.bias1 + parent2.bias1) / 2
            child.bias2 = (parent1.bias2 + parent2.bias2) / 2
            new_population.append(child)
        self.population = new_population[: self.population_size]

    def mutate(self, mutation_rate=0.01):
        for model in self.population:
            if np.random.rand() < mutation_rate:
                model.weights1 += np.random.randn(*model.weights1.shape) * mutation_rate
                model.weights2 += np.random.randn(*model.weights2.shape) * mutation_rate
                model.bias1 += np.random.randn(*model.bias1.shape) * mutation_rate
                model.bias2 += np.random.randn(*model.bias2.shape) * mutation_rate


def create_confusion_matrix(y_true, y_pred):
    """
    Create a confusion matrix
    """
    matrix = np.zeros((2, 2), dtype=int)
    for true, pred in zip(y_true, y_pred):
        matrix[true, pred] += 1
    return matrix


def plot_confusion_matrix(cm, labels, path):
    """
    Plot the confusion matrix and save it
    """
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)

    thresh = cm.max() / 2.0
    for i, j in np.ndindex(cm.shape):
        plt.text(
            j,
            i,
            cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def train_mlp_with_ga(
    x,
    y,
    hidden_size=20,
    population_size=20,
    generations=20,
    k_folds=10,
    save_path="accuracy_plot.png",
):
    """
    Main training process with cross-validation
    """
    fold_size = len(x) // k_folds
    accuracies = []
    generation_accuracies = []

    for fold in range(k_folds):
        start, end = fold * fold_size, (fold + 1) * fold_size
        x_train = np.concatenate([x[:start], x[end:]], axis=0)
        y_train = np.concatenate([y[:start], y[end:]], axis=0)
        x_test, y_test = x[start:end], y[start:end]

        ga = GeneticAlgorithm(population_size, x.shape[1], hidden_size, 1)
        fold_generation_accuracies = []

        for generation in range(generations):
            fitness = ga.evaluate(x_train, y_train)
            ga.selection(fitness)
            ga.crossover()
            ga.mutate()

            best_accuracy = max(fitness)
            fold_generation_accuracies.append(best_accuracy)
            print(
                f"Fold {fold + 1}, Generation {generation + 1}, Fitness: {best_accuracy}"
            )

        best_model = ga.population[np.argmax(fitness)]
        predictions = np.round([best_model.forward(x) for x in x_test]).flatten()
        accuracy = np.mean(predictions == y_test)
        accuracies.append(accuracy)

        cm = create_confusion_matrix(y_test, predictions.astype(int))
        plot_confusion_matrix(
            cm,
            labels=["Benign", "Malignant"],
            path=f"{global_path}confusion_matrix_fold_{fold + 1}.png",
        )

        generation_accuracies.append(fold_generation_accuracies)

    print("Average Cross-Validation Accuracy:", np.mean(accuracies))
    save_path = f"{global_path}accuracy_plot_{hidden_size}_{population_size}_{generations}_{np.mean(accuracies)}.png"

    plt.figure(figsize=(10, 6))
    for fold_accuracy in generation_accuracies:
        plt.plot(fold_accuracy, marker="o")
    plt.title("Accuracy per Generation for Each Fold")
    plt.xlabel("Generation")
    plt.ylabel("Accuracy")
    plt.grid()
    plt.legend([f"Fold {i + 1}" for i in range(k_folds)], loc="best")
    plt.savefig(save_path)
    plt.close()


x, y = load_data()
train_mlp_with_ga(x, y)
