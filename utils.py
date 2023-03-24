import numpy as np
import pandas as pd
from typing import Tuple
from matplotlib import pyplot as plt

def get_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # load the data
    train_df = pd.read_csv('data/mnist_train.csv')
    test_df = pd.read_csv('data/mnist_test.csv')

    X_train = train_df.drop('label', axis=1).values
    y_train = train_df['label'].values

    X_test = test_df.drop('label', axis=1).values
    y_test = test_df['label'].values

    return X_train, X_test, y_train, y_test

def normalize(X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalize the data by subtracting the mean and dividing by the standard deviation of X_train.

    Args:
    - X_train: np.ndarray of shape (n_train, n_features), training data
    - X_test: np.ndarray of shape (n_test, n_features), test data

    Returns:
    - X_train_norm: np.ndarray of shape (n_train, n_features), normalized training data
    - X_test_norm: np.ndarray of shape (n_test, n_features), normalized test data
    """
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0) + 1e-8

    X_train_norm = (X_train - mean) / std
    X_test_norm = (X_test - mean) / std

    return X_train_norm, X_test_norm


def plot_metrics(metrics: list) -> None:
    fig, ax = plt.subplots()

    metric_names = ['accuracy', 'precision', 'recall', 'f1_score']

    for i, metric_name in enumerate(metric_names):
        metric_values = [m[i+1] for m in metrics]
        ax.plot(metric_values, label=metric_name)

    ax.set_xlabel('k')
    ax.set_ylabel('Metric')
    ax.legend()

    plt.savefig('metrics.png')

    plt.show()

