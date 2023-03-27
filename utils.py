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

    X_train_norm = X_train / 255.0 * 2 - 1
    X_test_norm = X_test / 255.0 * 2 - 1

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