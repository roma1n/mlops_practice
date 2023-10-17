import pandas as pd
import torch
from sklearn.datasets import load_digits

from mlops_practice import config


def get_train_val_datasets():
    "Loads Fashion MNIST dataset and separates it into train-val splits"
    print("Loading train-val dataset")
    df = pd.read_csv(config.FASHION_MNIST_TRAIN_VAL_PATH)
    X, y = df.drop(columns=["label"]).to_numpy(), df["label"].to_numpy()
    X = torch.tensor(X).reshape(X.shape[0], 1, 28, 28).float()
    y = torch.tensor(y)

    train_bound = int(X.shape[0] * config.TRAIN_RATIO)
    X_train, X_val = X[:train_bound], X[train_bound:]
    y_train, y_val = y[:train_bound], y[train_bound:]
    print(f"Train samples: {X_train.shape[0]}; val samples: {X_val.shape[0]}")

    return (X_train, y_train), (X_val, y_val)


def get_test_dataset():
    "Loads test split of Fashion MNIST dataset"
    print("Loading test dataset")
    df = pd.read_csv(config.FASHION_MNIST_TEST_PATH)
    X, y = df.drop(columns=["label"]).to_numpy(), df["label"].to_numpy()
    X = torch.tensor(X).reshape(X.shape[0], 1, 28, 28).float()
    y = torch.tensor(y)

    print(f"Test samples: {X.shape[0]}")

    return X, y
