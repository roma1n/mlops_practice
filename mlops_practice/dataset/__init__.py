import torch
from sklearn.datasets import load_digits

from mlops_practice import config


def get_train_val_test_datasets():
    "Loads Digits dataset and separates it into train-val-test splits"
    print("Loading dataset")
    X, y = load_digits(return_X_y=True)
    X = torch.tensor(X).reshape(X.shape[0], 1, 8, 8).float()
    y = torch.tensor(y)

    print("Sample from dataset")
    print(
        "\n".join(
            map(
                lambda line: "".join(map(lambda elem: str(elem.item()), line)), (X[0, 0] > 0).long()
            )
        )
    )
    print(f"class: {y[0].item()}\n")

    train_bound = int(X.shape[0] * config.TRAIN_RATIO)
    val_bound = int(X.shape[0] * (config.TRAIN_RATIO + config.VAL_RATIO))
    X_train, X_val, X_test = X[:train_bound], X[train_bound:val_bound], X[val_bound:]
    y_train, y_val, y_test = y[:train_bound], y[train_bound:val_bound], y[val_bound:]
    print(
        f"Train samples: {X_train.shape[0]}; val samples: {X_val.shape[0]}; test samples: {X_test.shape[0]}"
    )

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)
