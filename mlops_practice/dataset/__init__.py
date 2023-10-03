import torch
from sklearn.datasets import load_digits


TRAIN_RATIO = 0.7


def get_train_val_datasets():
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

    train_size = int(X.shape[0] * TRAIN_RATIO)
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    print(f"Train samples: {X_train.shape[0]}; val samples: {X_val.shape[0]}")

    return (X_train, y_train), (X_val, y_val)
