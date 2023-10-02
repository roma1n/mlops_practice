import collections

import torch
from sklearn.datasets import load_digits
from torch import nn
from uniplot import plot


TRAIN_RATIO = 0.7
BATCH_SIZE = 32
N_EPOCH = 25


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class ConvPoolBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)

        return x


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = ConvPoolBlock(1, 2)
        self.conv2 = ConvPoolBlock(2, 2)
        self.flatten = nn.Flatten()
        self.head = nn.Linear(8, 10)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.head(x)
        x = self.softmax(x)

        return x


class ClassifierTrainer:
    def __init__(self, model, optim, criterion):
        self.model = model
        self.optim = optim
        self.criterion = criterion
        self.history = collections.defaultdict(list)

    def process_train(self, dataset, batch_size):
        X_train, y_train = dataset
        losses = []
        self.model.train(True)
        for i in range(0, X_train.shape[0], batch_size):
            X_batch = X_train[i : i + batch_size]
            y_batch = y_train[i : i + batch_size]

            logits = self.model(X_batch)
            loss = self.criterion(logits, y_batch)
            loss.backward()
            self.optim.step()
            self.optim.zero_grad()
            losses.append(loss.item())

        loss = sum(losses) / len(losses)
        print(f"Train loss: {loss:.4f}")
        self.history["train_loss"].append(loss)

    def process_val(self, dataset, batch_size):
        X_val, y_val = dataset
        losses = []
        accuracy = []
        self.model.train(False)
        with torch.no_grad():
            for i in range(0, X_val.shape[0], BATCH_SIZE):
                X_batch = X_val[i : i + BATCH_SIZE]
                y_batch = y_val[i : i + BATCH_SIZE]

                logits = self.model(X_batch)
                accuracy.extend((logits.argmax(axis=1) == y_batch).tolist())
                loss = self.criterion(logits, y_batch)
                losses.append(loss.item())

        loss = sum(losses) / len(losses)
        accuracy = sum(accuracy) / len(accuracy)
        print(f"Validation loss: {loss:.4f}")
        print(f"Validation accuracy: {accuracy:.4f}")
        self.history["val_loss"].append(loss)
        self.history["val_accuracy"].append(accuracy)

    def fit(self, train_dataset, val_dataset, n_epoch=N_EPOCH, batch_size=BATCH_SIZE):
        print("\n=== Validate model before training ===")
        self.process_val(val_dataset, batch_size)

        for epoch in range(1, 1 + N_EPOCH):
            print(f"\n=== Epoch {epoch} of {N_EPOCH} ===")
            self.process_train(train_dataset, batch_size)
            self.process_val(val_dataset, batch_size)
        print("Model fitted!")

    def plot(self):
        print("\n=== Val accuracy by epoch ===")
        plot(self.history["val_accuracy"])

        print("\n=== Train & val losses by epoch ===")
        plot([self.history["val_loss"], self.history["train_loss"]])


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


def main():
    train_dataset, val_dataset = get_train_val_datasets()
    model = Model()
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=3e-2, weight_decay=1e-2)
    trainer = ClassifierTrainer(model, optim, criterion)
    trainer.fit(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    trainer.plot()


if __name__ == "__main__":
    main()
