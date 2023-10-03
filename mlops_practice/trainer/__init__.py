import collections

import torch
from torch import nn
from uniplot import plot


BATCH_SIZE = 32
N_EPOCH = 25


class ClassifierTrainer:
    def __init__(self, model, optim=None, criterion=None):
        self.model = model
        self.optim = optim or torch.optim.Adam(self.model.parameters(), lr=3e-2, weight_decay=1e-2)
        self.criterion = criterion or nn.CrossEntropyLoss()
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
