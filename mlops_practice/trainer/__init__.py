import collections

import torch
from torch import nn
from uniplot import plot

import config


class ClassifierTrainer:
    def __init__(self, model, optim=None, criterion=None):
        self.model = model
        self.optim = optim or torch.optim.Adam(
            self.model.parameters(),
            lr=config.OPTIM_LR,
            weight_decay=config.OPTIM_WEIGHT_DECAY,
        )
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.history = collections.defaultdict(list)

    def process_train(self, dataset, batch_size=config.BATCH_SIZE):
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

    def process_val(self, dataset, batch_size=config.BATCH_SIZE):
        X_val, y_val = dataset
        losses = []
        accuracy = []
        self.model.train(False)
        with torch.no_grad():
            for i in range(0, X_val.shape[0], batch_size):
                X_batch = X_val[i : i + batch_size]
                y_batch = y_val[i : i + batch_size]

                logits = self.model(X_batch)
                accuracy.extend((logits.argmax(axis=1) == y_batch).tolist())
                loss = self.criterion(logits, y_batch)
                losses.append(loss.item())

        loss = sum(losses) / len(losses)
        accuracy = sum(accuracy) / len(accuracy)
        print(f"Loss: {loss:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        self.history["val_loss"].append(loss)
        self.history["val_accuracy"].append(accuracy)

    def fit(self, train_dataset, val_dataset, n_epoch=config.N_EPOCH, batch_size=config.BATCH_SIZE):
        print("\n=== Validate model before training ===")
        self.process_val(val_dataset, batch_size)

        for epoch in range(1, 1 + n_epoch):
            print(f"\n=== Epoch {epoch} of {n_epoch} ===")
            self.process_train(train_dataset, batch_size)
            self.process_val(val_dataset, batch_size)
        print("Model fitted!")

    def plot(self):
        print("\n=== Val accuracy by epoch ===")
        plot(self.history["val_accuracy"])

        print("\n=== Train & val losses by epoch ===")
        plot([self.history["val_loss"], self.history["train_loss"]])

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
