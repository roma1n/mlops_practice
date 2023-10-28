import pandas as pd
import pytorch_lightning as pl
import torch


class MNISTDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_val_dataset_path: str,
        test_dataset_path: str,
        train_ratio: float,
        batch_size: int,
        num_workers: int,
    ):
        super().__init__()

        self.train_val_dataset_path = train_val_dataset_path
        self.test_dataset_path = test_dataset_path
        self.train_ratio = train_ratio
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_dataset, self.val_dataset = self.get_train_val_datasets()
        self.test_dataset = self.get_test_dataset()

    def get_train_val_datasets(self):
        "Loads MNIST dataset and separates it into train-val splits"
        print("Loading train-val dataset")
        df = pd.read_csv(self.train_val_dataset_path)
        X, y = df.drop(columns=["label"]).to_numpy(), df["label"].to_numpy()
        X = torch.tensor(X).reshape(X.shape[0], 1, 28, 28).float()
        y = torch.tensor(y)

        train_bound = int(X.shape[0] * self.train_ratio)
        X_train, X_val = X[:train_bound], X[train_bound:]
        y_train, y_val = y[:train_bound], y[train_bound:]
        print(f"Train samples: {X_train.shape[0]}; val samples: {X_val.shape[0]}")

        return (
            torch.utils.data.TensorDataset(X_train, y_train),
            torch.utils.data.TensorDataset(X_val, y_val),
        )

    def get_test_dataset(self):
        "Loads test split of MNIST dataset"
        print("Loading test dataset")
        df = pd.read_csv(self.test_dataset_path)
        X, y = df.drop(columns=["label"]).to_numpy(), df["label"].to_numpy()
        X = torch.tensor(X).reshape(X.shape[0], 1, 28, 28).float()
        y = torch.tensor(y)

        print(f"Test samples: {X.shape[0]}")

        return torch.utils.data.TensorDataset(X, y)

    def make_data_loader(self, dataset: torch.utils.data.Dataset, shuffle=False):
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def train_dataloader(self):
        return self.make_data_loader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self.make_data_loader(self.val_dataset)

    def test_dataloader(self):
        return self.make_data_loader(self.test_dataset)
