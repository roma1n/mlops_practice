from typing import Any, Tuple

import pandas as pd
import pytorch_lightning as pl
import torch
from torch import nn


class TrainerModule(pl.LightningModule):
    "Trains model for multi-class classification task"

    def __init__(
        self,
        model: nn.Module,
        lr: float = None,
        weight_decay: float = None,
        optim: torch.optim.Optimizer = None,
        criterion: nn.Module = None,
    ):
        super().__init__()

        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.optim = optim

        self.criterion = criterion or nn.CrossEntropyLoss()

    def forward(self, *args: Any, **kwargs: Any):
        return self.model.forward(*args, **kwargs)

    def configure_optimizers(self):
        optim = self.optim or torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        return optim

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        "Trains model on batch"
        X_batch, y_batch = batch
        self.model.train(True)
        logits = self.model(X_batch)
        loss = self.criterion(logits, y_batch)
        self.log("train_loss", loss, on_step=False, on_epoch=True)

        predictions = logits.detach().cpu().argmax(axis=1)
        self.log(
            "train_accuracy",
            (predictions == y_batch).float().mean(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        "Validates model on batch"
        X_batch, y_batch = batch
        self.model.train(False)
        with torch.no_grad():
            logits = self.model(X_batch)
            loss = self.criterion(logits, y_batch)
            self.log("val_loss", loss, on_step=False, on_epoch=True)

            predictions = logits.detach().cpu().argmax(axis=1)
            self.log(
                "val_accuracy",
                (predictions == y_batch).float().mean(),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

            return loss

    def predict_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        "Infers model on batch"
        X_batch, y_batch = batch
        self.model.train(False)
        with torch.no_grad():
            logits = self.model(X_batch)
            predictions = logits.detach().cpu().argmax(axis=1)

            return {
                "logits": logits.detach().cpu(),
                "predicted_label": predictions,
                "label": y_batch,
            }


class ClassifierTrainer:
    def __init__(
        self,
        model: nn.Module,
        n_epoch: int,
        lr: float = None,
        weight_decay: float = None,
        optim: torch.optim.Optimizer = None,
        criterion: nn.Module = None,
    ):
        self.model = model
        self.n_epoch = n_epoch
        self.trainer_module = TrainerModule(
            model=model,
            lr=lr,
            weight_decay=weight_decay,
            optim=optim,
            criterion=criterion,
        )
        self.pl_trainer = pl.Trainer(
            precision="bf16-mixed",
            max_epochs=self.n_epoch,
            log_every_n_steps=1,
            logger=[
                pl.loggers.CSVLogger("logs"),
                # pl.loggers.TensorBoardLogger("tb_logs"),
            ],
        )

    def fit(self, datamodule: pl.LightningDataModule):
        self.pl_trainer.fit(self.trainer_module, datamodule)

    def save_model(self, filepath: str):
        self.pl_trainer.save_checkpoint(filepath, weights_only=True)

    def load_model(self, filepath: str):
        self.trainer_module = TrainerModule.load_from_checkpoint(filepath, model=self.model)

    def predict(self, dataloader: torch.utils.data.DataLoader, output_csv=None):
        prediction = self.pl_trainer.predict(model=self.trainer_module, dataloaders=[dataloader])[0]
        df_dict = {
            f"score_{i}": prediction["logits"][:, i].float().numpy()
            for i in range(prediction["logits"].shape[1])
        }
        df_dict.update(
            {
                "predicted_label": prediction["predicted_label"],
                "label": prediction["label"],
            }
        )
        df = pd.DataFrame(df_dict)

        if output_csv:
            df.reset_index().round(4).to_csv(output_csv, index=False)
        else:
            return df
