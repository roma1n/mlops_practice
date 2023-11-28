import subprocess

import fire
import hydra
import torch
from hydra.core.config_store import ConfigStore

from mlops_practice.config import Params
from mlops_practice.dataset import MNISTDataModule
from mlops_practice.nets import MultiLabelClassifier
from mlops_practice.trainer import ClassifierTrainer


cs = ConfigStore.instance()
cs.store(name="params", node=Params)


class MLOpsPractice(object):
    def __init__(self):
        print("Running dvc pull")
        subprocess.run(["dvc", "pull", "-r", "dvcreader"])

        with hydra.initialize(config_path="../conf", version_base="1.3"):
            self.params: Params = hydra.compose(config_name="config")
        torch.manual_seed(self.params.model.random_seed)

        self.datamodule = MNISTDataModule(
            train_val_dataset_path=self.params.path.fashion_mnist_train,
            test_dataset_path=self.params.path.fashion_mnist_test,
            train_ratio=self.params.dataset.train_ratio,
            batch_size=self.params.model.batch_size,
            num_workers=self.params.model.num_workers,
        )
        self.model = MultiLabelClassifier()
        self.trainer = ClassifierTrainer(
            model=self.model,
            n_epoch=self.params.model.n_epoch,
            lr=self.params.model.optim.lr,
            weight_decay=self.params.model.optim.weight_decay,
            tracking_uri=self.params.logging.mlflow_tracking_uri,
            params_for_logging=vars(self.params)["_content"],
        )

    def train(self):
        self.trainer.fit(self.datamodule)
        self.trainer.save_model(self.params.path.model)

    def infer(self):
        self.trainer.load_model(self.params.path.model)
        self.trainer.predict(self.datamodule.test_dataloader(), output_csv=self.params.path.result)

    def train_infer(self):
        self.train()
        self.infer()


def main():
    fire.Fire(MLOpsPractice())


if __name__ == "__main__":
    main()
