import fire
import hydra
import torch
from hydra.core.config_store import ConfigStore

from mlops_practice.config import Params
from mlops_practice.dataset import get_test_dataset, get_train_val_datasets
from mlops_practice.nets import MultiLabelClassifier
from mlops_practice.trainer import ClassifierTrainer


cs = ConfigStore.instance()
cs.store(name="params", node=Params)


class MLOpsPractice(object):
    def __init__(self):
        with hydra.initialize(config_path="../conf", version_base="1.3"):
            self.params: Params = hydra.compose(config_name="config")
        torch.manual_seed(self.params.model.random_seed)

    def train(self, plot: bool = False):
        train_dataset, val_dataset = get_train_val_datasets(
            dataset_path=self.params.path.fashion_mnist_train,
            train_ratio=self.params.dataset.train_ratio,
        )
        model = MultiLabelClassifier()
        trainer = ClassifierTrainer(
            model=model,
            batch_size=self.params.model.batch_size,
            n_epoch=self.params.model.n_epoch,
            lr=self.params.model.optim.lr,
            weight_decay=self.params.model.optim.weight_decay,
        )
        trainer.fit(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
        )

        if plot:
            trainer.plot()
        trainer.save_model(self.params.path.model)

    def infer(self):
        test_dataset = get_test_dataset(
            dataset_path=self.params.path.fashion_mnist_test,
        )
        model = MultiLabelClassifier()
        trainer = ClassifierTrainer(
            model=model,
            batch_size=self.params.model.batch_size,
            n_epoch=self.params.model.n_epoch,
            lr=self.params.model.optim.lr,
            weight_decay=self.params.model.optim.weight_decay,
        )
        trainer.load_model(self.params.path.model)
        trainer.process_val(test_dataset, output_csv=self.params.path.result)

    def train_infer(self):
        self.train()
        self.infer()


def main():
    fire.Fire(MLOpsPractice())


if __name__ == "__main__":
    main()
