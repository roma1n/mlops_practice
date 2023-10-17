import fire

from mlops_practice import config
from mlops_practice.dataset import get_test_dataset, get_train_val_datasets
from mlops_practice.nets import MultiLabelClassifier
from mlops_practice.trainer import ClassifierTrainer


class MLOpsPractice(object):
    def train(self, plot: bool = False):
        train_dataset, val_dataset = get_train_val_datasets()
        model = MultiLabelClassifier()
        trainer = ClassifierTrainer(model)
        trainer.fit(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
        )

        if plot:
            trainer.plot()
        trainer.save_model(config.MODEL_PATH)

    def infer(self):
        test_dataset = get_test_dataset()
        model = MultiLabelClassifier()
        trainer = ClassifierTrainer(model)
        trainer.load_model(config.MODEL_PATH)
        trainer.process_val(test_dataset, output_csv=config.RESULT_PATH)

    def train_infer(self):
        self.train()
        self.infer()


def main():
    fire.Fire(MLOpsPractice)


if __name__ == "__main__":
    main()
