from mlops_practice import config
from mlops_practice.dataset import get_train_val_test_datasets
from mlops_practice.nets import MultiLabelClassifier
from mlops_practice.trainer import ClassifierTrainer


def main():
    train_dataset, val_dataset, _ = get_train_val_test_datasets()
    model = MultiLabelClassifier()
    trainer = ClassifierTrainer(model)
    trainer.fit(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    trainer.plot()
    trainer.save_model(config.MODEL_PATH)


if __name__ == "__main__":
    main()
