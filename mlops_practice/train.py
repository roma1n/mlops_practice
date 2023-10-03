from dataset import get_train_val_datasets
from nets import MultiLabelClassifier
from trainer import ClassifierTrainer


def main():
    train_dataset, val_dataset = get_train_val_datasets()
    model = MultiLabelClassifier()
    trainer = ClassifierTrainer(model)
    trainer.fit(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    trainer.plot()


if __name__ == "__main__":
    main()
