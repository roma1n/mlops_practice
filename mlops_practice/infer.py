import config
from dataset import get_train_val_test_datasets
from nets import MultiLabelClassifier
from trainer import ClassifierTrainer


def main():
    _, _, test_dataset = get_train_val_test_datasets()
    model = MultiLabelClassifier()
    trainer = ClassifierTrainer(model)
    trainer.load_model(config.MODEL_PATH)
    trainer.process_val(test_dataset)


if __name__ == "__main__":
    main()
