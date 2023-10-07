from mlops_practice import config
from mlops_practice.dataset import get_train_val_test_datasets
from mlops_practice.nets import MultiLabelClassifier
from mlops_practice.trainer import ClassifierTrainer


def main():
    _, _, test_dataset = get_train_val_test_datasets()
    model = MultiLabelClassifier()
    trainer = ClassifierTrainer(model)
    trainer.load_model(config.MODEL_PATH)
    trainer.process_val(test_dataset, output_csv=config.RESULT_PATH)


if __name__ == "__main__":
    main()
