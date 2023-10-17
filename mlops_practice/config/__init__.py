from pathlib import Path


DATA_PATH = Path("data")
FASHION_MNIST_PATH = DATA_PATH.joinpath("fashion_mnist")
FASHION_MNIST_TRAIN_VAL_PATH = FASHION_MNIST_PATH.joinpath("train.csv")
FASHION_MNIST_TEST_PATH = FASHION_MNIST_PATH.joinpath("test.csv")
MODEL_PATH = Path("model.bin")
RESULT_PATH = Path("result.csv")

RANDOM_SEED = 42
BATCH_SIZE = 1024
N_EPOCH = 5
OPTIM_LR = 1e-3
OPTIM_WEIGHT_DECAY = 1e-5

TRAIN_RATIO = 0.7
