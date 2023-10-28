from dataclasses import dataclass


@dataclass
class PathParams:
    fashion_mnist_train: str
    fashion_mnist_test: str
    model: str
    result: str


@dataclass
class OptimParams:
    lr: float
    weight_decay: float


@dataclass
class ModelParams:
    random_seed: int
    batch_size: int
    num_workers: int
    n_epoch: int
    optim: OptimParams


@dataclass
class DatasetParams:
    train_ratio: float


@dataclass
class Params:
    path: PathParams
    model: ModelParams
    dataset: DatasetParams
