import os

from configs.data_config import DataConfig
from configs.mlp_config import ModelConfig

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))


class ExperimentConfig:
    SEED = 0
    NUM_EPOCHS = 10

    # Train parameters
    TRAIN_BATCH_SIZE = 64
    TRAIN_LEARNING_RATE = 5e-2

    # Overfit parameters
    OVERFIT_NUM_ITERATIONS = 500

    # Neptune parameters
    NEPTUNE_ENV_PATH = os.path.join(ROOT_DIR, '.env')
    NEPTUNE_PROJECT: str | None = None
    NEPTUNE_EXPERIMENT_NAME: str | None = None
    NEPTUNE_RUN_ID: str | None = None
    NEPTUNE_DEPENDENCIES_PATH = os.path.join(ROOT_DIR, 'requirements.txt')

    # MLFlow parameters
    MLFLOW_TRACKING_URI: str | None = None
    MLFLOW_EXPERIMENT_NAME: str | None = None
    MLFLOW_RUN_ID: int | None = None
    MLFLOW_DATASET_VERSION: str | None = None
    MLFLOW_DATASET_PREPROCESSING: str | None = None
    MLFLOW_DEPENDENCIES_PATH: str | None = None

    # Checkpoints parameters
    CHECKPOINTS_DIR = os.path.join(
        ROOT_DIR,
        'checkpoints',
        NEPTUNE_EXPERIMENT_NAME if NEPTUNE_EXPERIMENT_NAME else '',
    )

    # Data parameters
    DATA_CONFIG = DataConfig
    DATA_NUM_CLASSES = 7

    # Model parameters
    MODEL_CONFIG = ModelConfig
