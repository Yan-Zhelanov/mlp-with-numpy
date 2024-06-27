import os
from typing import Any

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
    NEPTUNE_PROJECT: str | None = 'somethingintheway/Emotions-MLP'
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
    DATA_CONFIG = DataConfig()
    DATA_NUM_CLASSES = 7

    # Model parameters
    MODEL_CONFIG = ModelConfig()

    def get_all_hyperparameters(self) -> dict[str, Any]:
        """Get all hyperparameters for the model.

        Returns:
            dict[str, Any]: dictionary with all hyperparameters.
        """
        return {
            'seed': self.SEED,
            'num_epochs': self.NUM_EPOCHS,
            'train_batch_size': self.TRAIN_BATCH_SIZE,
            'train_learning_rate': self.TRAIN_LEARNING_RATE,
            'overfit_num_iterations': self.OVERFIT_NUM_ITERATIONS,
            'neptune_env_path': self.NEPTUNE_ENV_PATH,
            'neptune_project': self.NEPTUNE_PROJECT,
            'neptune_experiment_name': self.NEPTUNE_EXPERIMENT_NAME,
            'neptune_run_id': self.NEPTUNE_RUN_ID,
            'neptune_dependencies_path': self.NEPTUNE_DEPENDENCIES_PATH,
            'mlflow_tracking_uri': self.MLFLOW_TRACKING_URI,
            'mlflow_experiment_name': self.MLFLOW_EXPERIMENT_NAME,
            'mlflow_run_id': self.MLFLOW_RUN_ID,
            'mlflow_dataset_version': self.MLFLOW_DATASET_VERSION,
            'data_config': self.DATA_CONFIG.get_all_hyperparameters(),
            'model_config': self.MODEL_CONFIG.get_all_hyperparameters(),
        }
