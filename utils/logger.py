import os
from abc import ABC, abstractmethod
from typing import List, Union

import mlflow
import neptune
import numpy as np
from dotenv import load_dotenv

from configs.experiment_config import ExperimentConfig
from utils.common_functions import convert_lists_and_tuples_to_string


class BaseLogger(ABC):
    """A base experiment logger class."""

    @abstractmethod
    def __init__(self, config):
        """Logs git commit id, dvc hash, environment."""
        pass

    @abstractmethod
    def log_hyperparameters(self, params: dict):
        pass

    @abstractmethod
    def save_metrics(self, *args, **kwargs):
        pass

    @abstractmethod
    def save_plot(self, *args, **kwargs):
        pass

    @abstractmethod
    def stop(self):
        pass


class NeptuneLogger(BaseLogger):
    """A neptune.ai experiment logger class."""

    def __init__(self, config: ExperimentConfig) -> None:
        load_dotenv(config.NEPTUNE_ENV_PATH)
        self.run = neptune.init_run(
            project=config.NEPTUNE_PROJECT,
            api_token=os.environ['NEPTUNE_API_TOKEN'],
            name=config.NEPTUNE_EXPERIMENT_NAME,
            dependencies=config.NEPTUNE_DEPENDENCIES_PATH,
            with_id=config.NEPTUNE_RUN_ID,
        )

    def log_hyperparameters(self, params: dict):
        """Log model hyperparameters logging."""
        self.run['hyperparameters'] = convert_lists_and_tuples_to_string(
            params,
        )

    def save_metrics(
        self,
        type_set: str,
        metric_name: list[str] | str,
        metric_value: list[float] | float | np.floating,
        step: int | None = None,
    ) -> None:
        if isinstance(metric_name, list) ^ isinstance(metric_value, list):
            raise ValueError(
                'metric_name and metric_value must be of the same type!',
            )
        if isinstance(metric_name, list) and isinstance(metric_value, list):
            for p_n, p_v in zip(metric_name, metric_value):
                self.run[f'{type_set}/{p_n}'].append(p_v)
        else:
            self.run[f'{type_set}/{metric_name}'].append(metric_value)

    def save_plot(self, type_set, plot_name, plt_fig):
        self.run[f'{type_set}/{plot_name}'].append(plt_fig)

    def stop(self):
        self.run.stop()


class MLFlowLogger(BaseLogger):
    def __init__(self, config: ExperimentConfig) -> None:
        if config.MLFLOW_TRACKING_URI:
            mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
        self._init_experiment(config)

    def _init_experiment(self, config: ExperimentConfig):
        """Set up experiment configurations to log."""
        mlflow.set_experiment(
            config.MLFLOW_EXPERIMENT_NAME or 'default_experiment',
        )
        if config.MLFLOW_RUN_ID:
            mlflow.start_run(run_id=config.MLFLOW_RUN_ID)
        else:
            mlflow.start_run()
        mlflow.log_artifact(config.MLFLOW_DATASET_VERSION)
        mlflow.log_artifact(config.MLFLOW_DATASET_PREPROCESSING)
        mlflow.log_artifact(config.MLFLOW_DEPENDENCIES_PATH)

    def log_hyperparameters(self, params: dict):
        mlflow.log_params(params)

    def save_metrics(
        self,
        type_set,
        metric_name: Union[List[str], str],
        metric_value: Union[List[float], float],
        step,
    ) -> None:
        if isinstance(metric_name, list) ^ isinstance(metric_value, list):
            raise ValueError(
                'metric_name and metric_value must be of the same type!',
            )
        if isinstance(metric_name, list) and isinstance(metric_value, list):
            for p_n, p_v in zip(metric_name, metric_value):
                mlflow.log_metric(f'{type_set}_{p_n}', p_v, step)
        else:
            mlflow.log_metric(f'{type_set}_{metric_name}', metric_value, step)

    def save_plot(self, type_set, plot_name, plt_fig):
        plot_path = f'temp_plots/{type_set}_{plot_name}.png'
        plt_fig.savefig(plot_path)
        mlflow.log_artifact(plot_path, artifact_path='plots')

    def stop(self):
        mlflow.end_run()
