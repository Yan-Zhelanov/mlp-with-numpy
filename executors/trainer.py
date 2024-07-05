import os
import pickle

import numpy as np

from configs.experiment_config import ExperimentConfig
from data_loaders.data_loader import BatchType, DataLoader
from dataset.emotions_dataset import EmotionsDataset
from models.mlp import MLP
from modules.losses.cross_entropy_loss import CrossEntropyLoss
from modules.optimizers.sgd import SGD
from transforms.image_transforms import Sequential
from transforms.target_transforms import OneHotEncoding
from utils.common_functions import set_seed
from utils.enums import SetType
from utils.logger import NeptuneLogger
from utils.metrics import get_balanced_accuracy_score


class Trainer:
    """A class for model training."""

    def __init__(self, config: ExperimentConfig) -> None:
        self._config = config
        self._epoch = 0
        self._best_balanced_accuracy: int | float = 0
        set_seed(self._config.SEED)
        self._logger = NeptuneLogger(self._config)
        self._logger.log_hyperparameters(
            self._config.get_all_hyperparameters(),
        )
        self._prepare_data()
        self._prepare_model()

    def _prepare_data(self):
        """Preparing training and validation data."""
        data_config = self._config.DATA_CONFIG
        batch_size = self._config.TRAIN_BATCH_SIZE
        num_classes = self._config.DATA_NUM_CLASSES
        train_transforms = Sequential(data_config.train_transforms)
        validation_transforms = Sequential(data_config.eval_transforms)
        one_hot_encoding = OneHotEncoding(num_classes)
        self._train_dataset = EmotionsDataset(
            data_config,
            SetType.TRAIN,
            transforms=train_transforms,
            target_transforms=one_hot_encoding,
        )
        self._train_dataloader = DataLoader(
            self._train_dataset,
            batch_size,
            shuffle=True,
            sampler=data_config.sampler_type,
        )
        self._validation_dataset = EmotionsDataset(
            data_config,
            SetType.VALIDATION,
            transforms=validation_transforms,
            target_transforms=one_hot_encoding,
        )
        self._validation_dataloader = DataLoader(
            self._validation_dataset, batch_size=batch_size, shuffle=False,
        )

    def _prepare_model(self):
        """Preparing model, optimizer and loss function."""
        self._model = MLP(self._config.MODEL_CONFIG)
        self._optimizer = SGD(
            self._model, learning_rate=self._config.TRAIN_LEARNING_RATE,
        )
        self._criterion = CrossEntropyLoss()

    def save(self, filepath: str):
        """Save trained model."""
        checkpoint = {
            'model': self._model.get_params(),
        }
        os.makedirs(self._config.CHECKPOINTS_DIR, exist_ok=True)
        filepath = os.path.join(self._config.CHECKPOINTS_DIR, filepath)
        with open(filepath, 'wb') as checkpoint_file:
            pickle.dump(checkpoint, checkpoint_file)

    def load(self, filepath: str):
        """Load trained model."""
        with open(filepath, 'rb') as checkpoint_file:
            checkpoint = pickle.load(checkpoint_file)
        self._model.load_params(checkpoint['model'])

    def make_step(self, batch: BatchType, update_model=False):
        """Perform one step of forward and backward propagation.

        And updating the model weights (if update_model is True).

        Args:
            batch (BatchType): batch with images and labels.
            update_model (bool): if True it is necessary to perform a backward
                pass and update the model weights.

        Returns:
            loss: loss function value
            output: model output (batch_size x num_classes)
        """
        images = batch['images']
        targets = batch['ohe_targets']
        predictions = self._model(images)
        loss = self._criterion(targets, predictions)
        if update_model:
            self._optimizer.zero_grad()
            gradient = self._criterion.compute_backward_gradient(
                targets, predictions,
            )
            self._optimizer.compute_backward_gradient(gradient)
            self._optimizer.step()
        return loss, predictions

    def train_epoch(self):
        """Train the model on training data for one epoch.

        The method goes through all train_dataloader batches and calls the
        self.make_step() method at each step.
        """
        self._model.set_train()
        for batch in self._train_dataloader:
            loss, output = self.make_step(batch, update_model=True)
            balanced_accuracy = get_balanced_accuracy_score(
                batch['targets'], output.argmax(axis=-1),
            )
            self._logger.save_metrics(
                SetType.TRAIN.name.lower(), 'loss', loss, step=self._epoch,
            )
            self._logger.save_metrics(
                SetType.TRAIN.name.lower(),
                'balanced_accuracy',
                balanced_accuracy,
                step=self._epoch,
            )

    def fit(self):
        """The main model training loop."""
        for epoch in range(self._config.NUM_EPOCHS):
            self._epoch = epoch
            self.train_epoch()
            self.evaluate(epoch, self._train_dataloader, SetType.TRAIN)
            balanced_accuracy = self.evaluate(
                epoch, self._validation_dataloader, SetType.VALIDATION,
            )
            self._save_if_better(balanced_accuracy)

    def _save_if_better(self, balanced_accuracy: float) -> None:
        if balanced_accuracy > self._best_balanced_accuracy:
            self._best_balanced_accuracy = balanced_accuracy
            self.save(f'model_{balanced_accuracy:.3f}.pickle')

    def evaluate(
        self, epoch: int, dataloader: DataLoader, set_type: SetType,
    ) -> np.floating:
        """Evaluate the model.

        The method is used to make the model performance evaluation on
        training/validation/test data.

        Args:
            epoch: current training epoch
            dataloader: dataloader for the chosen set type
            set_type: set type chosen to evaluate

        Returns:
            balanced_accuracy: the balanced accuracy value.
        """
        self._model.set_eval()
        total_loss = []
        all_labels = []
        all_outputs = []
        for batch in dataloader:
            loss, output = self.make_step(batch, update_model=False)
            total_loss.append(loss)
            all_labels.append(batch['targets'])
            all_outputs.append(output)
        total_loss = np.mean(total_loss)
        all_predictions = np.concatenate(all_outputs).argmax(axis=-1)
        balanced_accuracy = get_balanced_accuracy_score(
            np.concatenate(all_labels), all_predictions,
        )
        self._logger.save_metrics(
            set_type.name.lower(), 'loss', total_loss, step=epoch,
        )
        self._logger.save_metrics(
            set_type.name.lower(),
            'balanced_accuracy',
            balanced_accuracy,
            step=epoch,
        )
        return balanced_accuracy

    def batch_overfit(self):
        """One batch overfitting.

        This feature can be useful for debugging and evaluating your model's
        ability to learn and update its weights.
        """
        self._model.set_train()
        batch = next(iter(self._train_dataloader))
        for _ in range(self._config.OVERFIT_NUM_ITERATIONS):
            loss_value, output = self.make_step(batch, update_model=True)
            balanced_accuracy = get_balanced_accuracy_score(
                batch['targets'], output.argmax(-1),
            )
            self._logger.save_metrics(
                SetType.TRAIN.name.lower(), 'loss', loss_value,
            )
            self._logger.save_metrics(
                SetType.TRAIN.name.lower(),
                'balanced_accuracy',
                balanced_accuracy,
            )
