import os
import pickle
from configs.experiment_config import ExperimentConfig

import numpy as np

from data_loaders.data_loader import DataLoader
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

    def make_step(self, batch: dict, update_model=False):
        """This method performs one step, including forward pass, calculation of the target function, backward
        pass and updating the model weights (if update_model is True).

        Args:
            batch: batch data
            update_model (bool): if True it is necessary to perform a backward pass and update the model weights

        Returns:
            loss: loss function value
            output: model output (batch_size x num_classes)
        """
        # TODO: Implement one step of forward and backward propagation:
        #       1. Get images and OHE targets from batch
        #       2. Get model output
        #       3. Compute loss
        #       4. If update_model is True, make backward propagation:
        #           - Set the gradient of the layer parameters to zero using optimizer.zero_grad()
        #           - Compute the gradient of the loss function using criterion.backward()
        #           - Make backward propagation through all the layers using optimizer.backward()
        #               and computed loss function's gradient
        #           - Update the model parameters using optimizer.step()
        raise NotImplementedError

    def train_epoch(self):
        """Train the model on training data for one epoch.

        The method goes through all train_dataloader batches and calls the
        self.make_step() method at each step.
        """
        # TODO: Implement the training process for one epoch. For all batches in train_dataloader do:
        #       1. Make training step by calling make_step() method with update_model=True
        #       2. Get the model predictions from the outputs with argmax
        #       3. Compute metrics
        #       4. Log the value of the loss functions and metrics with logger
        self._model.train()
        raise NotImplementedError

    def fit(self):
        """The main model training loop."""
        # TODO: Implement the main model training loop iterating over the epochs. At each epoch:
        #       1. The model is first trained on the training data using the self.train_epoch() method
        #       2. The model performance is then evaluated on the validation data with self.evaluate() method
        #       3. (Optionally) If performance metrics on the validation data exceeds the best values achieved,
        #               model parameters should be saved with save() method
        raise NotImplementedError

    def evaluate(self, epoch: int, dataloader: DataLoader, set_type: SetType):
        """Evaluation.

        The method is used to make the model performance evaluation on
        training/validation/test data.

        Args:
            epoch: current training epoch
            dataloader: dataloader for the chosen set type
            set_type: set type chosen to evaluate
        """
        # TODO: To implement the model performance evaluation for each batch in the given dataloader do:
        #       1. Make model forward pass using self.make_step(batch, update_model=False)
        #       2. Add loss value to total_loss list
        #       3. Add model output to all_outputs list
        #       4. Add batch labels to all_labels list
        #    Get total loss and metrics values, log them with logger.
        self._model.eval()
        total_loss = []
        all_outputs, all_labels = [], []
        raise NotImplementedError

    def batch_overfit(self):
        """One batch overfitting.

        This feature can be useful for debugging and evaluating your model's
        ability to learn and update its weights.
        """
        self._model.train()
        batch = next(iter(self._train_dataloader))
        for _ in range(self._config.OVERFIT_NUM_ITERATIONS):
            loss_value, output = self.make_step(batch, update_model=True)
            balanced_acc = get_balanced_accuracy_score(
                batch['target'], output.argmax(-1),
            )
            self._logger.save_metrics(
                SetType.TRAIN.name.lower(), 'loss', loss_value,
            )
            self._logger.save_metrics(
                SetType.TRAIN.name.lower(), 'balanced_acc', balanced_acc,
            )
