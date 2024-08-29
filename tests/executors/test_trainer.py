import numpy as np
import pytest
from pytest_mock import MockerFixture

from data_loaders.data_loader import BatchType
from executors.trainer import Trainer
from utils.enums import WeightsInitType


@pytest.fixture
def trainer(mocker: MockerFixture):
    """Fixture to set up the Trainer object with mocks."""
    config = mocker.MagicMock()
    config.SEED = 42
    config.TRAIN_BATCH_SIZE = 32
    config.DATA_NUM_CLASSES = 10
    config.MODEL_CONFIG = mocker.MagicMock()
    config.MODEL_CONFIG.INIT_TYPE = WeightsInitType.NORMAL
    config.TRAIN_LEARNING_RATE = 0.01
    mocker.patch('executors.trainer.NeptuneLogger')
    mocker.patch('executors.trainer.EmotionsDataset')
    mocker.patch('executors.trainer.DataLoader')
    trainer = Trainer(config)
    trainer._model = mocker.MagicMock()
    trainer._optimizer = mocker.MagicMock()
    trainer._criterion = mocker.MagicMock()
    return trainer


def test_make_step_forward_pass(trainer: Trainer):
    """Test forward pass of make_step without model update."""
    batch: BatchType = {
        'images': np.random.rand(32, 3, 224, 224),
        'targets': np.random.randint(0, 10, size=(32,)),
        'ohe_targets': np.random.rand(32, 10),
    }
    trainer._model.return_value = np.random.rand(32, 10)
    trainer._criterion.return_value = 1.0

    loss, predictions = trainer.make_step(batch, update_model=False)

    trainer._model.assert_called_once_with(batch['images'])
    trainer._criterion.assert_called_once_with(
        batch['ohe_targets'], predictions,
    )
    assert loss == pytest.approx(1.0)
    assert predictions.shape == (32, 10)


def test_make_step_backward_pass(trainer: Trainer):
    """Test forward and backward pass of make_step with model update."""
    batch: BatchType = {
        'images': np.random.rand(32, 3, 224, 224),
        'targets': np.random.randint(0, 10, size=(32,)),
        'ohe_targets': np.random.rand(32, 10),
    }
    trainer._model.return_value = np.random.rand(32, 10)
    trainer._criterion.return_value = 1.0
    trainer._criterion.compute_backward_gradient.return_value = np.random.rand(
        32, 10,
    )

    loss, predictions = trainer.make_step(batch, update_model=True)

    trainer._model.assert_called_once_with(batch['images'])
    trainer._criterion.assert_called_once_with(
        batch['ohe_targets'], predictions,
    )
    trainer._optimizer.zero_grad.assert_called_once()
    trainer._criterion.compute_backward_gradient.assert_called_once_with(
        batch['ohe_targets'], predictions,
    )
    trainer._optimizer.compute_backward_gradient.assert_called_once_with(
        trainer._criterion.compute_backward_gradient.return_value,
    )
    trainer._optimizer.step.assert_called_once()
    assert loss == pytest.approx(1.0)
    assert predictions.shape == (32, 10)
