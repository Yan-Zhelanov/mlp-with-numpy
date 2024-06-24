from typing import Self, cast

import numpy as np
from numpy import typing as npt

from models.mlp import MLP
from modules.layers.base import BaseLayer
from modules.optimizers.sgd import SGD


class MockLayer(BaseLayer):
    def __init__(self: Self, parameters: list[str] | None = None) -> None:
        self.parameters = ['weights', 'bias']
        self.weights = np.ones((5, 5))
        self.bias = np.ones(5)
        self.gradient_weights = np.ones((5, 5))
        self.gradient_bias = np.ones(5)

    def __call__(self: Self, *args, **kwargs) -> npt.NDArray[np.floating]:
        return np.ones((5, 5))

    def compute_backward_gradient(
        self: Self, gradient: npt.NDArray[np.floating],
    ) -> npt.NDArray[np.floating]:
        return gradient + 1


class MockModel:
    def __init__(self, layers_count: int) -> None:
        self._layers = [MockLayer() for _ in range(layers_count)]

    @property
    def layers(self) -> list[MockLayer]:
        return self._layers


def test_sgd_compute_gradient() -> None:
    model = SGD(cast(MLP, MockModel(3)))
    gradient = np.ones((5, 5))

    result = model.compute_backward_gradient(gradient)

    assert (result == gradient + 3).all()


def test_sgd_step():
    model = MockModel(1)
    expected_value = 0.9

    sgd = SGD(model, learning_rate=0.1)
    sgd.step()

    assert np.allclose(model.layers[0].weights, expected_value)
    assert np.allclose(model.layers[0].bias, expected_value)
