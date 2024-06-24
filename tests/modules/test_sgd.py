from typing import Self, cast

import numpy as np
from numpy import typing as npt

from models.mlp import MLP
from modules.layers.base import BaseLayer
from modules.optimizers.sgd import SGD


class MockLayer(BaseLayer):
    def __call__(self: Self, *args, **kwargs) -> npt.NDArray[np.floating]:
        return np.ones((5, 5))

    def compute_backward_gradient(
        self: Self, gradient: npt.NDArray[np.floating],
    ) -> npt.NDArray[np.floating]:
        return gradient + 1


class MockModel:
    def __init__(self, layers: int) -> None:
        self._layers_count = layers

    @property
    def layers(self) -> list[BaseLayer]:
        return [MockLayer() for _ in range(self._layers_count)]


def test_sgd_compute_gradient() -> None:
    model = SGD(cast(MLP, MockModel(3)))
    gradient = np.ones((5, 5))

    result = model.compute_backward_gradient(gradient)

    assert (result == gradient + 3).all()
