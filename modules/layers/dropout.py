import numpy as np
from numpy import typing as npt

from modules.layers.base import BaseLayer


class Dropout(BaseLayer):
    """Dropout layer."""

    def __init__(self, keeping_probability: float) -> None:
        """Initialize dropout layer.

        Args:
            keeping_probability: neuron keeping probability.
        """
        super().__init__()
        self._keeping_probability = keeping_probability

    def __call__(
        self, inputs: npt.NDArray[np.floating],
    ) -> npt.NDArray[np.floating]:
        """Pass forward through dropout layer.

        Args:
            inputs: matrix of shape (batch_size, shape).

        Returns:
            npt.NDArray[np.floating]: output matrix (same shape as inputs).
        """
        if self._is_trainable:
            mask = np.random.rand(*inputs.shape) < self._keeping_probability
            self.inputs_cache = mask
            return 1 / self._keeping_probability * mask * inputs
        return inputs

    def compute_backward_gradient(
        self, gradient: npt.NDArray[np.floating],
    ) -> npt.NDArray[np.floating]:
        """Pass backward through dropout layer.

        Args:
            gradient: matrix of shape (batch_size, shape)

        Returns:
            npt.NDArray[np.floating]: gradient matrix (same shape as inputs).
        """
        return 1 / self._keeping_probability * self.inputs_cache * gradient
