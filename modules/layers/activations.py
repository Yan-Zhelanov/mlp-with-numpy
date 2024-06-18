import numpy as np
from numpy import typing as npt

from modules.layers.base import BaseLayer


class ReLU(BaseLayer):
    """ReLU (rectified linear unit) activation function."""

    def __call__(
        self, layer_input: npt.NDArray[np.floating],
    ) -> npt.NDArray[np.floating]:
        """Forward pass for ReLU.

        For mini-batch, ReLU forward pass can be defined as follows:
        z = max(0, a)

        where:
        - a (batch_size x M_l matrix) represents the output of fully-connected
            layer,
        - z (batch_size x M_l matrix) represents activations.

        During the training phase, inputs are stored in self.inputs_cache for
        back propagation.

        Args:
            layer_input: matrix of shape (batch_size, M_l)

        Returns:
            npt.NDArray: matrix of shape (batch_size, M_l)
        """
        if self._is_trainable:
            self._inputs_cache = layer_input
        mask = layer_input > 0
        return layer_input * mask

    def compute_backward_gradient(
        self, gradient: npt.NDArray[np.floating],
    ) -> npt.NDArray[np.floating]:
        """Backward pass for ReLU.

        For mini-batch, activation function backward pass can be defined as
        follows: ∇_{A^l} E = f'(A^l) * ∇_{Z^l} E,

        where:
        - f'(A^l) (batch_size x M_l matrix): derivative of activation function
        - A^l (batch_size x M_l matrix): inputs_cache, that stored during the
            training phase at forward propagation

        The derivative of ReLU activation function can be defined as follows:
            f'(x) = {0 if x < 0 and 1 otherwise}

        Args:
            gradient: matrix of shape (batch_size, M_l) - ∇_{Z^l} E

        Returns:
            ∇_{A^l} E: matrix of shape (batch_size, M_l)
        """
        if self._inputs_cache is None:
            raise RuntimeError('Layer is not in training mode!')
        return (self._inputs_cache > 0) * gradient


class LeakyReLU(BaseLayer):
    """Leaky ReLU (rectified linear unit) activation function."""

    def __init__(self, alpha: float = 0.01) -> None:
        super().__init__()
        self._alpha = alpha

    def __call__(
        self, layer_input: npt.NDArray[np.floating],
    ) -> npt.NDArray[np.floating]:
        """Forward pass for LeakyReLU.

        For mini-batch, LeakyReLU forward pass can be defined as follows:
        z = max(0, a) + alpha * min(0, a)

        where:
        - a (batch_size x M_l matrix) represents the output of fully-connected
            layer,
        - z (batch_size x M_l matrix) represents activations.

        During the training phase, inputs are stored in self.inputs_cache for
        back propagation.

        Args:
            layer_input: matrix of shape (batch_size, M_l)

        Returns:
            np.ndarray: matrix of shape (batch_size, M_l)
        """
        if self._is_trainable:
            self._inputs_cache = layer_input
        return np.maximum(self._alpha * layer_input, layer_input)

    def compute_backward_gradient(
        self, gradient: npt.NDArray[np.floating],
    ) -> npt.NDArray[np.floating]:
        """Backward pass for LeakyReLU.

        For mini-batch, activation function backward pass can be defined as
        follows: ∇_{A^l} E = f'(A^l) * ∇_{Z^l} E,

        where:
        - f'(A^l) (batch_size x M_l matrix): derivative of activation function
        - A^l (batch_size x M_l matrix): inputs_cache, that stored during the
            training phase at forward propagation


        The derivative of LeakyReLU activation function can be defined as
        follows: f'(x) = {1 if x > 0 and alpha otherwise}

        Args:
            gradient: matrix of shape (batch_size, M_l) - ∇_{Z^l} E

        Returns:
            ∇_{A^l} E: matrix of shape (batch_size, M_l)
        """
        # TODO: Implement LeakyReLU backward propagation
        raise NotImplementedError


class Sigmoid(BaseLayer):
    """Sigmoid activation function."""

    def __call__(
        self, layer_input: npt.NDArray[np.floating],
    ) -> npt.NDArray[np.floating]:
        """Forward pass for sigmoid function.

        For mini-batch, sigmoid function forward pass can be defined as
        follows: z = 1 / (1 + e^(-a))

        where:
        - a (batch_size x M_l matrix) represents the output of fully-connected
            layer,
        - z (batch_size x M_l matrix) represents activations.

        During the training phase, inputs are stored in self.inputs_cache for
        back propagation.

        Args:
            layer_input: matrix of shape (batch_size, M_l)

        Returns:
            np.ndarray: matrix of shape (batch_size, M_l)
        """
        # TODO: Implement this method
        raise NotImplementedError

    def compute_backward_gradient(
        self, gradient: npt.NDArray[np.floating],
    ) -> npt.NDArray[np.floating]:
        """Backward pass for Sigmoid.

        For mini-batch, activation function backward pass can be defined as
        follows: ∇_{A^l} E = f'(A^l) * ∇_{Z^l} E,

        where:
        - f'(A^l) (batch_size x M_l matrix): derivative of activation function
        - A^l (batch_size x M_l matrix): inputs_cache, that stored during the
        training phase at forward propagation


        The derivative of Sigmoid activation function can be defined as
        follows: f'(x) = f(x) * (1 - f(x)),

        where:
            - f(x) = 1 / (1 + e^(-x))

        Args:
            gradient: matrix of shape (batch_size, M_l) - ∇_{Z^l} E

        Returns:
            ∇_{A^l} E: matrix of shape (batch_size, M_l)
        """
        # TODO: Implement Sigmoid backward propagation
        raise NotImplementedError


class Tanh(BaseLayer):
    """Tanh activation function."""

    def __call__(
        self, layer_input: npt.NDArray[np.floating],
    ) -> npt.NDArray[np.floating]:
        """Forward pass for Tanh.

        For mini-batch, Tanh function forward pass can be defined as follows:
        z = (e^a - e^(-a)) / (e^a + e^(-a))

        where:
        - a (batch_size x M_l matrix) represents the output of fully-connected
            layer,
        - z (batch_size x M_l matrix) represents activations.

        During the training phase, inputs are stored in self.inputs_cache for
        back propagation.

        Args:
            layer_input: matrix of shape (batch_size, M_l)

        Returns:
            np.ndarray: matrix of shape (batch_size, M_l)
        """
        # TODO: Implement this method
        raise NotImplementedError

    def compute_backward_gradient(
        self, gradient: npt.NDArray[np.floating],
    ) -> npt.NDArray[np.floating]:
        """Backward pass for Tanh.

        For mini-batch, activation function backward pass can be defined as
        follows: ∇_{A^l} E = f'(A^l) * ∇_{Z^l} E,

        where:
        - f'(A^l) (batch_size x M_l matrix): derivative of activation function
        - A^l (batch_size x M_l matrix): inputs_cache, that stored during the
            training phase at forward propagation

        The derivative of Tanh activation function can be defined as follows:
            f'(x) = 1 - f(x)^2,

        where:
            - f(x) = tanh(x)

        Args:
            gradient: matrix of shape (batch_size, M_l) - ∇_{Z^l} E

        Returns:
            ∇_{A^l} E: matrix of shape (batch_size, M_l)
        """
        # TODO: Implement Tanh backward propagation
        raise NotImplementedError
