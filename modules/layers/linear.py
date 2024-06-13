import numpy as np
from numpy import typing as npt

from modules.layers.base import BaseLayer


class Linear(BaseLayer):
    """Fully-connected layer."""

    def __init__(self, input_shape: int, output_shape: int) -> None:
        """Initialize weights and bias for fully-connected layer.

        Args:
            input_shape: number of input features (M_{l-1})
            output_shape: number of output features (M_l)
        """
        super().__init__(['weights', 'bias'])
        self._input_shape = input_shape
        self._output_shape = output_shape
        self._weights = np.zeros((output_shape, input_shape))
        self._grad_weights = None
        self._bias = np.zeros(output_shape)
        self._grad_bias = None

    def __call__(
        self, layer_input: npt.NDArray[np.floating],
    ) -> npt.NDArray[np.floating]:
        """Forward pass for fully-connected layer.

        For minibatch, FC layer forward pass can be defined as follows:
        z * W^T + b,
        where:
        - z (batch_size x input_shape matrix) represents the output of the
            previous layer,
        - W (output_shape x input_shape matrix) a matrix represents the weight
            matrix,
        - b (vector of length output_shape) represents the bias vector.

        During the training phase, inputs are stored in self.inputs_cache for
        back propagation.

        Args:
            layer_input: matrix of shape (batch_size, input_shape).

        Returns:
            np.ndarray: matrix of shape (batch_size, output_shape)
        """
        return layer_input @ self._weights.T + self._bias

    def backward(
        self, grad: npt.NDArray[np.floating],
    ) -> npt.NDArray[np.floating]:
        """Backward pass for fully-connected layer.

        For mini-batch, FC layer backward pass can be defined as follows:

            ∇_{b^l} E = Σ(i=0 to N-1) u_i
            ∇_{W^l} E = (∇_{A^l} E)^T * Z^{l-1}
            ∇_{Z^{l-1}} E = ∇_{A^l} E * W^l

        where:
            - u_i:  i-th row of matrix ∇_{A^l} E
            - W^l (output_shape x input_shape matrix): weights of current layer
            - Z^{l-1} (batch_size x input_shape matrix): inputs_cache, that
                stored during the training phase at forward propagation

        Store gradients of weights and bias in grad_weights and grad_bias

        Args:
            grad: matrix of shape (batch_size, output_shape) - ∇_{A^l} E

        Returns:
            ∇_{Z^{l-1}} E: matrix of shape (batch_size, input_shape)
        """
        # TODO: Implement fully-connected layer backward propagation
        raise NotImplementedError
