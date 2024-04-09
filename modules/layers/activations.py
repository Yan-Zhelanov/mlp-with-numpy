from modules.layers.base import BaseLayer

import numpy as np


class ReLU(BaseLayer):
    """ReLU (rectified linear unit) activation function."""

    def __init__(self):
        super().__init__()

    def __call__(self, a):
        """Forward pass for ReLU.

        For mini-batch, ReLU forward pass can be defined as follows:
            z = max(0, a)

            where:
                - a (batch_size x M_l matrix) represents the output of fully-connected layer,
                - z (batch_size x M_l matrix) represents activations.

        During the training phase, inputs are stored in self.inputs_cache for back propagation.

        Args:
            a: matrix of shape (batch_size, M_l)

        Returns:
            np.ndarray: matrix of shape (batch_size, M_l)
        """
        # TODO: Implement this method
        raise NotImplementedError

    def backward(self, grad):
        """Backward pass for ReLU.

        For mini-batch, activation function backward pass can be defined as follows:
            ∇_{A^l} E = f'(A^l) * ∇_{Z^l} E,

        where:
            - f'(A^l) (batch_size x M_l matrix): derivative of activation function
            - A^l (batch_size x M_l matrix): inputs_cache, that stored during the training phase at forward propagation


        The derivative of ReLU activation function can be defined as follows:
            f'(x) = {0 if x < 0 and 1 otherwise}

        Args:
            grad: matrix of shape (batch_size, M_l) - ∇_{Z^l} E

        Returns:
            ∇_{A^l} E: matrix of shape (batch_size, M_l)
        """
        # TODO: Implement ReLU backward propagation
        raise NotImplementedError


class LeakyReLU(BaseLayer):
    """Leaky ReLU (rectified linear unit) activation function."""

    def __init__(self, alpha=0.01):
        super().__init__()
        self.alpha = alpha

    def __call__(self, a):
        """Forward pass for LeakyReLU.

        For mini-batch, LeakyReLU forward pass can be defined as follows:
            z = max(0, a) + alpha * min(0, a)

           where:
                - a (batch_size x M_l matrix) represents the output of fully-connected layer,
                - z (batch_size x M_l matrix) represents activations.

        During the training phase, inputs are stored in self.inputs_cache for back propagation.

        Args:
            a: matrix of shape (batch_size, M_l)

        Returns:
            np.ndarray: matrix of shape (batch_size, M_l)
        """
        # TODO: Implement this method
        raise NotImplementedError

    def backward(self, grad):
        """Backward pass for LeakyReLU.

        For mini-batch, activation function backward pass can be defined as follows:
            ∇_{A^l} E = f'(A^l) * ∇_{Z^l} E,

        where:
            - f'(A^l) (batch_size x M_l matrix): derivative of activation function
            - A^l (batch_size x M_l matrix): inputs_cache, that stored during the training phase at forward propagation


        The derivative of LeakyReLU activation function can be defined as follows:
            f'(x) = {1 if x > 0 and alpha otherwise}

        Args:
            grad: matrix of shape (batch_size, M_l) - ∇_{Z^l} E

        Returns:
            ∇_{A^l} E: matrix of shape (batch_size, M_l)
        """
        # TODO: Implement LeakyReLU backward propagation
        raise NotImplementedError


class Sigmoid(BaseLayer):
    """Sigmoid activation function."""

    def __init__(self):
        super().__init__()

    def __call__(self, a):
        """Forward pass for sigmoid function.

        For mini-batch, sigmoid function forward pass can be defined as follows:
            z = 1 / (1 + e^(-a))

            where:
                - a (batch_size x M_l matrix) represents the output of fully-connected layer,
                - z (batch_size x M_l matrix) represents activations.

        During the training phase, inputs are stored in self.inputs_cache for back propagation.

        Args:
            a: matrix of shape (batch_size, M_l)

        Returns:
            np.ndarray: matrix of shape (batch_size, M_l)
        """
        # TODO: Implement this method
        raise NotImplementedError

    def backward(self, grad):
        """Backward pass for Sigmoid.

        For mini-batch, activation function backward pass can be defined as follows:
            ∇_{A^l} E = f'(A^l) * ∇_{Z^l} E,

        where:
            - f'(A^l) (batch_size x M_l matrix): derivative of activation function
            - A^l (batch_size x M_l matrix): inputs_cache, that stored during the training phase at forward propagation


        The derivative of Sigmoid activation function can be defined as follows:
            f'(x) = f(x) * (1 - f(x)),

        where:
            - f(x) = 1 / (1 + e^(-x))

        Args:
            grad: matrix of shape (batch_size, M_l) - ∇_{Z^l} E

        Returns:
            ∇_{A^l} E: matrix of shape (batch_size, M_l)
        """
        # TODO: Implement Sigmoid backward propagation
        raise NotImplementedError


class Tanh(BaseLayer):
    """Tanh activation function."""

    def __init__(self):
        super().__init__()

    def __call__(self, a):
        """Forward pass for Tanh.

        For mini-batch, Tanh function forward pass can be defined as follows:
            z = (e^a - e^(-a)) / (e^a + e^(-a))

            where:
                - a (batch_size x M_l matrix) represents the output of fully-connected layer,
                - z (batch_size x M_l matrix) represents activations.

        During the training phase, inputs are stored in self.inputs_cache for back propagation.

        Args:
            a: matrix of shape (batch_size, M_l)

        Returns:
            np.ndarray: matrix of shape (batch_size, M_l)
        """
        # TODO: Implement this method
        raise NotImplementedError

    def backward(self, grad):
        """Backward pass for Tanh.

        For mini-batch, activation function backward pass can be defined as follows:
            ∇_{A^l} E = f'(A^l) * ∇_{Z^l} E,

        where:
            - f'(A^l) (batch_size x M_l matrix): derivative of activation function
            - A^l (batch_size x M_l matrix): inputs_cache, that stored during the training phase at forward propagation


        The derivative of Tanh activation function can be defined as follows:
            f'(x) = 1 - f(x)^2,

        where:
            - f(x) = tanh(x)

        Args:
            grad: matrix of shape (batch_size, M_l) - ∇_{Z^l} E

        Returns:
            ∇_{A^l} E: matrix of shape (batch_size, M_l)
        """
        # TODO: Implement Tanh backward propagation
        raise NotImplementedError
