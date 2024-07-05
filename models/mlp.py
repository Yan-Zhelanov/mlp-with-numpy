import sys

import numpy as np
from numpy import typing as npt

from configs.mlp_config import ModelConfig
from modules.layers.activations import (  # noqa: F401
    LeakyReLU,
    ReLU,
    Sigmoid,
    Tanh,
)
from modules.layers.base import BaseLayer
from modules.layers.dropout import Dropout  # noqa: F401
from modules.layers.linear import Linear  # noqa: F401
from modules.utils.parameter_initialization import ParameterInitializator


class MLP:
    """A class for implementing Multilayer perceptron model."""

    def __init__(self, config: ModelConfig) -> None:
        """Initialize MLP model.

        Args:
            config: model configurations
        """
        self._config = config
        self._initializator = ParameterInitializator(config)
        self._layers: list[BaseLayer] = self._init_layers()

    def __call__(
        self, inputs: npt.NDArray[np.floating],
    ) -> npt.NDArray[np.floating]:
        """Forward propagation implementation.

        This method propagates inputs through all the layers from self.layers.
        It should include the following steps:
        1. Reshape inputs into a 2D array (first dimension is the mini-batch
        size).
        2. Update inputs sequentially by passing through each layer in
        self._layers (using layer's __call__() method).
        3. Return updated inputs.

        Args:
            inputs (npt.NDArray[np.floating]): inputs to the model.

        Returns:
            npt.NDArray: updated inputs.
        """
        inputs = inputs.reshape(inputs.shape[0], -1)
        for layer in self._layers:
            inputs = layer(inputs)
        return inputs

    @property
    def layers(self) -> list[BaseLayer]:
        """Returns the list of layers in the model.

        Returns:
            list[BaseLayer]: list of layers.
        """
        return self._layers

    def set_train(self) -> None:
        """Set the training mode for each layer."""
        for layer in self._layers:
            layer.set_train()

    def set_eval(self) -> None:
        """Set the evaluation mode for each layer."""
        for layer in self._layers:
            layer.set_eval()

    def load_params(self, parameters: list[dict[str, npt.NDArray]]) -> None:
        """Load model parameters.

        Args:
            parameters (list[dict[str, npt.NDArray]]): list of dictionaries
                with parameters names and their values.

        Raises:
            ValueError: if the number of parameters does not match the number
                of layers in the model.
        """
        if len(parameters) != len(self._layers):
            raise ValueError(
                f'Invalid number of parameters: {len(parameters)} when'
                + f' expected: {len(self._layers)}',
            )
        for layer_index, layer in enumerate(self._layers):
            layer.load_params(parameters[layer_index])

    def get_params(self) -> list[dict[str, npt.NDArray]]:
        """Get model parameters.

        Returns:
            list[dict[str, npt.NDArray]]: list of dictionaries with parameters
                names and their values.
        """
        parameters = []
        for layer in self._layers:
            parameters.append(layer.get_params())
        return parameters

    def _init_layers(self) -> list[BaseLayer]:
        """Initialize MLP layers.

        Returns:
            list of initialized layers
        """
        layers: list[BaseLayer] = []
        for layer_name, layer_params in self._config.LAYERS:
            layer: BaseLayer = getattr(sys.modules[__name__], layer_name)(
                **layer_params,
            )
            if layer.parameters:
                self._initializator(layer)
            layers.append(layer)
        return layers
