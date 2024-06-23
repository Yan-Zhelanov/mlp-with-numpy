import sys

import numpy as np
from numpy import typing as npt

from configs.mlp_config import ModelConfig
from modules.layers.activations import LeakyReLU, ReLU, Sigmoid, Tanh
from modules.layers.base import BaseLayer
from modules.layers.linear import Linear
from modules.utils.parameter_initialization import ParametersInit


class MLP:
    """A class for implementing Multilayer perceptron model."""

    def __init__(self, config: ModelConfig) -> None:
        """Initialize MLP model.

        Args:
            config: model configurations
        """
        self._config = config
        self._params_init = ParametersInit(config)
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
        # TODO: Implement this method
        raise NotImplementedError

    @property
    def layers(self) -> list[BaseLayer]:
        """Returns the list of layers in the model.

        Returns:
            list[BaseLayer]: list of layers.
        """
        return self._layers

    def train(self) -> None:
        """Set the training mode for each layer."""
        for layer in self._layers:
            layer.set_train()

    def eval(self) -> None:
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

        This method should include the following steps:
        1. Go through the layers ((layer_name, layer_params) tuples) defined in
            self.config.layers one by one.
        2. For each layer, get an instance of the layer class with layer_name
            and layer_params as:
            layer = getattr(sys.modules[__name__], layer_name)(**layer_params)
        3. For layer instance call self.params_init function to initialize
            layer parameters if it has those (!).
        4. Append the initialized layer to the layers list.
        5. Return list with all the initialized layers.

        Returns:
            list of initialized layers
        """
        # TODO: Implement this method
        layers: list[BaseLayer] = []
        raise NotImplementedError
