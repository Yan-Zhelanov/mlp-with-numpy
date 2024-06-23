import numpy as np
from numpy import typing as npt

from models.mlp import MLP


class SGD:
    """A class for implementing Stochastic gradient descent."""

    def __init__(self, model: MLP, learning_rate=0.0001) -> None:
        self._learning_rate = learning_rate
        self._model = model

    def compute_backward_gradient(
        self, gradient: npt.NDArray[np.floating],
    ) -> npt.NDArray[np.floating]:
        """Compute a backward gradient for the all model.

        This method propagates gradient across all layers from model.layers in
        reverse order.
        Compute gradients sequentially by passing through each layer in
        model.layers (using layer's backward() method).

        Args:
            gradient: the gradient of the loss function w.r.t. the model
                output - ∇_{Z^L} E.
        """
        # TODO: Implement this method
        raise NotImplementedError

    def step(self) -> None:
        """Update the parameters of the model layers."""
        # TODO: For each layer in model.layers:
        #  If the layer has parameters, for each parameter in layer.parameters do:
        #       - get the parameter value with getattr(layer, parameter_name)
        #       - get stored gradient value with getattr(layer, f'grad_{parameter_name}')
        #       - compute new parameter value with update_param() method
        #       - set new parameter value with setattr(layer, parameter_name, new_value)
        raise NotImplementedError

    def update_parameters(
        self,
        parameters: npt.NDArray[np.floating],
        gradient: npt.NDArray[np.floating],
    ) -> npt.NDArray[np.floating]:
        """Update layer parameters.

        Layers parameters should be updated according to the following rule:
        w_new = w_old - γ * ∇_{w_old} E,

        where:
        γ - self.learning_rate,
        w_old - layer's current parameter value,
        ∇_{w_old} E - gradient w.r.t. the layer's parameter,
        w_new - new parameter value to set.

        Args:
            parameters (npt.NDArray[np.floating]): The old weights matrix.
            gradient (npt.NDArray[np.floating]): Gradient matrix (∇_{w_old} E).

        Returns:
            npt.NDArray[np.floating]: The new weights matrix.
        """
        # TODO: Implement this method
        raise NotImplementedError

    def zero_grad(self) -> None:
        """Reset gradient parameters for the model layers."""
        for layer in self._model._layers:
            layer.zero_grad()
