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

        This method propagates gradient across all layers in reverse order.

        Args:
            gradient (npt.NDArray[np.floating]): the gradient of the loss
                function w.r.t. the model output - ∇_{Z^L} E.

        Returns:
            npt.NDArray: the gradient of the loss function.
        """
        for layer in reversed(self._model.layers):
            gradient = layer.compute_backward_gradient(gradient)
        return gradient

    def step(self) -> None:
        """Update the parameters of the model layers."""
        for layer in self._model.layers:
            if hasattr(layer, 'parameters') and layer.parameters is not None:
                for parameter in layer.parameters:
                    old_value = getattr(layer, parameter)
                    gradient = getattr(layer, f'gradient_{parameter}')
                    new_value = self.update_parameters(old_value, gradient)
                    setattr(layer, parameter, new_value)

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
        return parameters - self._learning_rate * gradient

    def zero_grad(self) -> None:
        """Reset gradient parameters for the model layers."""
        for layer in self._model._layers:
            layer.set_gradients_to_zero()
