import numpy as np


class SGD:
    """A class for implementing Stochastic gradient descent."""

    def __init__(self, model, learning_rate=1e-4):
        self.learning_rate = learning_rate
        self.model = model

    def backward(self, grad: np.ndarray):
        """Backward pass for the model.

        This method propagates gradient across all layers from model.layers in reverse order.

        Compute gradients sequentially by passing through each layer in model.layers (using layer's backward() method)

        Args:
            grad: the gradient of the loss function w.r.t. the model output - ∇_{Z^L} E
        """
        # TODO: Implement this method
        raise NotImplementedError

    def step(self):
        """Updates the parameters of the model layers."""
        # TODO: For each layer in model.layers:
        #  If the layer has parameters, for each parameter in layer.parameters do:
        #       - get the parameter value with getattr(layer, parameter_name)
        #       - get stored gradient value with getattr(layer, f'grad_{parameter_name}')
        #       - compute new parameter value with update_param() method
        #       - set new parameter value with setattr(layer, parameter_name, new_value)
        raise NotImplementedError

    def update_param(self, param: np.ndarray, grad: np.ndarray):
        """Update layer parameters.

        Layers parameters should be updated according to the following rule:
            w_new = w_old - γ * ∇_{w_old} E,

            where:
                γ - self.learning_rate,
                w_old - layer's current parameter value,
                ∇_{w_old} E - gradient w.r.t. the layer's parameter,
                w_new - new parameter value to set.

        Args:
            param: parameter matrix (w_old)
            grad: gradient matrix (∇_{w_old} E)

        Returns:
            w_new: matrix
        """
        # TODO: Implement this method
        raise NotImplementedError

    def zero_grad(self):
        """Reset gradient parameters for the model layers."""
        for layer in self.model.layers:
            layer.zero_grad()
