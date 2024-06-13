from abc import abstractmethod
from typing import Self

import numpy as np
from numpy import typing as npt


class BaseLayer:
    """A base layer class."""

    def __init__(self: Self, parameters: list[str] | None = None) -> None:
        self._parameters = parameters if parameters is not None else []
        self._inputs_cache: npt.NDArray[np.floating] | None = None
        self._is_trainable = True

    @abstractmethod
    def __call__(self: Self, *args, **kwargs) -> npt.NDArray[np.floating]:
        raise NotImplementedError

    def set_train(self: Self) -> None:
        """Set training mode."""
        self._is_trainable = True

    def set_eval(self: Self) -> None:
        """Set evaluation mode."""
        self._is_trainable = False

    def load_params(self: Self, params: dict) -> None:
        """Load layer parameters.

        Args:
            params: dictionary with parameters names (as dict keys) and their
                values (as dict values).
        """
        for param_name, param_value in params.items():
            setattr(self, param_name, param_value)

    def get_params(self: Self) -> dict:
        """Return layer parameters."""
        return {param: getattr(self, param) for param in self._parameters}

    @abstractmethod
    def backward(
        self: Self, grad: npt.NDArray[np.floating],
    ) -> npt.NDArray[np.floating]:
        raise NotImplementedError

    def zero_grad(self: Self) -> None:
        for param_name in self._parameters:
            setattr(
                self, f'grad_{param_name}', np.zeros_like(
                    getattr(self, param_name),
                ),
            )
