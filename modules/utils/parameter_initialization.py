import numpy as np
from numpy import typing as npt

from configs.mlp_config import ModelConfig


class ParametersInit:
    """Training parameters initialization."""

    def __init__(self, config: ModelConfig):
        self._config = config
        self._init_function = getattr(
            self, f'get_init_{self._config.INIT_TYPE.name.lower()}',
        )

    def __call__(self, layer):
        """Initializing layer parameters"""
        for param_name in layer.parameters:
            param = getattr(layer, param_name)
            if param_name == 'bias' and self._config.ZERO_BIAS:
                setattr(layer, param_name, self.get_init_zeros(param.shape))
            else:
                setattr(layer, param_name, self._init_function(param.shape))

    def get_init_zeros(
        self, param_shape: tuple[int, ...],
    ) -> npt.NDArray[np.floating]:
        """Initialize with zeros.

        Args:
            param_shape: shape of the parameters.

        Returns:
            npt.NDArray[np.floating]: initialized parameters.
        """
        return np.zeros(param_shape)

    def get_init_normal(
        self, param_shape: tuple[int, ...],
    ) -> npt.NDArray[np.floating]:
        """Initialization with values from a normal distribution.

        W ~ N(mu, sigma^2),
        where:
        - mu and sigma can be defined in self.config.init_kwargs

        Args:
            param_shape: shape of the parameters.

        Returns:
            npt.NDArray[np.floating]: initialized parameters.
        """
        # TODO: Implement this method using np.random.normal
        #  (method should return initialized values)
        raise NotImplementedError

    def get_init_uniform(
        self, param_shape: tuple[int, ...],
    ) -> npt.NDArray[np.floating]:
        """Initialize with values from a uniform distribution.

        W ~ U(-epsilon, epsilon),

        where:
        - epsilon can be defined in self.config.init_kwargs

        Args:
            param_shape: shape of the parameters.

        Returns:
            npt.NDArray[np.floating]: initialized parameters.
        """
        # TODO: Implement this method using np.random.uniform
        #  (method should return initialized values)
        raise NotImplementedError

    def get_init_he(
        self, param_shape: tuple[int, ...],
    ) -> npt.NDArray[np.floating]:
        """Initialize He.

        Initialization with values from a normal distribution with the
        following parameters: W ~ N(0, 2 / M_{l-1}),
        where:
        - M_{l-1} - number of input features of a layer

        Args:
            param_shape: shape of the parameters.

        Returns:
            npt.NDArray[np.floating]: initialized parameters.
        """
        # TODO: Implement this method using np.random.normal
        #  (method should return initialized values)
        raise NotImplementedError

    def get_init_xavier(
        self, param_shape: tuple[int, ...],
    ) -> npt.NDArray[np.floating]:
        """Xavier initialization.

        Initialization with values from a uniform distribution with the
        following parameters: W ~ U(-epsilon, epsilon),

        where:
        - epsilon = sqrt{1 / M_{l-1}},
        - M_{l-1} - number of input features of a layer

        Args:
            param_shape: shape of the parameters.

        Returns:
            npt.NDArray[np.floating]: initialized parameters.
        """
        # TODO: Implement this method using np.random.uniform
        #  (method should return initialized values)
        raise NotImplementedError

    def get_init_xavier_normalized(
        self, param_shape: tuple[int, ...],
    ) -> npt.NDArray[np.floating]:
        """Xavier normalized initialization.

        Initialization with values from a uniform distribution with the
        following parameters: W ~ U(-epsilon, epsilon),
        where:
        - epsilon = sqrt{6 / (M_{l-1} + M_l)},
        - M_{l-1} - number of input features of a layer,
        - M_l - number of output features of a layer.

        Args:
            param_shape: shape of the parameters.

        Returns:
            npt.NDArray[np.floating]: initialized parameters.
        """
        # TODO: Implement this method using np.random.uniform
        #  (method should return initialized values)
        raise NotImplementedError
