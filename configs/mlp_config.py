from typing import Any

from utils.enums import WeightsInitType

_IMAGE_SIZE = 32
_INPUT_SHAPE = _IMAGE_SIZE * _IMAGE_SIZE


class ModelConfig:
    LAYERS = [
        ('Linear', {'input_shape': _INPUT_SHAPE, 'output_shape': 128}),
        ('ReLU', {}),
        ('Linear', {'input_shape': 128, 'output_shape': 64}),
        ('ReLU', {}),
        ('Linear', {'input_shape': 64, 'output_shape': 7}),
    ]

    # Weights and bias initialization
    INIT_TYPE = WeightsInitType.XAVIER_NORMALIZED
    INIT_KWARGS = {'mu': 0, 'sigma': 0.01}
    ZERO_BIAS = True

    def get_all_hyperparameters(self) -> dict[str, Any]:
        """Get all hyperparameters for the model.

        Returns:
            dict[str, Any]: dictionary with all hyperparameters.
        """
        return {
            'layers': self.LAYERS,
            'init_type': self.INIT_TYPE,
            'init_kwargs': self.INIT_KWARGS,
            'zero_bias': self.ZERO_BIAS,
        }
