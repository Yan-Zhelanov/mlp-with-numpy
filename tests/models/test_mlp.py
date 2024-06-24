import numpy as np

from configs.mlp_config import WeightsInitType
from models.mlp import MLP
from modules.layers.linear import Linear

_INPUT_SHAPE = 4
_OUTPUT_SHAPE = 2
_TOLERANCE = 0.1


class MockConfig:
    LAYERS = [
        (
            'Linear', {
                'input_shape': _INPUT_SHAPE, 'output_shape': _OUTPUT_SHAPE,
            },
        ),
    ]
    INIT_TYPE = WeightsInitType.NORMAL
    INIT_KWARGS = {'mu': 0, 'sigma': 0.001}
    ZERO_BIAS = True


def test_init_layers():
    config = MockConfig()
    mlp = MLP(config)

    layers = mlp._init_layers()

    assert len(layers) == 1
    assert isinstance(layers[0], Linear)
    assert layers[0]._input_shape == _INPUT_SHAPE
    assert layers[0]._output_shape == _OUTPUT_SHAPE


def test_mlp():
    config = MockConfig()
    mlp = MLP(config)
    inputs = np.array([1, 2, 3, 4]).reshape((1, 4))

    inputs = mlp(inputs)

    assert (inputs[0] != 0).all()
    assert np.allclose(
        inputs, np.array([0, 0]).reshape((1, 2)), atol=_TOLERANCE,
    )
