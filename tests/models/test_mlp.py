from configs.mlp_config import WeightsInitType
from models.mlp import MLP
from modules.layers.linear import Linear


class MockConfig:
    LAYERS = [
        ('Linear', {'input_shape': 1024, 'output_shape': 128}),
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
    assert layers[0]._input_shape == 1024
    assert layers[0]._output_shape == 128
