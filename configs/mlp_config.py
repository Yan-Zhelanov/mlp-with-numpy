from utils.enums import WeightsInitType


class ModelConfig(object):
    LAYERS = [
        ('Linear', {'input_shape': 32 * 32, 'output_shape': 128}),
        ('ReLU', {}),
        ('Linear', {'input_shape': 128, 'output_shape': 64}),
        ('ReLU', {}),
        ('Linear', {'input_shape': 64, 'output_shape': 7}),
    ]

    # Weights and bias initialization
    INIT_TYPE = WeightsInitType.NORMAL
    INIT_KWARGS = {'mu': 0, 'sigma': 0.001}
    ZERO_BIAS = True
