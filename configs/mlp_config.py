from easydict import EasyDict

from utils.enums import WeightsInitType

model_cfg = EasyDict()

# Layers configuration
model_cfg.layers = [
    ('Linear', dict(input_shape=32 * 32, output_shape=128)),
    ('ReLU', dict()),
    ('Linear', dict(input_shape=128, output_shape=64)),
    ('ReLU', dict()),
    ('Linear', dict(input_shape=64, output_shape=7)),
]

# Weights and bias initialization
model_cfg.params = EasyDict()
model_cfg.params.init_type = WeightsInitType.NORMAL
model_cfg.params.init_kwargs = {'mu': 0, 'sigma': 0.001}
model_cfg.params.zero_bias = True
