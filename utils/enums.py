from enum import IntEnum, Enum

SetType = IntEnum('SetType', ('train', 'validation', 'test'))
SamplerType = IntEnum('SamplerType', ('Default', 'Upsampling'))
TransformsType = IntEnum('TransformsType', ('Resize', 'Normalize', 'Standardize', 'ToFloat'))
WeightsInitType = IntEnum('WeightsInitType', ('normal', 'uniform', 'he', 'xavier', 'xavier_normalized'))
