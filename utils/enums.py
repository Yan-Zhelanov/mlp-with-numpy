from enum import Enum


class SetType(Enum):
    TRAIN = 1
    VALIDATION = 2
    TEST = 3


class SamplerType(Enum):
    DEFAULT = 1
    UPSAMPLING = 2


class TransformsType(Enum):
    RESIZE = 1
    NORMALIZE = 2
    STANDARDIZE = 3
    TO_FLOAT = 4


class WeightsInitType(Enum):
    NORMAL = 1
    UNIFORM = 2
    HE = 3
    XAVIER = 4
    XAVIER_NORMALIZED = 5
