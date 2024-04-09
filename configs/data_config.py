import os
from easydict import EasyDict

from utils.enums import SamplerType, TransformsType

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))

data_config = EasyDict()

# Path to the directory with dataset files
data_config.path_to_data = os.path.join(ROOT_DIR, 'data', 'emotion_detection')
data_config.annot_filename = 'data_info.csv'

# Label mapping
data_config.label_mapping = {
    'angry': 0,
    'disgusted': 1,
    'fearful': 2,
    'happy': 3,
    'neutral': 4,
    'sad': 5,
    'surprised': 6
}

data_config.sampler_type = SamplerType.Default

data_config.train_transforms = [
    (TransformsType.Resize, dict(size=(32, 32))),
    (TransformsType.ToFloat, dict()),
    (TransformsType.Normalize, dict(a=-1, b=1)),
]
data_config.eval_transforms = [
    (TransformsType.Resize, dict(size=(32, 32))),
    (TransformsType.ToFloat, dict()),
    (TransformsType.Normalize, dict(a=-1, b=1)),
]
