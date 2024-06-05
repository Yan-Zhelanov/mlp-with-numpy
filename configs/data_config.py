import os

from utils.enums import SamplerType, TransformsType

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))


class DataConfig(object):
    PATH_TO_DATA = os.path.join(ROOT_DIR, 'data', 'emotion_detection')
    ANNOTATION_FILE = 'data_info.csv'
    label_mapping = {
        'angry': 0,
        'disgusted': 1,
        'fearful': 2,
        'happy': 3,
        'neutral': 4,
        'sad': 5,
        'surprised': 6,
    }
    sampler_type = SamplerType.DEFAULT
    train_transforms = [
        (TransformsType.RESIZE, dict(size=(32, 32))),
        (TransformsType.TO_FLOAT, dict()),
        (TransformsType.NORMALIZE, dict(a=-1, b=1)),
    ]
    eval_transforms = [
        (TransformsType.RESIZE, dict(size=(32, 32))),
        (TransformsType.TO_FLOAT, dict()),
        (TransformsType.NORMALIZE, dict(a=-1, b=1)),
    ]
