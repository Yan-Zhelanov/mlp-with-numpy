import os
from typing import Any

from utils.enums import SamplerType, TransformsType

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))


class DataConfig:
    PATH_TO_DATA = os.path.join(ROOT_DIR, 'data')
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
        (TransformsType.RESIZE, {'size': (32, 32)}),
        (TransformsType.TO_FLOAT, {}),
        (TransformsType.NORMALIZE, {'min_value': -1, 'max_value': 1}),
    ]
    eval_transforms: list[tuple[TransformsType, dict]] = [
        (TransformsType.RESIZE, {'size': (32, 32)}),
        (TransformsType.TO_FLOAT, {}),
        (TransformsType.NORMALIZE, {'min_value': -1, 'max_value': 1}),
    ]

    def get_all_hyperparameters(self) -> dict[str, Any]:
        """Get all hyperparameters for the model.

        Returns:
            dict[str, Any]: dictionary with all hyperparameters.
        """
        return {
            'path_to_data': self.PATH_TO_DATA,
            'annotation_file': self.ANNOTATION_FILE,
            'label_mapping': self.label_mapping,
            'sampler_type': self.sampler_type,
            'train_transforms': self.train_transforms,
            'eval_transforms': self.eval_transforms,
        }
