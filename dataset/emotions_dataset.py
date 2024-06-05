import os
from dataclasses import dataclass

import cv2
import numpy as np
from numpy import typing as npt

from configs.data_config import DataConfig
from transforms.image_transforms import ImageTransform
from transforms.target_transforms import TargetsType, TargetTransform
from utils.common_functions import read_dataframe_file
from utils.enums import SetType


@dataclass(frozen=True)
class DataSample(object):
    image: npt.NDArray[np.integer | np.float_]
    target: int
    ohe_target: TargetsType | None
    path: str


class EmotionsDataset(object):
    """A class for loading the emotions dataset."""

    def __init__(
        self,
        config: DataConfig,
        set_type: SetType,
        transforms: ImageTransform | None = None,
        target_transforms: TargetTransform | None = None,
    ) -> None:
        self._config = config
        self._set_type = set_type
        self._transforms = transforms
        self._target_transforms = target_transforms
        # Reading an annotation file that contains the image path, set_type,
        # and target values for the entire dataset.
        annotation = read_dataframe_file(
            os.path.join(
                config.PATH_TO_DATA, config.ANNOTATION_FILE,
            ),
        )
        # Filter the annotation file according to set_type.
        self._annotation = annotation[
            annotation['set'] == self._set_type.name.lower()
        ]
        self._paths = [
            os.path.join(config.PATH_TO_DATA, path)
            for path in self._annotation['path'].to_list()
        ]
        self._targets: npt.NDArray[np.int32] = np.array(
            [] if set_type is SetType.TEST
            else self._annotation['target'].map(
                self._config.label_mapping,
            ).to_list(),
        )
        self._ohe_targets = None
        is_test_set = set_type is SetType.TEST
        if not is_test_set and self._target_transforms is not None:
            self._ohe_targets = self._target_transforms.transform(
                self._targets,
            )

    @property
    def labels(self):
        return self._targets

    def __len__(self):
        return len(self._paths)

    def __getitem__(self, index: int) -> DataSample:
        """Load and return one sample from a dataset with the given idx index.

        Args:
            index: The index of the sample to load.

        Returns:
            DataSample: A dataclass with the image, target, one-hot encoded
                target, and image path.
        """
        image: npt.NDArray[np.integer | np.float_] = cv2.imread(
            self._paths[index], cv2.IMREAD_GRAYSCALE,
        )
        target = -1
        if self._targets.size:
            target = self._targets[index]
        if self._transforms is not None:
            image = self._transforms.transform(image)
        ohe_target: TargetsType | None = None
        if self._ohe_targets is not None:
            ohe_target = self._ohe_targets[index]
        return DataSample(
            image=image,
            target=target,
            ohe_target=ohe_target,
            path=self._paths[index],
        )
