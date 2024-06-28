from typing import TypedDict

import numpy as np
from numpy import typing as npt

from data_loaders import batch_samplers
from dataset.emotions_dataset import DataSample, EmotionsDataset
from utils.enums import SamplerType


class BatchType(TypedDict):
    images: npt.NDArray[np.integer | np.floating]
    targets: npt.NDArray[np.integer]
    ohe_targets: npt.NDArray[np.integer | np.floating]


class DataType(TypedDict):
    image: npt.NDArray[np.integer | np.floating]
    label: int


class DataLoader(object):
    """Provide an iterable over the given dataset."""

    def __init__(
        self,
        dataset: EmotionsDataset,
        batch_size: int,
        sampler: SamplerType = SamplerType.DEFAULT,
        shuffle: bool = False,
        drop_last: bool = False,
        **kwargs,
    ) -> None:
        self._dataset = dataset
        self._batch_size = batch_size
        self._drop_last = drop_last
        self._num_samples = len(self._dataset)
        self._sampler = getattr(
            batch_samplers, f'{sampler.name.title()}Sampler',
        )(self._dataset, shuffle, **kwargs)

    def __iter__(self):
        """Return a batch at each iteration."""
        batch_data = []
        for index in self._sampler:
            batch_data.append(self._dataset[index])
            if len(batch_data) == self._batch_size:
                yield self._collate_fn(batch_data)
                batch_data = []
        if not self._drop_last and len(batch_data) > 0:
            yield self._collate_fn(batch_data)

    @staticmethod
    def _collate_fn(batch: list[DataSample]) -> BatchType:
        """Combine a list of samples into a dictionary.

        Example:
            batch = [
                {'image': img_1, 'label': 0},
                {'image': img_2, 'label': 1},
                {'image': img_3, 'label': 1},
                {'image': img_4, 'label': 2},
            ]
            batch = {
                'image': np.array([img_1, img_2, img_3, img_4]),
                'label': np.array([0, 1, 1, 2])
            }

        Args:
            batch_data: list of dicts.

        Returns:
            BatchType: dict with labels and images.
        """
        new_batch: dict[str, list] = {
            'images': [],
            'targets': [],
            'ohe_targets': [],
        }
        for observation in batch:
            new_batch['images'].append(observation.image)
            new_batch['targets'].append(observation.target)
            new_batch['ohe_targets'].append(observation.ohe_target)
        return {
            'images': np.array(new_batch['images']),
            'targets': np.array(new_batch['targets']),
            'ohe_targets': np.array(new_batch['ohe_targets']),
        }
