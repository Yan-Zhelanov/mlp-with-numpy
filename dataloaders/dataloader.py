import numpy as np

from dataloaders import batch_samplers
from utils.enums import SamplerType


class Dataloader:
    """Provides an iterable over the given dataset."""

    def __init__(self, dataset, batch_size: int, sampler: SamplerType = SamplerType.Default,
                 shuffle: bool = False, drop_last: bool = False, **kwargs):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last

        self.num_samples = len(self.dataset)
        self.sampler = getattr(batch_samplers, sampler.name + 'Sampler')(self.dataset, shuffle, **kwargs)

    def __iter__(self):
        """Returns a batch at each iteration."""
        # TODO:
        #  1) For every idx in self.sampler:
        #       - add dataset[idx] to batch_data
        #       - if length of batch_data is equal to the batch_size:
        #               - return Dataloader._collate_fn(batch_data) with yield
        #               - empty the batch_data list
        #  2) If self.drop_last is False and list of batch_data is not empty, return last batch with yield
        batch_data = []
        raise NotImplementedError

    @staticmethod
    def _collate_fn(batch_data: list) -> dict:
        """Combines a list of samples into a dictionary

        Example:
            batch_data = [
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
            batch_data: list of dicts

        Returns:
            batch: dict
        """
        # TODO:
        #  1) Get the first item from batch_data to determine which keys each item has
        #  2) For each key:
        #       - combine elements from batch_data with that key
        #       - concatenate data to np.ndarray
        #  3) Return dict with these keys
        raise NotImplementedError