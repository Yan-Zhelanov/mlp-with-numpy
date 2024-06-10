import numpy as np

from dataset.emotions_dataset import EmotionsDataset


class BaseSampler(object):
    """Class for sampling dataset indexes."""

    def __init__(
        self, dataset: EmotionsDataset, shuffle: bool, **kwargs,
    ) -> None:
        self._indices = np.arange(len(dataset))
        self._labels = dataset.labels
        self._shuffle = shuffle

    def _get_indices(self) -> np.ndarray | list:
        raise NotImplementedError

    def __iter__(self):
        """Iterating over indices"""
        indices = self._get_indices()
        if self._shuffle:
            indices = np.random.permutation(indices)
        return iter(indices)


class DefaultSampler(BaseSampler):
    def _get_indices(self):
        return self._indices


class UpsamplingSampler(BaseSampler):
    def _get_indices(self):
        """Upsampling Minority Class.

        Upsampling is a technique used to create additional data points of the minority class to balance class labels.
            This is usually done by duplicating existing samples or creating new ones

        # TODO:
            1) For each class get the indices of its elements using self.labels
            2) Get the maximum number of elements by class
            3) For each class:
                - add all indices of the class to the indices-list
                - if the number of samples is less than the maximum number, randomly sample indices from this class
                so that the total number of indices equals the maximum number
            4) Return np.ndarray or list of indexes
        """
        indices = []
        raise NotImplementedError
