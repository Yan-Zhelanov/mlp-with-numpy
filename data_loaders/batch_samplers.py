from typing import Iterator

import numpy as np
from numpy import typing as npt

from dataset.emotions_dataset import EmotionsDataset


class BaseSampler(object):
    """Class for sampling dataset indexes."""

    def __init__(
        self, dataset: EmotionsDataset, shuffle: bool, **kwargs,
    ) -> None:
        self._indices = np.arange(len(dataset))
        self._labels = dataset.labels
        self._shuffle = shuffle

    def _get_indices(self) -> npt.NDArray[np.integer]:
        raise NotImplementedError

    def __iter__(self) -> Iterator[np.integer]:
        """Iterating over indices"""
        indices = self._get_indices()
        if self._shuffle:
            indices = np.random.permutation(indices)
        return iter(indices)


class DefaultSampler(BaseSampler):
    def _get_indices(self) -> npt.NDArray[np.integer]:
        return self._indices


class UpsamplingSampler(BaseSampler):
    def _get_indices(self) -> npt.NDArray[np.integer]:
        """Upsampling Minority Class.

        Upsampling is a technique used to create additional data points of the
        minority class to balance class labels. This is usually done by
        duplicating existing samples or creating new ones.

        Returns:
            npt.NDArray[np.integer]: Indices of the samples.
        """
        unique_labels, counts = np.unique(self._labels, return_counts=True)
        label_indices = {
            label: np.where(self._labels == label)[0]
            for label in unique_labels
        }
        max_samples = np.max(counts)
        indices: list[int] = []
        for label_indices_for_class in label_indices.values():
            indices.extend(label_indices_for_class)
            if len(label_indices_for_class) < max_samples:
                difference = max_samples - len(label_indices_for_class)
                indices.extend(
                    np.random.choice(label_indices_for_class, difference),
                )
        return np.array(indices)
