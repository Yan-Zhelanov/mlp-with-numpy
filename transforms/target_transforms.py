from abc import ABC, abstractmethod

import numpy as np
from numpy import typing as npt

TargetsType = npt.NDArray[np.integer | np.floating]


class TargetTransform(ABC):
    @abstractmethod
    def transform(self, targets: TargetsType) -> TargetsType:
        """Transform targets.

        Args:
            targets (_TargetsType): The input targets to transform.

        Returns:
            _TargetsType: Transformed targets.
        """


class OneHotEncoding(TargetTransform):
    """Create matrix of one-hot encoding vectors for input targets."""

    def __init__(self, num_classes: int) -> None:
        self._num_classes = num_classes

    def transform(self, targets: TargetsType) -> TargetsType:
        """Transform targets into one-hot encoding vector representation.

        t_i^(k) = 1 if k = t_i otherwise  0,
        where:
        - k in [0, self.k-1],
        - t_i - target class of i-sample.

        Args:
            targets (_TargetsType): The input targets to transform.

        Returns:
            _TargetsType: Transformed targets.
        """
        return np.eye(self._num_classes)[targets]
