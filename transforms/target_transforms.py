import numpy as np


class OneHotEncoding:
    """Creates matrix of one-hot encoding vectors for input targets"""

    def __init__(self, num_classes: int):
        self.num_classes = num_classes

    def __call__(self, targets: np.ndarray):
        """
        One-hot encoding vector representation:
            t_i^(k) = 1 if k = t_i otherwise  0,

            where:
                - k in [0, self.k-1],
                - t_i - target class of i-sample.
        Args:
            targets: np.ndarray
        """
        # TODO: Implement this function, it is possible to do it without loop using numpy
        raise NotImplementedError