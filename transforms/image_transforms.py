import cv2
import numpy as np
import sys
from typing import Union


class Normalize:
    """Transforms image by scaling each pixel to a range [a, b]"""

    def __init__(self, a: Union[float, int] = -1, b: Union[float, int] = 1):
        self.a = a
        self.b = b

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        Args:
            img: np.ndarray, all pixels in [0, 1]

        Returns:
             normalized_img (numpy.array)
        """
        # TODO: implement data normalization
        #       normalized_img = a + (b - a) * img,
        #       where a = self.a, b = self.b
        raise NotImplementedError


class Standardize:
    """Standardizes image with mean and std."""

    def __init__(self, mean: Union[float, list, tuple], std: Union[float, list, tuple]):
        self.mean = np.array(mean)
        self.std = np.array(std)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        Args:
            img: np.ndarray, all pixels in [0, 1]

        Returns:
             standardized_img (numpy.array)
        """
        # TODO: implement data standardization
        #       standardized_x = (x - self.mean) / self.std
        raise NotImplementedError


class ToFloat:
    """Convert image from uint to float and scale it to [0, 1]"""

    def __init__(self):
        pass

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        Args:
            img: np.ndarray

        Returns:
             float_img (numpy.array)
        """
        # TODO: implement converting to float and scaling in [0, 1]
        raise NotImplementedError


class Resize:
    """Image resize"""

    def __init__(self, size: Union[int, tuple, list]):
        self.size = size

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        Args:
            img: np.ndarray

        Returns:
             resized_img (numpy.array)
        """
        # TODO: Implement resizing with cv2.resize
        raise NotImplementedError


class Sequential:
    """Composes several transforms together."""

    def __init__(self, transform_list):
        self.transforms = [
            getattr(sys.modules[__name__], transform.name)(**params) for transform, params in transform_list
        ]

    def __call__(self, img: np.ndarray) -> np.ndarray:
        # consistent self.transforms application
        img_aug = img.copy()
        for transform in self.transforms:
            img_aug = transform(img_aug)
        return img_aug