import sys
from abc import ABC, abstractmethod
from typing import Union

import cv2
import numpy as np
from numpy import typing as npt

from utils.enums import TransformsType

_MAX_VALUE = 255

_ImageType = npt.NDArray[np.integer | np.floating]


class ImageTransform(ABC):
    @abstractmethod
    def transform(self, image: _ImageType) -> _ImageType:
        """Transform image.

        Args:
            image: The input image to transform.

        Returns:
            Image: Transformed image.
        """


class Normalize(ImageTransform):
    """Transforms image by scaling each pixel to a range [a, b]"""

    def __init__(
        self, a: Union[float, int] = -1, b: Union[float, int] = 1,
    ) -> None:
        self.a = a
        self.b = b

    def transform(
        self, image: npt.NDArray[np.integer | np.float_],
    ) -> npt.NDArray[np.integer | np.float_]:
        """
        Args:
            image: np.ndarray, all pixels in [0, 1]

        Returns:
             normalized_image (numpy.array)
        """
        # TODO: implement data normalization
        #       normalized_image = a + (b - a) * image,
        #       where a = self.a, b = self.b
        raise NotImplementedError


class Standardize(ImageTransform):
    """Standardizes image with mean and std."""

    def __init__(
        self, mean: Union[float, list, tuple], std: Union[float, list, tuple],
    ) -> None:
        self.mean = np.array(mean)
        self.std = np.array(std)

    def transform(
        self, image: npt.NDArray[np.integer | np.float_],
    ) -> npt.NDArray[np.integer | np.float_]:
        """
        Args:
            image: np.ndarray, all pixels in [0, 1]

        Returns:
             standardized_image (numpy.array)
        """
        # TODO: implement data standardization
        #       standardized_x = (x - self.mean) / self.std
        raise NotImplementedError


class ToFloat(ImageTransform):
    """Convert image from uint to float and scale it to [0, 1]."""

    def __init__(self) -> None:
        """Initialize the transform."""

    def transform(self, image: _ImageType) -> _ImageType:
        """Transform image by converting it to float and scaling it to [0, 1].

        Args:
            image: np.ndarray

        Returns:
            _ImageType: The image in the range [0, 1].
        """
        return image.astype(np.float64) / _MAX_VALUE


class Resize(ImageTransform):
    """Image resize."""

    def __init__(self, size: tuple[int, int]) -> None:
        self._size = size

    def transform(self, image: _ImageType) -> _ImageType:
        """Transform image by resizing it.

        Args:
            image (npt.NDArray[npt.integer | npt.floating]): The input image to
                transform.

        Returns:
            npt.NDArray[npt.floating]: The image in the range [0, 1].
        """
        return cv2.resize(image, self._size)


class Sequential(ImageTransform):
    """Compose several transforms together."""

    def __init__(
        self, transform_list: list[tuple[TransformsType, dict]],
    ) -> None:
        self._transforms: list[ImageTransform] = [
            getattr(sys.modules[__name__], transform.name.lower())(**params)
            for transform, params in transform_list
        ]

    def transform(
        self, image: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.integer | np.float_]:
        transformed_image = image.copy()
        for transform in self._transforms:
            transformed_image = transform.transform(transformed_image)
        return transformed_image
