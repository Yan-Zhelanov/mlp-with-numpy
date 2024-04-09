import cv2
import os

from utils.common_functions import read_dataframe_file
from utils.enums import SetType


class EmotionsDataset:
    """A class for the Emotions dataset. This class defines how data is loaded."""

    def __init__(self, config, set_type: SetType, transforms=None, target_transforms=None):
        self.config = config
        self.set_type = set_type
        self.transforms = transforms
        self.target_transforms = target_transforms

        # Reading an annotation file that contains the image path, set_type, and target values for the entire dataset
        annotation = read_dataframe_file(os.path.join(config.path_to_data, config.annot_filename))
        # Filter the annotation file according to set_type
        self.annotation = annotation[annotation['set'] == self.set_type.name]

        # TODO: Get image paths from 'path' column
        self._paths = ...

        # TODO: Make mapping from the 'target' column to int values using self.config.label_mapping
        #       When set_type is SetType.test, the target does not exist.
        self._targets = ...

        self._ohe_targets = None
        if set_type != SetType.test and self.target_transforms is not None:
            self._ohe_targets = self.target_transforms(self._targets)

    @property
    def labels(self):
        return self._targets

    def __len__(self):
        # TODO: Return the number of samples in the dataset
        raise NotImplementedError

    def __getitem__(self, idx):
        """Loads and returns one sample from a dataset with the given idx index.

        Returns:
            A dict with the following data:
                {
                    'image: image (numpy.ndarray),
                    'target': target (int),
                    'ohe_target': ohe target (numpy.ndarray),
                    'path': image path (str)
                }
        """
        # TODO:
        #  1) Read an image by its path using a given index with OpenCV or PIL, convert it to GRAYSCALE mode
        #  2) Extract the corresponding label
        #  3) Call the self.transforms functions for the image (if needed)
        #  4) Return the image, the corresponding label, one-hot encoded target (from self._ohe_targets or None)
        #  and image path as a dictionary with keys "image", "target", "ohe_target" and "path", respectively
        raise NotImplementedError