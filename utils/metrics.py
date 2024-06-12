import numpy as np
from numpy import typing as npt

from constants.types import TargetsType


def get_accuracy_score(
    targets: TargetsType, predictions: TargetsType,
) -> float:
    """Get accuracy score.

    The formula is as follows:
        accuracy = (1 / N) Σ(i=0 to N-1) I(y_i == t_i),

        where:
            - N - number of samples,
            - y_i - predicted class of i-sample,
            - t_i - correct class of i-sample,
            - I(y_i == t_i) - indicator function.

    Args:
        targets: true labels
        predictions: predicted class

    Returns:
        float: Accuracy score.
    """
    return np.mean(targets == predictions)


def accuracy_score_per_class(
    targets: TargetsType, predictions: TargetsType,
) -> list | np.ndarray | dict:
    """Accuracy score for each class.

    The formula is as follows:
        accuracy_k = (1 / N_k) Σ(i=0 to N) I(y_i == t_i) * I(t_i == k)

        where:
            - N_k -  number of k-class elements,
            - y_i - predicted class of i-sample,
            - t_i - correct class of i-sample,
            - I(y_i == t_i), I(t_i == k) - indicator function.

    Args:
        targets: true labels
        predictions: predicted class

    Returns:
        accuracy for each class: list, np.ndarray or dict
    """
    # TODO: Implement computation of accuracy for each class
    raise NotImplementedError


def balanced_accuracy_score(
    targets: TargetsType, predictions: TargetsType,
) -> float:
    """Balanced accuracy score.

    The formula is as follows:
        balanced_accuracy = (1 / K) Σ(k=0 to K-1) accuracy_k,
        accuracy_k = (1 / N_k) Σ(i=0 to N) I(y_i == t_i) * I(t_i == k)

        where:
            - K - number of classes,
            - N_k - number of k-class elements,
            - accuracy_k - accuracy for k-class,
            - y_i - predicted class of i-sample,
            - t_i - correct class of i-sample,
            - I(y_i == t_i), I(t_i == k) - indicator function.

    Args:
        targets: true labels
        predictions: predicted class
    """
    # TODO: Implement computation of balanced accuracy
    raise NotImplementedError


def get_confusion_matrix(
    targets: TargetsType, predictions: TargetsType,
) -> npt.NDArray[np.float64]:
    """Get confusion matrix.

    Confusion matrix C with shape KxK:
        c[i, j] - number of observations known to be in class i and predicted
        to be in class j,
        where:
            - K is the number of classes.

    Args:
        targets: Labels.
        predictions: Predicted classes.

    Returns:
        np.ndarray: Confusion matrix.
    """
    count_classes = len(np.unique(targets))
    confusion_matrix = np.zeros((count_classes, count_classes))
    np.add.at(confusion_matrix, (targets.astype(int), predictions), 1)
    return confusion_matrix
