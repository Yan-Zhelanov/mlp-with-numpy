import numpy as np
from numpy import typing as npt


def calculate_softmax(
    logits: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """Calculate softmax function.

    The formula for numerically stable softmax function is:
    y_j = e^(z_j - c) / Σ(i=0 to K-1) e^(z_i - c)

    where:
    - y_j is the softmax probability of class j,
    - z_j is the model output (logits) for class j before softmax,
    - K is the total number of classes,
    - c is maximum(z),
    - Σ denotes summation.

    Args:
        logits (npt.NDArray[np.floating]): The model output before softmax.

    Returns:
        npt.NDArray[np.floating]: The softmax probabilities.
    """
    max_logits = np.max(logits, axis=1, keepdims=True)
    logits -= max_logits
    numerator = np.exp(logits)
    denominator = np.sum(numerator, axis=1, keepdims=True)
    return numerator / denominator
