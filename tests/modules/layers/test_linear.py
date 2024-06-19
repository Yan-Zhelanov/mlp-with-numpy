import numpy as np

from modules.layers.linear import Linear


def test_linear():
    linear = Linear(2, 3)
    linear._weights = np.array([[1, 2], [2, 3], [3, 4]])
    expected_result = np.array([[5, 8, 11], [11, 18, 25]])
    expected_cache = np.array([[1, 2], [3, 4]])

    result = linear(np.array([[1, 2], [3, 4]]))

    assert (result == expected_result).all()
    assert (linear._inputs_cache == expected_cache).all()
