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


def test_linear_compute_gradient():
    linear = Linear(2, 3)
    linear._weights = np.array([[1, 2], [2, 3], [3, 4]])
    linear._inputs_cache = np.array([[1, 2], [3, 4]])
    expected_result = np.array([[38, 56], [56, 83]])
    expected_weights_gradient = np.array([[29, 42], [33, 48], [37, 54]])
    expected_bias_gradient = np.array([13, 15, 17])

    result = linear.compute_backward_gradient(
        np.array([[5, 6, 7], [8, 9, 10]]),
    )

    assert (result == expected_result).all()
    assert (linear._weights_gradient == expected_weights_gradient).all()
    assert (linear._bias_gradient == expected_bias_gradient).all()
