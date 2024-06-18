import numpy as np

from modules.layers.activations import ReLU


def test_relu():
    relu = ReLU()

    result = relu(np.array([[0, 1, -1]]))

    assert (result == np.array([0, 1, 0])).all()
    assert (relu._inputs_cache == np.array([[0, 1, -1]])).all()


def test_relu_compute_gradient():
    relu = ReLU()
    relu._inputs_cache = np.array([[0, 1, -1]])

    result = relu.compute_backward_gradient(np.array([[2, 3, 4]]))

    assert (result == np.array([[0, 3, 0]])).all()
