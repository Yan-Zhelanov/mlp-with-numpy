import numpy as np

from modules.layers.activations import ReLU


def test_relu():
    relu = ReLU()

    result = relu(np.array([[0, 1, -1]]))

    assert (result == np.array([0, 1, 0])).all()
    assert (relu._inputs_cache == np.array([[0, 1, -1]])).all()
