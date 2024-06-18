import numpy as np

from modules.layers.activations import LeakyReLU, ReLU, Sigmoid, Tanh


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


def test_leaky_relu():
    leaky_relu = LeakyReLU(0.1)

    result = leaky_relu(np.array([[0, 1, -1]]))

    assert (result == np.array([0, 1, -0.1])).all()
    assert (leaky_relu._inputs_cache == np.array([[0, 1, -1]])).all()


def test_leaky_relu_compute_gradient():
    leaky_relu = LeakyReLU(0.1)
    leaky_relu._inputs_cache = np.array([[0, 1, -1]])

    result = leaky_relu.compute_backward_gradient(np.array([[2, 3, 4]]))

    assert (result == np.array([[0.2, 3, 0.4]])).all()


def test_sigmoid():
    sigmoid = Sigmoid()

    result = sigmoid(np.array([[0, 1, -1]]))

    assert np.allclose(result, np.array([[0.5, 0.73105858, 0.26894142]]))
    assert (sigmoid._inputs_cache == np.array([[0, 1, -1]])).all()
