import numpy as np
import pytest

from modules.utils.parameter_initialization import ParameterInitializator

_SIGMA = 0.01
_EPSILON = 0.01
_MU = 0
_TOLERANCE = 0.01


class MockInitType:
    name = 'normal'


class MockModelConfig:
    INIT_TYPE = MockInitType()
    ZERO_BIAS = False
    INIT_KWARGS = {
        'mu': _MU,
        'sigma': _SIGMA,
        'epsilon': _EPSILON,
    }


@pytest.fixture
def init():
    config = MockModelConfig()
    return ParameterInitializator(config)


def test_get_init_zeros(init: ParameterInitializator):
    param_shape = (3, 3)

    result = init.get_init_zeros(param_shape)

    assert np.array_equal(result, np.zeros(param_shape))


def test_get_init_normal(init: ParameterInitializator):
    param_shape = (3, 3)

    result = init.get_init_normal(param_shape)

    assert result.shape == param_shape
    assert np.allclose(result.mean(), _MU, atol=_TOLERANCE)
    assert np.allclose(result.std(), _SIGMA, atol=_TOLERANCE)


def test_get_init_uniform(init: ParameterInitializator):
    param_shape = (3, 3)
    epsilon = 0.1

    result = init.get_init_uniform(param_shape)

    assert result.shape == param_shape
    assert np.all(result >= -epsilon)
    assert np.all(result <= epsilon)


def test_get_init_he(init: ParameterInitializator):
    param_shape = (3, 3)
    std = np.sqrt(2.0 / param_shape[0])
    he_tolerance = 0.8

    result = init.get_init_he(param_shape)

    assert result.shape == param_shape
    assert np.allclose(result.mean(), _MU, atol=he_tolerance)
    assert np.allclose(result.std(), std, atol=he_tolerance)


def test_get_init_xavier(init: ParameterInitializator):
    param_shape = (3, 3)
    epsilon = np.sqrt(1.0 / param_shape[0])

    result = init.get_init_xavier(param_shape)

    assert result.shape == param_shape
    assert np.all(result >= -epsilon)
    assert np.all(result <= epsilon)


def test_get_init_xavier_normalized(init: ParameterInitializator):
    param_shape = (3, 3)
    epsilon = np.sqrt(6.0 / (param_shape[0] + param_shape[1]))

    result = init.get_init_xavier_normalized(param_shape)

    assert result.shape == param_shape
    assert np.all(result >= -epsilon)
    assert np.all(result <= epsilon)
