import numpy as np

from modules.layers.softmax import calculate_softmax


def test_calculate_softmax():
    logits = np.array([[1, 2, 3], [5, 5, 6]])
    expected_result = np.array([
        [0.09003057, 0.24472847, 0.66524096],
        [0.21194156, 0.21194156, 0.57611688],
    ])

    result = calculate_softmax(logits)

    assert np.allclose(result, expected_result), (
        f'{result} != {expected_result}'
    )
