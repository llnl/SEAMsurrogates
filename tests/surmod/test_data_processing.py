import numpy as np
import pytest

from surmod import data_processing


def test_scale_inputs_uses_training_statistics_only():
    x_train = np.array(
        [
            [1.0, 10.0],
            [3.0, 14.0],
            [5.0, 18.0],
        ]
    )
    x_test = np.array(
        [
            [7.0, 22.0],
            [9.0, 26.0],
        ]
    )

    scaled_train, scaled_test = data_processing.scale_inputs(
        x_train,
        x_test,
        scale_inputs=True,
    )

    assert np.allclose(np.mean(scaled_train, axis=0), np.zeros(2))
    assert np.allclose(np.std(scaled_train, axis=0), np.ones(2))

    expected_test = (x_test - np.mean(x_train, axis=0)) / np.std(x_train, axis=0)
    assert np.allclose(scaled_test, expected_test)


def test_scale_inputs_handles_zero_variance_columns():
    x_train = np.array(
        [
            [2.0, 5.0],
            [2.0, 7.0],
            [2.0, 9.0],
        ]
    )
    x_test = np.array([[2.0, 11.0]])

    scaled_train, scaled_test = data_processing.scale_inputs(
        x_train,
        x_test,
        scale_inputs=True,
    )

    assert np.allclose(scaled_train[:, 0], np.zeros(3))
    assert np.allclose(scaled_test[:, 0], np.zeros(1))


def test_scale_inputs_rejects_incompatible_shapes():
    x_train = np.array([[1.0, 2.0]])
    x_test = np.array([[3.0, 4.0, 5.0]])

    with pytest.raises(ValueError, match="same number of features"):
        data_processing.scale_inputs(x_train, x_test, scale_inputs=True)
