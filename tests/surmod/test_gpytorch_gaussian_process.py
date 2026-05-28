import os
import numpy as np
import pytest
import torch

from surmod import gpytorch_gaussian_process as gpgp


@pytest.fixture
def train_data():
    x = np.linspace(0.0, 1.0, 8).reshape(-1, 1)
    y = np.sin(2 * np.pi * x).ravel()
    return x, y


@pytest.fixture
def test_data():
    x = np.linspace(0.1, 0.9, 4).reshape(-1, 1)
    y = np.sin(2 * np.pi * x).ravel()
    return x, y


@pytest.fixture
def surrogate(train_data, test_data):
    x_train, y_train = train_data
    x_test, y_test = test_data
    gp = gpgp.GPSurrogate(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        kernel="rbf",
        optimization_restarts=1,
    )
    gp.model.eval()
    gp.mll.eval()
    return gp


def test_init_shapes_and_types(train_data, test_data):
    x_train, y_train = train_data
    x_test, y_test = test_data

    gp = gpgp.GPSurrogate(
        x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test
    )

    assert isinstance(gp.x_train, torch.Tensor)
    assert isinstance(gp.y_train, torch.Tensor)
    assert gp.x_train.dtype == torch.float64
    assert gp.y_train.dtype == torch.float64
    assert gp.x_train.shape == (8, 1)
    assert gp.y_train.shape == (8, 1)
    assert gp.x_test.shape == (4, 1)
    assert gp.y_test.shape == (4, 1)
    assert gp.model is not None
    assert gp.mll is not None


@pytest.mark.parametrize(
    "kernel_name,expected_type",
    [
        ("rbf", gpgp.RBFKernel),
        ("matern", gpgp.MaternKernel),
        ("periodic", gpgp.PeriodicKernel),
    ],
)
def test_get_covar_module_kernel_types(train_data, kernel_name, expected_type):
    x_train, y_train = train_data
    gp = gpgp.GPSurrogate(x_train=x_train, y_train=y_train, kernel=kernel_name)
    covar = gp._get_covar_module()

    assert isinstance(covar, gpgp.ScaleKernel)
    assert isinstance(covar.base_kernel, expected_type)


def test_invalid_kernel_raises(train_data):
    x_train, y_train = train_data
    with pytest.raises(ValueError, match="kernel must be"):
        gpgp.GPSurrogate(x_train=x_train, y_train=y_train, kernel="bad_kernel")


def test_predict_uses_x_test_when_no_argument(surrogate):
    mean, std = surrogate.predict()

    assert isinstance(mean, np.ndarray)
    assert isinstance(std, np.ndarray)
    assert mean.shape == (4,)
    assert std.shape == (4,)
    assert np.all(std >= 0.0)


def test_predict_with_explicit_x_numpy(surrogate):
    x = np.array([[0.2], [0.4], [0.6]])
    mean, std = surrogate.predict(x)

    assert mean.shape == (3,)
    assert std.shape == (3,)


def test_predict_with_explicit_x_tensor(surrogate):
    x = torch.tensor([[0.2], [0.4]], dtype=torch.float64)
    mean, std = surrogate.predict(x)

    assert mean.shape == (2,)
    assert std.shape == (2,)


def test_predict_raises_without_any_inputs(train_data):
    x_train, y_train = train_data
    gp = gpgp.GPSurrogate(x_train=x_train, y_train=y_train)
    gp.model.eval()

    with pytest.raises(ValueError, match="No prediction data provided"):
        gp.predict()


def test_evaluate_returns_metrics(surrogate):
    results = surrogate.evaluate()

    assert set(results.keys()) == {"mse", "rmse", "coverage", "mean", "std"}
    assert isinstance(results["mse"], float)
    assert isinstance(results["rmse"], float)
    assert isinstance(results["coverage"], float)
    assert results["mean"].shape == (4,)
    assert results["std"].shape == (4,)
    assert 0.0 <= results["coverage"] <= 1.0


def test_evaluate_raises_without_test_data(train_data):
    x_train, y_train = train_data
    gp = gpgp.GPSurrogate(x_train=x_train, y_train=y_train)

    with pytest.raises(ValueError, match="x_test and y_test must be provided"):
        gp.evaluate()


def test_compute_max_error():
    output = np.array([1.0, 4.0, 2.0])
    target = np.array([1.5, 1.0, 2.5])
    inputs = np.array([[10.0], [20.0], [30.0]])

    max_err, x_at_max = gpgp.GPSurrogate.compute_max_error(output, target, inputs)

    assert max_err == pytest.approx(3.0)
    assert np.array_equal(x_at_max, np.array([20.0]))


def test_sample_posterior_shape_with_explicit_x(surrogate):
    x = np.array([[0.2], [0.5], [0.8]])
    samples = surrogate.sample_posterior(x=x, n_samples=5)

    assert isinstance(samples, np.ndarray)
    assert samples.shape == (5, 3)


def test_sample_posterior_uses_x_test(surrogate):
    samples = surrogate.sample_posterior(n_samples=2)
    assert samples.shape == (2, 4)


def test_sample_posterior_raises_without_inputs(train_data):
    x_train, y_train = train_data
    gp = gpgp.GPSurrogate(x_train=x_train, y_train=y_train)

    with pytest.raises(ValueError, match="No input data provided"):
        gp.sample_posterior()


def test_posterior_gradient_shape(surrogate):
    x = np.array([[0.25], [0.75]])
    grad = surrogate.posterior_gradient(x)

    assert isinstance(grad, np.ndarray)
    assert grad.shape == (2, 1)


def test_plot_test_predictions_saves_file(surrogate, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    surrogate.plot_test_predictions("unit_test_plot")

    plot_dir = tmp_path / "plots"
    files = list(plot_dir.glob("unit_test_plot_test_predictions_*.png"))

    assert plot_dir.exists()
    assert len(files) == 1


def test_plot_test_predictions_raises_without_test_data(train_data):
    x_train, y_train = train_data
    gp = gpgp.GPSurrogate(x_train=x_train, y_train=y_train)

    with pytest.raises(ValueError, match="x_test and y_test must be provided"):
        gp.plot_test_predictions()


def test_fit_runs_on_small_dataset(train_data, test_data):
    x_train, y_train = train_data
    x_test, y_test = test_data

    gp = gpgp.GPSurrogate(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        optimization_restarts=1,
    )
    gp.fit()

    mean, std = gp.predict()
    assert mean.shape == (4,)
    assert std.shape == (4,)
