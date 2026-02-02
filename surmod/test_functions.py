import numpy as np
import numpy.typing as npt
import torch
from botorch.test_functions.synthetic import SyntheticTestFunction
from typing import Optional, List, Tuple, Union

class Parabola_synth_test_func(SyntheticTestFunction):
    """Parabola test function.

    Default is bivariate parabola evaluated on [-8,8]x[-8,8].
    """

    _check_grad_at_opt: bool = False

    def __init__(
        self,
        dim: int = 2,
        noise_std: Optional[float] = None,
        negate: bool = True,
        bounds: Optional[List[Tuple[float, float]]] = None,
    ) -> None:
        """
        Args:
            dim: Dimensionality of the parabola.
            noise_std: Standard deviation of the observation noise.
            negate: If True, negate the function.
            bounds: Custom bounds for the function specified as (lower, upper) pairs.
        """
        self.dim = dim
        bounds = [(-8, 8) for _ in range(self.dim)]
        self.continuous_inds = list(range(dim))
        self.discrete_inds = []
        self.categorical_inds = []
        super().__init__(noise_std=noise_std, negate=negate, bounds=bounds)

    def _evaluate_true(self, X: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        if isinstance(X, torch.Tensor):
            result = -torch.sum(X**2, dim=1) if X.ndim > 1 else -torch.sum(X**2)
        elif isinstance(X, np.ndarray):
            result = -np.sum(X**2, axis=1) if X.ndim > 1 else -np.sum(X**2)
            result = torch.from_numpy(result)
        else:
            raise TypeError("Input must be a torch.Tensor or numpy.ndarray.")

        return -result if self.negate else result
    
class Borehole_synth_test_func(SyntheticTestFunction):
    """
    Borehole test function.

    This is the 8 dimensional borehole function used as a test case
    in computer experiments. Implementation follows the definition from
    Sonja Surjanovic and Derek Bingham (SFU).

    Inputs (in order):
        rw  : radius of borehole (m)
        r   : radius of influence (m)
        Tu  : transmissivity upper aquifer (m^2/yr)
        Hu  : potentiometric head upper aquifer (m)
        Tl  : transmissivity lower aquifer (m^2/yr)
        Hl  : potentiometric head lower aquifer (m)
        L   : length of borehole (m)
        Kw  : hydraulic conductivity of borehole (m/yr)

    Vector form:
        x = [rw, r, Tu, Hu, Tl, Hl, L, Kw]

    Output:
        y = water flow rate (m^3/yr)

    Reference:
        https://www.sfu.ca/~ssurjano/borehole.html
    """

    _check_grad_at_opt: bool = False

    def __init__(
        self,
        noise_std: Optional[float] = None,
        negate: bool = False,
        bounds: Optional[List[Tuple[float, float]]] = None,
    ) -> None:
        """
        Initialize the Borehole synthetic test function.

        Args:
            noise_std (float or None): Standard deviation of observation noise.
                If None, the function is noise free.
            negate (bool): If True, returns the negative of the standard
                Borehole output, so that the function is maximized at the
                original minimum.
            bounds (list[tuple[float, float]] or None): Optional custom bounds
                as a list of (lower, upper) tuples, one per input dimension,
                in the order documented in the class docstring. If None, uses
                the standard SFU Borehole bounds.
        """

        # Borehole has fixed dimension 8
        self.dim = 8

        # Default bounds from SFU (Surjanovic & Bingham) matching the input order
        if bounds is None:
            bounds = [
                (0.05, 0.15),  # rw
                (100.0, 50000.0),  # r
                (63070.0, 115600.0),  # Tu
                (990.0, 1110.0),  # Hu
                (63.1, 116.0),  # Tl
                (700.0, 820.0),  # Hl
                (1120.0, 1680.0),  # L
                (9855.0, 12045.0),  # Kw
            ]

        self.continuous_inds = list(range(self.dim))
        self.discrete_inds = []
        self.categorical_inds = []

        super().__init__(noise_std=noise_std, negate=negate, bounds=bounds)

    def _evaluate_true(self, X: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """
        Evaluate the Borehole test function at given inputs.

        Args:
            X (torch.Tensor or np.ndarray): Input locations, either:
                - 1D tensor/array of shape [8] for a single point, or
                - 2D tensor/array of shape [n, 8] for a batch of n points.

        Returns:
            torch.Tensor: 1D tensor of shape [n] with Borehole function values
            (or shape [1] for a single 1D input). If `self.negate` is True,
            returns the negative of the original Borehole function values.

        Raises:
            TypeError: If `X` is not a `torch.Tensor` or `np.ndarray`.
            ValueError: If the last dimension of `X` is not 8.
        """

        # Convert numpy to torch if needed
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X.astype(np.float32))

        if not isinstance(X, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor or numpy.ndarray.")

        # Ensure 2D: [batch, dim]
        if X.ndim == 1:
            X = X.unsqueeze(0)

        if X.shape[-1] != 8:
            raise ValueError(f"Borehole expects input dimension 8, got {X.shape[-1]}")

        # xx = [rw, r, Tu, Hu, Tl, Hl, L, Kw]
        rw = X[..., 0]
        r = X[..., 1]
        Tu = X[..., 2]
        Hu = X[..., 3]
        Tl = X[..., 4]
        Hl = X[..., 5]
        L = X[..., 6]
        Kw = X[..., 7]

        # SFU implementation
        log_term = torch.log(r / rw)

        frac1 = 2.0 * np.pi * Tu * (Hu - Hl)
        frac2a = 2.0 * L * Tu / (log_term * rw.pow(2) * Kw)
        frac2b = Tu / Tl
        frac2 = log_term * (1.0 + frac2a + frac2b)

        y = frac1 / frac2

        if self.negate:
            y = -y

        return y


def parabola(
    x: npt.NDArray,
    beta1: float,
    beta2: float,
    beta12: float,
) -> npt.NDArray:
    """
    Computes a quadratic function with an interaction term for a set of 2D input points.

    The function is defined as:
        f(x1, x2) = beta1 * x1^2 + beta2 * x2^2 + beta12 * sin(6 * x1 * x2 - 3)

    Parameters
    ----------
    x : np.ndarray
        Array of shape (n_samples, 2), where each row is a 2D input point [x1, x2].
    beta1 : float
        Coefficient for the x1^2 term.
    beta2 : float
        Coefficient for the x2^2 term.
    beta12 : float
        Coefficient for the interaction term sin(6 * x1 * x2 - 3).

    Returns
    -------
    np.ndarray
        Array of shape (n_samples,) containing the computed function values for each input.
    """
    return (
        beta1 * x[:, 0] ** 2
        + beta2 * x[:, 1] ** 2
        + beta12 * np.sin((6 * x[:, 0] * x[:, 1]) - 3)
    )


def scale_inputs(
    x: npt.NDArray,
    bounds: dict[str, tuple[float, float]],
) -> np.ndarray:
    """
    Scales normalized input values to their actual ranges based on provided bounds.

    Parameters
    ----------
    x : np.ndarray
        Array of shape (n_samples, n_variables) with normalized values in [0, 1].
        Each column corresponds to an input variable, scaled according to its bounds.
    bounds : dict
        Dictionary mapping variable names to (min, max) tuples.
        The order of variables in x columns should match the order of keys in bounds.

    Raises
    ------
    ValueError
        If any element in x is outside the [0, 1] interval.

    Returns
    -------
    np.ndarray
        Array of shape (n_samples, n_variables) with values scaled to their respective bounds.
    """
    if not ((0 <= x).all() and (x <= 1).all()):
        raise ValueError("All elements in x must be within the [0, 1] interval.")

    min_vals = np.array([bounds[key][0] for key in bounds])
    max_vals = np.array([bounds[key][1] for key in bounds])

    x_scaled = x * (max_vals - min_vals) + min_vals

    return x_scaled


def otlcircuit(
    x: npt.NDArray,
    *args,
) -> npt.NDArray:
    """
    This function computes the midpoint voltage of output transformerless (OTL)
    push-pull circuit.

    Parameters
    ----------
    x : np.ndarray
        Array of shape (n_samples, n_variables) with normalized values in [0, 1].
        Each column corresponds to an input variable, scaled according to its bounds.

    Returns
    -------
    np.ndarray
        Array of calculated midpoint voltages (in volts) for each input sample.

    References
    ----------
    [1] Formula source: OTL Circuit Function, Simon Fraser University,
        https://www.sfu.ca/~ssurjano/otlcircuit.html (accessed July 2024).
    [2] Ben-Ari, E. N., & Steinberg, D. M. (2007). Modeling data from computer experiments:
        an empirical comparison of kriging with MARS and projection pursuit regression.
        Quality Engineering, 19(4), 327-338.
    """
    # Define variable bounds
    bounds = {
        "Rb1": (50, 150),  # Resistance b1 (K-Ohms)
        "Rb2": (25, 70),  # Resistance b2 (K-Ohms)
        "Rf": (0.5, 3),  # Feedback resistance (K-Ohms)
        "Rc1": (1.2, 2.5),  # Resistance c1 (K-Ohms)
        "Rc2": (0.25, 1.2),  # Resistance c2 (K-Ohms)
        "beta": (50, 300),  # Current gain (Amperes)
    }

    # Scale inputs from unit-cube to actual ranges
    x_scaled = scale_inputs(x, bounds)

    # Unpack variables
    Rb1, Rb2, Rf, Rc1, Rc2, beta = x_scaled.T

    # Compute midpoint voltage
    Vb1 = 12 * Rb2 / (Rb1 + Rb2)
    denom = beta * (Rc2 + 9) + Rf

    term1 = ((Vb1 + 0.74) * beta * (Rc2 + 9)) / denom
    term2 = 11.35 * Rf / denom
    term3 = 0.74 * Rf * beta * (Rc2 + 9) / (denom * Rc1)

    midpoint_voltage = term1 + term2 + term3

    return midpoint_voltage


def piston(
    x: npt.NDArray,
    *args,
) -> npt.NDArray:
    """
    This function computes the time it takes a piston to complete one cycle.

    Parameters
    ----------
    x : np.ndarray
        Array of shape (n_samples, n_variables) with normalized values in [0, 1].
        Each column corresponds to an input variable, scaled according to its bounds.

    Returns
    -------
    np.ndarray
        Array of calculated cycle times (in seconds) for each input sample.

    References
    ----------
    [1] Formula source: Piston Simulation Function, Simon Fraser University,
        https://www.sfu.ca/~ssurjano/piston.html (accessed July 2024).
    [2] Ben-Ari, E. N., & Steinberg, D. M. (2007). Modeling data from computer experiments:
        an empirical comparison of kriging with MARS and projection pursuit regression.
        Quality Engineering, 19(4), 327-338.
    """
    # Define variable bounds
    bounds = {
        "M": (30, 60),  # Piston weight (kg)
        "S": (0.005, 0.02),  # Piston surface area (m^2)
        "V0": (0.002, 0.01),  # Initial gas volume (m^3)
        "k": (1000, 5000),  # Spring coefficient (N/m)
        "P0": (90000, 110000),  # Atmospheric pressure (N/m^2)
        "Ta": (290, 296),  # Ambient temperature (K)
        "T0": (340, 360),  # Filling gas temperature (K)
    }

    # Scale inputs from unit-cube to actual ranges
    x_scaled = scale_inputs(x, bounds)

    # Unpack variables
    M, S, V0, k, P0, Ta, T0 = x_scaled.T

    # Compute cycle time
    A = P0 * S + 19.62 * M - (k * V0 / S)
    V = (S / (2 * k)) * (np.sqrt(A**2 + 4 * k * (P0 * V0 / T0) * Ta) - A)

    denom_cycle_time = k + (S**2) * (P0 * V0 / T0) * (Ta / (V**2))
    cycle_time = 2 * np.pi * np.sqrt(M / denom_cycle_time)

    return cycle_time


def wingweight(
    x: npt.NDArray,
    *args,
) -> npt.NDArray:
    """
    This function computes the weight of a light aircraft wing.

    Parameters
    ----------
    x : np.ndarray
        Array of shape (n_samples, n_variables) with normalized values in [0, 1].
        Each column corresponds to an input variable, scaled according to its bounds.

    Returns
    -------
    np.ndarray
        Array of wing weights (in pounds) for each input sample.

    References
    ----------
    [1] Formula source: Wing Weight Function, Simon Fraser University,
        https://www.sfu.ca/~ssurjano/wingweight.html (accessed July 2024).
    """
    # Define variable bounds
    bounds = {
        "Sw": (150, 200),  # Wing area (ft^2)
        "Wfw": (220, 300),  # Weight of fuel in the wing (lb)
        "A": (6, 10),  # Aspect ratio
        "Lam": (
            -10 * np.pi / 180,
            10 * np.pi / 180,
        ),  # Quarter-chord Sweep (radians)
        "q": (16, 45),  # Dynamic pressure at cruise (lb / ft^2)
        "lam": (0.5, 1.0),  # Taper ratio
        "tc": (0.08, 0.18),  # Aerofoil thickness-to-chord ratio
        "Nz": (2.5, 6.0),  # Ultimate load factor
        "Wdg": (1700, 2500),  # Flight design gross weight (lb)
        "Wp": (0.025, 0.08),  # Paint weight (lb / ft^2)
    }

    # Scale inputs from unit-cube to actual ranges
    x_scaled = scale_inputs(x, bounds)

    # Unpack variables
    Sw, Wfw, A, LamCaps, q, lam, tc, Nz, Wdg, Wp = x_scaled.T

    # Calculate wing weight
    factors = [
        0.036 * Sw**0.758 * Wfw**0.0035,
        (A / (np.cos(LamCaps) ** 2)) ** 0.6,
        q**0.006 * lam**0.04,
        (100 * tc / np.cos(LamCaps)) ** (-0.3),
        (Nz * Wdg) ** 0.49,
    ]

    wing_weight = np.prod(factors, axis=0) + (Sw * Wp)

    return wing_weight


def borehole(
    x: npt.NDArray,
    *args,
) -> npt.NDArray:
    """
    This function computes the water flow rate through a borehole.

    Parameters
    ----------
    x : np.ndarray
        Array of shape (n_samples, n_variables) with normalized values in [0, 1].
        Each column corresponds to an input variable, scaled according to its bounds.

    Returns
    -------
    np.ndarray
        Array of borehole water flow rates (in m^3/year) for each input sample.

    References
    ----------
    [1] Formula source: Borehole Function, Simon Fraser University,
        https://www.sfu.ca/~ssurjano/borehole.html (accessed Dec 2025).
    """
    # Define variable bounds
    bounds = {
        "rw": (0.05, 0.15),  # radius of borehole (m)
        "r": (100, 50000),  # radius of influence (m)
        "Tu": (63070, 115600),  # transmissivity upper aquifer (m^2/year)
        "Hu": (990, 1110),  # potentiometric head upper aquifer (m)
        "Tl": (63.1, 116),  # transmissivity lower aquifer (m^2/year)
        "Hl": (700, 820),  # potentiometric head lower aquifer (m)
        "L": (1120, 1680),  # length of borehole (m)
        "Kw": (9855, 12045),  # hydraulic conductivity of borehole (m/year)
    }

    # Scale inputs from unit-cube to actual ranges
    x_scaled = scale_inputs(x, bounds)

    # Unpack variables
    rw, r, Tu, Hu, Tl, Hl, L, Kw = x_scaled.T

    # Compute borehole flow rate
    frac1 = 2 * np.pi * Tu * (Hu - Hl)
    frac2a = 2 * L * Tu / (np.log(r / rw) * rw**2 * Kw)
    frac2b = Tu / Tl
    frac2 = np.log(r / rw) * (1 + frac2a + frac2b)

    flow_rate = frac1 / frac2

    return flow_rate
