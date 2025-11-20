import numpy as np
import numpy.typing as npt


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
