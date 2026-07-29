import numpy as np
from scipy.stats import qmc


def _scale_to_bounds(x_unit: np.ndarray, bounds_low, bounds_high) -> np.ndarray:
    """
    Scale unit-hypercube points to the specified lower and upper bounds.

    Args:
        x_unit: Points defined in the unit hypercube.
        bounds_low: Lower bound for each dimension.
        bounds_high: Upper bound for each dimension.

    Returns:
        Points transformed from the unit hypercube to the specified bounds.
    """
    bounds_low = np.asarray(bounds_low, dtype=float)
    bounds_high = np.asarray(bounds_high, dtype=float)
    return bounds_low + x_unit * (bounds_high - bounds_low)


def _lhd_permutation_matrix(
    n_samples: int, dim: int, rng: np.random.Generator
) -> np.ndarray:
    """
    Generate a Latin hypercube permutation matrix.

    Args:
        n_samples: Number of samples, or rows, to generate.
        dim: Number of dimensions, or columns, in the matrix.
        rng: NumPy random number generator used to create permutations.

    Returns:
        An integer array of shape ``(n_samples, dim)`` where each column
        contains a permutation of the integers from 0 to ``n_samples - 1``.
    """
    return np.column_stack([rng.permutation(n_samples) for _ in range(dim)])


def _perm_to_unit_lhd(
    perm: np.ndarray,
    jitter: bool = False,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Convert a Latin hypercube permutation matrix to unit-hypercube samples.

    Args:
        perm: Permutation matrix of shape ``(n_samples, dim)``.
        jitter: Whether to randomly jitter points within each Latin
            hypercube interval. If ``False``, interval midpoints are used.
        rng: Optional random number generator used when jittering. If omitted,
            a default generator is created.

    Returns:
        An array of shape ``(n_samples, dim)`` containing samples in the
        unit hypercube.
    """
    n_samples, dim = perm.shape
    if rng is None:
        rng = np.random.default_rng()

    if jitter:
        u = rng.uniform(size=(n_samples, dim))
    else:
        u = np.full((n_samples, dim), 0.5)

    return (perm + u) / n_samples


def _pairwise_distances(x: np.ndarray) -> np.ndarray:
    """
    Compute Euclidean distances between all unique pairs of points.

    Args:
        x: Array of points with shape ``(n_samples, n_dimensions)``.

    Returns:
        One-dimensional array containing the pairwise distances.
    """
    diff = x[:, None, :] - x[None, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    iu = np.triu_indices(x.shape[0], k=1)
    return np.sqrt(dist2[iu])


def phi_p_criterion(x: np.ndarray, p: float = 50) -> float:
    """
    Compute the DiceDesign-style phi-p space-filling criterion.

    Lower values indicate better space-filling designs. As ``p`` approaches
    infinity, minimizing this criterion approaches maximin optimization.

    Args:
        x: Design points with shape ``(n_samples, n_dimensions)``.
        p: Exponent controlling the emphasis on small pairwise distances.

    Returns:
        The scalar phi-p criterion value.
    """
    dists = _pairwise_distances(x)
    dists = np.clip(dists, 1e-15, None)
    return float(np.sum(dists ** (-p)) ** (1.0 / p))


def mindist_criterion(x: np.ndarray) -> float:
    """
    Compute the minimum pairwise Euclidean distance between design points.

    Larger values indicate better space-filling designs.

    Args:
        x: Design points with shape ``(n_samples, n_dimensions)``.

    Returns:
        The smallest Euclidean distance between any two distinct points.
    """
    dists = _pairwise_distances(x)
    return float(np.min(dists))


def _temperature(profile: str, T0: float, c: float, iteration: int, it: int) -> float:
    """
    Compute the simulated annealing temperature for a given iteration.

    Supports geometric cooling, which scales the initial temperature by
    ``c**iteration``, and linear cooling, which decreases to zero over the
    total number of iterations.

    Args:
        profile: Cooling schedule, either ``"GEOM"`` or ``"LINEAR"``.
        T0: Initial temperature.
        c: Geometric cooling factor.
        iteration: Current iteration number.
        it: Total number of iterations.

    Returns:
        The temperature for the specified iteration.

    Raises:
        ValueError: If ``profile`` is not ``"GEOM"`` or ``"LINEAR"``.
    """
    profile = profile.upper()

    if profile == "GEOM":
        return T0 * (c**iteration)

    if profile == "LINEAR":
        frac = max(0.0, 1.0 - iteration / max(it, 1))
        return T0 * frac

    raise ValueError("profile must be 'GEOM' or 'LINEAR'")


def maximin_sa_lhd(
    bounds_low,
    bounds_high,
    n_samples: int,
    T0: float = 10.0,
    c: float = 0.95,
    it: int = 2000,
    p: float = 50,
    profile: str = "GEOM",
    Imax: int = 100,
    jitter: bool = False,
    seed: int | None = None,
    return_history: bool = False,
):
    """
    Optimize a Latin hypercube design using simulated annealing.

    The optimization operates on the permutation structure of a Latin
    hypercube and minimizes the phi-p space-filling criterion. Candidate
    designs are generated by swapping two entries within a randomly selected
    dimension and are accepted according to a simulated annealing schedule.

    Args:
        bounds_low: Lower bounds for each design dimension.
        bounds_high: Upper bounds for each design dimension.
        n_samples: Number of design points to generate.
        T0: Initial annealing temperature.
        c: Geometric cooling factor.
        it: Number of simulated annealing iterations.
        p: Exponent used by the phi-p criterion.
        profile: Cooling schedule, such as ``"GEOM"`` or ``"LINEAR"``.
        Imax: Number of non-improving iterations used by the Morris-style
            stagnation adjustment.
        jitter: Whether to randomly jitter points within Latin hypercube
            intervals.
        seed: Optional random seed.
        return_history: If ``True``, return the optimized design and
            optimization diagnostics. Otherwise, return only the design.

    Returns:
        Either the optimized design scaled to the specified bounds, or a
        dictionary containing the design, initial design, annealing
        parameters, criterion history, temperature history, acceptance
        probabilities, and final quality metrics.
    """
    rng = np.random.default_rng(seed)
    bounds_low = np.asarray(bounds_low, dtype=float)
    bounds_high = np.asarray(bounds_high, dtype=float)
    dim = len(bounds_low)

    perm = _lhd_permutation_matrix(n_samples, dim, rng)
    x_unit = _perm_to_unit_lhd(perm, jitter=jitter, rng=rng)
    current_phi = phi_p_criterion(x_unit, p=p)

    best_perm = perm.copy()
    best_phi = current_phi

    crit_values = [current_phi]
    temp_values = []
    proba_values = []

    no_improve = 0

    for k in range(it):
        temp = max(_temperature(profile, T0, c, k, it), 1e-12)
        temp_values.append(temp)

        cand_perm = perm.copy()

        col = rng.integers(dim)
        i, j = rng.choice(n_samples, size=2, replace=False)
        cand_perm[i, col], cand_perm[j, col] = cand_perm[j, col], cand_perm[i, col]

        cand_unit = _perm_to_unit_lhd(cand_perm, jitter=jitter, rng=rng)
        cand_phi = phi_p_criterion(cand_unit, p=p)

        delta = cand_phi - current_phi
        accepted = False
        accept_prob = 0.0

        if delta <= 0:
            accepted = True
            accept_prob = 1.0
        else:
            accept_prob = float(np.exp(-delta / temp))
            if rng.uniform() < accept_prob:
                accepted = True

        proba_values.append(accept_prob)

        if accepted:
            perm = cand_perm
            current_phi = cand_phi

            if cand_phi < best_phi:
                best_phi = cand_phi
                best_perm = cand_perm.copy()
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1

        crit_values.append(current_phi)

        # crude Morris-style stagnation nudge
        if profile.upper() == "GEOM_MORRIS" and no_improve >= Imax:
            no_improve = 0
            col = rng.integers(dim)
            idx = rng.choice(n_samples, size=min(4, n_samples), replace=False)
            shuffled = cand_perm[idx, col].copy()
            rng.shuffle(shuffled)
            perm[idx, col] = shuffled
            x_unit = _perm_to_unit_lhd(perm, jitter=jitter, rng=rng)
            current_phi = phi_p_criterion(x_unit, p=p)

            if current_phi < best_phi:
                best_phi = current_phi
                best_perm = perm.copy()

    best_unit = _perm_to_unit_lhd(best_perm, jitter=jitter, rng=rng)
    best_design = _scale_to_bounds(best_unit, bounds_low, bounds_high)

    result = {
        "design": best_design,
        "InitialDesign": _scale_to_bounds(
            _perm_to_unit_lhd(
                _lhd_permutation_matrix(n_samples, dim, np.random.default_rng(seed)),
                jitter=jitter,
                rng=np.random.default_rng(seed),
            ),
            bounds_low,
            bounds_high,
        ),
        "T0": T0,
        "c": c,
        "it": it,
        "p": p,
        "profile": profile,
        "Imax": Imax,
        "critValues": np.asarray(crit_values, dtype=float),
        "tempValues": np.asarray(temp_values, dtype=float),
        "probaValues": np.asarray(proba_values, dtype=float),
        "phiP_best": float(best_phi),
        "mindist_best": mindist_criterion(best_unit),
    }

    if return_history:
        return result
    return best_design


def random_design(bounds_low, bounds_high, n_samples, seed=None):
    """
    Generate uniformly distributed random design points within bounds.

    Args:
        bounds_low: Lower bound for each dimension.
        bounds_high: Upper bound for each dimension.
        n_samples: Number of design points to generate.
        seed: Optional seed for reproducible sampling.

    Returns:
        Array of shape ``(n_samples, n_dimensions)`` containing the sampled
        design points.
    """
    rng = np.random.default_rng(seed)
    bounds_low = np.asarray(bounds_low, dtype=float)
    bounds_high = np.asarray(bounds_high, dtype=float)
    return rng.uniform(bounds_low, bounds_high, size=(n_samples, len(bounds_low)))


def latin_hypercube_design(bounds_low, bounds_high, n_samples, seed=None):
    """
    Generate a Latin hypercube design scaled to specified bounds.

    Args:
        bounds_low: Lower bound for each dimension.
        bounds_high: Upper bound for each dimension.
        n_samples: Number of design points to generate.
        seed: Optional seed for reproducible sampling.

    Returns:
        Array of shape ``(n_samples, n_dimensions)`` containing the scaled
        Latin hypercube design.
    """
    bounds_low = np.asarray(bounds_low, dtype=float)
    bounds_high = np.asarray(bounds_high, dtype=float)
    dim = len(bounds_low)

    sampler = qmc.LatinHypercube(d=dim, seed=seed)
    x_unit = sampler.random(n=n_samples)
    return qmc.scale(x_unit, bounds_low, bounds_high)


def generate_initial_design(
    bounds_low,
    bounds_high,
    n_samples,
    method="random",
    seed=None,
    **kwargs,
):
    """
    Generate an initial design using the selected sampling method.

    Supported methods include uniform random sampling, Latin hypercube
    sampling, and simulated-annealing-optimized maximin Latin hypercube
    sampling.

    Args:
        bounds_low: Lower bound for each dimension.
        bounds_high: Upper bound for each dimension.
        n_samples: Number of design points to generate.
        method: Design method, one of ``"random"``, ``"lhd"``,
            ``"latin_hypercube"``, or ``"maximin_lhd"``.
        seed: Optional seed for reproducible sampling.
        **kwargs: Additional options passed to the selected method, such as
            ``jitter`` for Latin hypercube designs or simulated annealing
            parameters for ``"maximin_lhd"``.

    Returns:
        The generated design array. For ``"maximin_lhd"``, the return value
        may be either the optimized design or a diagnostics dictionary,
        depending on the supplied keyword arguments.

    Raises:
        ValueError: If ``method`` is not supported.
    """
    method = method.lower()

    if method == "random":
        rng = np.random.default_rng(seed)
        bounds_low = np.asarray(bounds_low, dtype=float)
        bounds_high = np.asarray(bounds_high, dtype=float)
        return rng.uniform(bounds_low, bounds_high, size=(n_samples, len(bounds_low)))

    if method in {"lhd", "latin_hypercube"}:
        rng = np.random.default_rng(seed)
        dim = len(bounds_low)
        perm = np.column_stack([rng.permutation(n_samples) for _ in range(dim)])
        x_unit = _perm_to_unit_lhd(perm, jitter=kwargs.get("jitter", False), rng=rng)
        return _scale_to_bounds(x_unit, bounds_low, bounds_high)

    if method in {"maximin_lhd"}:
        return maximin_sa_lhd(
            bounds_low=bounds_low,
            bounds_high=bounds_high,
            n_samples=n_samples,
            seed=seed,
            **kwargs,
        )

    raise ValueError(f"Unknown method: {method}")
