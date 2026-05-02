import numpy as np
from scipy.stats import qmc


def _scale_to_bounds(x_unit: np.ndarray, bounds_low, bounds_high) -> np.ndarray:
    bounds_low = np.asarray(bounds_low, dtype=float)
    bounds_high = np.asarray(bounds_high, dtype=float)
    return bounds_low + x_unit * (bounds_high - bounds_low)


def _lhd_permutation_matrix(
    n_samples: int, dim: int, rng: np.random.Generator
) -> np.ndarray:
    return np.column_stack([rng.permutation(n_samples) for _ in range(dim)])


def _perm_to_unit_lhd(
    perm: np.ndarray,
    jitter: bool = False,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    n_samples, dim = perm.shape
    if rng is None:
        rng = np.random.default_rng()

    if jitter:
        u = rng.uniform(size=(n_samples, dim))
    else:
        u = np.full((n_samples, dim), 0.5)

    return (perm + u) / n_samples


def _pairwise_distances(x: np.ndarray) -> np.ndarray:
    diff = x[:, None, :] - x[None, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    iu = np.triu_indices(x.shape[0], k=1)
    return np.sqrt(dist2[iu])


def phi_p_criterion(x: np.ndarray, p: float = 50) -> float:
    """
    DiceDesign-style phi_p criterion.

    Lower is better.
    As p -> infinity, minimizing phi_p approaches maximin optimization.
    """
    dists = _pairwise_distances(x)
    dists = np.clip(dists, 1e-15, None)
    return float(np.sum(dists ** (-p)) ** (1.0 / p))


def mindist_criterion(x: np.ndarray) -> float:
    dists = _pairwise_distances(x)
    return float(np.min(dists))


def _temperature(profile: str, T0: float, c: float, iteration: int, it: int) -> float:
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
    Python approximation of DiceDesign::maximinSA_LHS in R.

    Parameters mirror the R routine where practical.
    The optimization is done over a Latin hypercube permutation structure.
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
    rng = np.random.default_rng(seed)
    bounds_low = np.asarray(bounds_low, dtype=float)
    bounds_high = np.asarray(bounds_high, dtype=float)
    return rng.uniform(bounds_low, bounds_high, size=(n_samples, len(bounds_low)))


def latin_hypercube_design(bounds_low, bounds_high, n_samples, seed=None):
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
