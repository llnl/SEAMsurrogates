import numpy as np
import torch
from botorch.test_functions.synthetic import SyntheticTestFunction
from typing import Optional, List, Tuple, Union


class Parabola(SyntheticTestFunction):
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
