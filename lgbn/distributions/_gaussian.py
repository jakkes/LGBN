from typing import Sequence, Optional

import numpy as np

import lgbn.distributions as distributions


class Gaussian(distributions.Base):
    """Gaussian distribution, defined by a mean value vector and a covariance matrix."""

    def __init__(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        variable_names: Optional[Sequence[str]] = None,
    ):
        """
        Args:
            mean (np.ndarray): Mean vector.
            covariance (np.ndarray): Covariance matrix.
            variable_names (Optional[Sequence[str]], optional): Optional variable names.
                Defaults to None. If None, names are automatically generated for each
                variable as "0", "1", "2", ...
        """

        mean = mean.flatten()   # .flatten() == .ravel().copy()
        n = mean.shape[0]
        covariance = covariance.reshape((n, n)).copy()
        if variable_names is None:
            variable_names = [str(x) for x in range(n)]

        super().__init__(variable_names)
        self._mean = mean
        self._cov = covariance

    @property
    def mean(self) -> np.ndarray:
        """Mean vector."""
        return self._mean.copy()

    @property
    def covariance(self) -> np.ndarray:
        """Covariance matrix."""
        return self._cov.copy()

    def sample(self, batches: Optional[int] = None) -> np.ndarray:
        return np.random.multivariate_normal(self._mean, self._cov, batches)
