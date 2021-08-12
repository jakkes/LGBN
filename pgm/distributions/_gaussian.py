from typing import Sequence, Optional

import numpy as np

import pgm.distributions as distributions


class Gaussian(distributions.Base):
    """Gaussian distribution, defined by a mean value vector and a covariance matrix."""

    def __init__(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
    ):
        """
        Args:
            mean (np.ndarray): Mean vector.
            covariance (np.ndarray): Covariance matrix.
        """

        mean = mean.flatten()   # .flatten() == .ravel().copy()
        n = mean.shape[0]
        covariance = covariance.reshape((n, n))
        covariance = 0.5 * (covariance + covariance.transpose())

        super().__init__(len(mean.shape))
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

    def marginalize(self, axis: int) -> "Gaussian":
        new_cov = np.delete(self._cov, axis, axis=1)
        new_cov = np.delete(new_cov, axis, axis=0)

        return Gaussian(
            np.concatenate((self._mean[:axis], self._mean[axis+1:]), axis=0),
            new_cov,
        )
