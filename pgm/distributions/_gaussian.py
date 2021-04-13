from typing import Sequence, Optional

import numpy as np

import pgm.distributions as distributions


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
        covariance = covariance.reshape((n, n))
        covariance = 0.5 * (covariance + covariance.transpose())
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

    def marginalize(self, variable_name: str) -> "Gaussian":
        i = self.variable_names.index(variable_name)

        new_cov = np.delete(self._cov, i, axis=1)
        new_cov = np.delete(new_cov, i, axis=0)

        return Gaussian(
            np.concatenate((self._mean[:i], self._mean[i+1:]), axis=0),
            new_cov,
            [*self.variable_names[:i], *self.variable_names[i+1:]]
        )

    def _reorder(self, variable_names: Sequence[str]) -> "Gaussian":
        mean = self._mean
        cov = self._cov
        current_order = list(self.variable_names)

        for i, name in enumerate(variable_names):
            j = current_order.index(name)
            mean[i], mean[j] = mean[j].copy(), mean[i].copy()
            cov = np.swapaxes(cov, j, i)
            current_order[i], current_order[j] = current_order[j], current_order[i]

        return Gaussian(mean, cov, current_order)
