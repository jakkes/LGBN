from typing import Sequence, Optional

import numpy as np

import pgm.distributions as distributions


class Discrete(distributions.Base):
    """Discrete multivariate distribution."""

    def __init__(
        self,
        values: Sequence[Sequence[float]],
        probabilities: np.ndarray,
    ):
        """
        Args:
            values (Sequence[Sequence[float]]): Values obtainable by each variable in
                the multivariate distribution. The first output can take values in the
                first sequence, second variable values in the second sequence, etc.
            probabilities (np.ndarray): Probabilities of each value combination. If
                there are `D` variables, each taking `n_i` possible values, then this
                array should be of shape `(n_1, n_2, ..., n_D)`.
        """
        super().__init__(len(values))
        self._values = np.array([np.array(x, copy=True) for x in values], dtype=object)
        self._probabilities = np.array(probabilities, copy=True)
        self._cumsummed = self._probabilities.ravel().cumsum(0)

        if self._cumsummed[-1] != 1.0:
            raise ValueError("Probabilities need to sum to one.")
        if np.any(self._probabilities < 0):
            raise ValueError("Probabilities cannot be negative.")
        for values, dim in zip(self._values, self._probabilities.shape):
            if len(values) != dim:
                raise ValueError("Probabilities were not of expected shape.")
        for values in self._values:
            if not np.all(np.diff(values) > 0):
                raise ValueError("Values must be in increasing order.")

    @property
    def values(self) -> np.ndarray:
        """Values obtainable by the distribution."""
        return self._values.copy()

    @property
    def probabilities(self) -> np.ndarray:
        """Probabilities of taking each value."""
        return self._probabilities.copy()

    def sample(self, batches: Optional[int] = None) -> np.ndarray:
        if batches is None:
            return self.sample(1)[0]

        w = np.random.random(size=(batches, 1))
        i = np.unravel_index(
            np.sum(w > self._cumsummed, axis=1), self._probabilities.shape
        )
        return np.stack([value[j] for value, j in zip(self._values, i)], axis=1)

    def marginalize(self, axis: int) -> "Discrete":
        return Discrete(
            np.concatenate((self._values[:axis], self._values[axis+1:]), axis=0),
            self._probabilities.sum(axis=axis)
        )
