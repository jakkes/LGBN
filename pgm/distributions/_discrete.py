from typing import Sequence, Optional, Tuple

import numpy as np

import pgm.distributions as distributions


class Discrete(distributions.Base):
    """Discrete multivariate distribution."""

    def __init__(
        self,
        values: Sequence[Tuple[float, ...]],
        probabilities: Sequence[float],
        variable_names: Sequence[str],
    ):
        """
        Args:
            values (Sequence[Tuple[float, ...]]): Values obtainable in the distribution.
            probabilities (Sequence[float]): Probabilities of each value.
            variable_names (Sequence[str]): Names of the variables.
        """
        super().__init__(variable_names)
        self._values = np.asarray(values, dtype=np.float32).reshape(
            len(probabilities), -1
        )
        self._probabilities = np.asarray(probabilities, dtype=np.float32).flatten()

        if self._values.shape[0] != self._probabilities.shape[0]:
            raise ValueError("Need to provide equally many probabilities and values.")
        if self._probabilities.sum() != 1.0:
            raise ValueError("Probabilities need to sum to one.")
        if np.any(self._probabilities < 0):
            raise ValueError("Probabilities cannot be negative.")

    @property
    def values(self) -> np.ndarray:
        """Values obtainable by the distribution."""
        return self._values.copy()

    @property
    def probabilities(self) -> np.ndarray:
        """Probabilities of taking each value."""
        return self._probabilities.copy()

    def sample(self, batches: Optional[int] = None) -> np.ndarray:
        i = np.random.choice(
            self.dim, size=batches, replace=True, p=self._probabilities
        )
        return self.values[i]
