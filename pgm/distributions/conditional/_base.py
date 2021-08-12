import abc
from typing import Sequence, List

import numpy as np

import pgm.distributions as distributions

class Base(abc.ABC):
    """Base conditional distribution."""

    def __init__(self, dim: int, conditioned_dim: int):
        """
        Args:
            dim (int): Dimension of the distribution.
            conditioned_dim (int): Dimension of the conditional distribution.
        """
        super().__init__()
        self._dim = dim
        self._conditioned_dim = conditioned_dim

    @property
    def dim(self) -> int:
        """Dimension in which the distribution generates samples."""
        return self._dim

    @property
    def conditioned_dim(self) -> int:
        return self._conditioned_dim

    @abc.abstractmethod
    def marginalize(self, distribution: distributions.Base) -> distributions.Base:
        """Performs marginalization across the conditioned variable.

        Args:
            distribution (distributions.Base): Distribution of the conditioned variable.

        Returns:
            distributions.Base: Resulting, marginalized distribution.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def by_evidence(self, evidence: np.ndarray) -> distributions.Base:
        """Returns the distribution given the value of the conditioned variable.

        Args:
            evidence (np.ndarray): Value of the conditioned variable.

        Returns:
            distributions.Base: Distribution.
        """
        raise NotImplementedError
