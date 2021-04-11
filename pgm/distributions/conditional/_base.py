import abc
from typing import Sequence, List

import numpy as np

import pgm.distributions as distributions

class Base(abc.ABC):
    """Base conditional distribution."""

    def __init__(self, variable_names: Sequence[str]):
        """
        Args:
            variable_names (Sequence[str]): Sequence of variable name identifiers.
                Output variables are ordered, and thus identifiable, by these names.
        """
        self._variable_names = list(variable_names)

    @property
    def dim(self) -> int:
        """Dimension in which the distribution generates samples."""
        return len(self._variable_names)

    @property
    def variable_names(self) -> List[str]:
        """Variable names of outputs."""
        return self._variable_names

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
