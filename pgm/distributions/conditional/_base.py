import abc

import numpy as np

import pgm.distributions as distributions

class Base(abc.ABC):
    """Base conditional distribution."""

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
