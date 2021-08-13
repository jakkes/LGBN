import abc
from typing import Union

import numpy as np

import pgm.distributions as distributions


class Continuous(distributions.Base):
    """Base class for continuous distributions."""

    @abc.abstractmethod
    def likelihood(self, evidence: np.ndarray) -> Union[np.ndarray, float]:
        """Computes likelihood of the observed data.

        Args:
            evidence (np.ndarray): Array of observed values of shape
                `(d1, d2, ..., dn, N)`, with `N` denoting the distribution dimension.
                All operations are performed on the last axis only, allowing for batch
                processing.

        Returns:
            Union[np.ndarray, float]: Likelihood of observed values. Either an array of
                shape `(d1, d2, ..., dn)` or, if the input is of shape `(N, )`, a
                scalar.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def cdf_probability(self, target: np.ndarray) -> Union[np.ndarray, float]:
        """Computes the probability of observing values smaller than `target`, i.e.
        `P(X < t)`, where `t` denotes `target`.

        Args:
            target (np.ndarray): Array of values of shape `(d1, d2, ..., dn, N)`, with
                `N` denoting the distribution dimension. All operations are performed on
                the last axis only, allowing for batch processing.

        Returns:
            Union[np.ndarray, float]: Probability values. Either an array of
                shape `(d1, d2, ..., dn)` or, if the input is of shape `(N, )`, a
                scalar.
        """
        raise NotImplementedError
