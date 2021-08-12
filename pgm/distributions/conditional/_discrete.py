from typing import Sequence

import numpy as np

import pgm.distributions as distributions
import pgm.distributions.conditional as conditional


class Discrete(conditional.Base):
    """Multivariate distribution of discrete variables, conditionally dependent on some
    other multivariate distribution."""

    def __init__(
        self,
        values: Sequence[Sequence[float]],
        conditioned_values: Sequence[Sequence[float]],
        probabilities: np.ndarray,
    ):
        """
        Args:
            values (Sequence[Sequence[float]]): Values obtainable by the output random
                variables. First sequence of floats denotes the values obtainable by
                the first variable, etc.
            conditioned_values (Sequence[Sequence[float]]): Values obtainable by the
                conditioned random variables. First sequence of floats sdenotes the
                values obtainable by the first conditioned variable, etc.
            probabilities (np.ndarray): Probabilities of each combination of output
                values given the values of the conditioned variables. The format of this
                object may seem daunting at first, but is rather straight forward. If
                there are `N` output variables, each taking `l_i` values, and `M`
                conditioned variables, each taking `k_i` values, then this is a tensor
                of shape `(k_1, k_2, ..., k_M)` where each element itself is a tensor
                of shape `(l_1, l_2, ..., l_N)` summing to one.
        """
        super().__init__(len(values), len(conditioned_values))

        self._values = [np.asarray(x) for x in values]
        self._conditioned_values = [np.asarray(x) for x in conditioned_values]
        self._probabilities = np.asarray(probabilities, dtype=object)

        for values in self._values:
            if not np.all(np.diff(values) > 0):
                raise ValueError("Values must be in increasing order.")
        for values in self._conditioned_values:
            if not np.all(np.diff(values) > 0):
                raise ValueError("Conditioned values must be in increasing order.")

    def by_evidence(self, evidence: np.ndarray) -> "distributions.Discrete":
        i = tuple(
            np.where(self._conditioned_values[i] == x)[0][0] for i, x in enumerate(evidence)
        )
        return distributions.Discrete(
            self._values, self._probabilities[i]
        )

    def marginalize(
        self, distribution: "distributions.Discrete"
    ) -> "distributions.Discrete":

        if not np.array_equal(distribution.values, self._conditioned_values):
            raise ValueError(
                "Values of the given distribution do not align with the conditioned "
                "values."
            )
        conditioned_shape = tuple(len(x) for x in self._conditioned_values)
        new_p = np.zeros(tuple(len(x) for x in self._values))
        for p, di in zip(
            distribution.probabilities.ravel(), np.ndindex(conditioned_shape)
        ):
            new_p += np.asarray(p * self._probabilities[di], dtype=np.float32)
        return distributions.Discrete(self._values, new_p)
