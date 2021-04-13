import abc
from typing import Sequence, List, Optional, TypeVar

import numpy as np


T = TypeVar("T")


class Base(abc.ABC):
    """Base class for multivariate distributions."""

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
    def sample(self, batches: Optional[int] = None) -> np.ndarray:
        """Samples from the distribution.

        Args:
            batches (Optional[int], optional): If not None, multiple samples are generated and
                stacked. Defaults to None.

        Returns:
            np.ndarray: Sampled values. If `batches` is not None, then the output
                contains multiple samples stacked in the first dimension.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def marginalize(self, variable_name: str) -> "Base":
        """Marginalizes the distribution across the given variable.

        Args:
            variable_name (str): Variable to marginalize over.

        Returns:.
            Base: Marginalized distribution
        """
        raise NotImplementedError

    def reorder(self, variable_names: Sequence[str]) -> "Base":
        """Reorders the output variables of the distribution.

        Args:
            variable_names (Sequence[str]): Requested order of variables. Variables of
                the distribution that are not in the given sequence, are marginalized.
        
        Returns:
            Base: New distribution with the specified output order.
        """
        new = self
        for name in self._variable_names:
            if name not in variable_names:
                new = new.marginalize(name)
        return new._reorder(variable_names)

    @abc.abstractmethod
    def _reorder(self, variable_names: Sequence[str]) -> "Base":
        """Reorders the output variables according to the given sequence of
        variable names. This method is called from `reorder`, which has already
        marginalized non-requested variable names."""
        raise NotImplementedError

    def cast(self, cls: T) -> T:
        """Casts the object to a specific distribution.

        Args:
            cls (T): Class to cast to.

        Raises:
            RuntimeError: If the cast failed.

        Returns:
            T: View of the same distribution.
        """
        if not isinstance(self, cls):
            raise RuntimeError(f"Failed casting from {self} to {T}")
        return self
