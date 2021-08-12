import abc
from typing import Sequence, List, Optional, TypeVar

import numpy as np


T = TypeVar("T")


class Base(abc.ABC):
    """Base class for distributions."""

    def __init__(self, dim: int) -> None:
        """
        Args:
            dim (int): Dimension of the distribution.
        """
        super().__init__()

        if dim < 0:
            raise ValueError("Dimension must be a positive integer.")
        self._dim = dim

    @property
    def dim(self) -> int:
        """Dimension in which the distribution generates samples."""
        return self._dim

    @abc.abstractmethod
    def sample(self, batches: Optional[int] = None) -> np.ndarray:
        """Samples from the distribution.

        Args:
            batches (Optional[int], optional): If not None, multiple samples are
                generated and stacked. Defaults to None.

        Returns:
            np.ndarray: Sampled values. If `batches` is not None, then the output
                contains multiple samples stacked in the first dimension.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def marginalize(self, axis: int) -> "Base":
        """Marginalizes the distribution across the given axis.

        Args:
            axis (int): Axis to marginalize over.

        Returns:.
            Base: Marginalized distribution.
        """
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
