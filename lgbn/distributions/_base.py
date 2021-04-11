import abc
from typing import Sequence, List, Optional

import numpy as np


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
