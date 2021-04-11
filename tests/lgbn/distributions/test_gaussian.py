import numpy as np

from lgbn import distributions


def test_gaussian():
    d = distributions.Gaussian(
        np.array([5.0, 3.0]), np.array([[1.0, 0.8], [0.8, 1.0]]), ["a", "b"]
    )
    assert d.sample().shape == (2, )
    assert d.sample(5).shape == (5, 2)
