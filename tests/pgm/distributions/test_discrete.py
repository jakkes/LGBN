import numpy as np
import pgm.distributions as distributions


def test_discrete():
    d = distributions.Discrete([(1, 2), (2, 1)], [1.0, 0.0], ["a", "b"])
    assert np.array_equal(d.sample(), np.array([1, 2]))
    assert d.sample().shape == (2, )
    assert d.sample(5).shape == (5, 2)
