import numpy as np
import pgm.distributions as distributions


def test_discrete():
    d = distributions.Discrete([1, 2, 3], [1.0, 0.0, 0.0])
    assert np.array_equal(d.sample(), np.array(1))
    assert d.sample().shape == ()
    assert d.sample(5).shape == (5, )
