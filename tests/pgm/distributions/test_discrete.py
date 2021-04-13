import numpy as np
import pgm.distributions as distributions


def test_discrete():
    d = distributions.Discrete([[1, 2, 3]], [1.0, 0.0, 0.0], ["a"])
    assert np.array_equal(d.sample(), np.array([1]))
    assert d.sample().shape == (1, )
    assert d.sample(5).shape == (5, 1)


def test_multivariate():
    d = distributions.Discrete(
        [[1, 2, 3], [4, 5, 6]],
        [[0, 0, 0],
        [0, 0, 1],
        [0, 0, 0]],
        ["a", "b"]
    )
    assert np.array_equal(d.sample(), np.array([2, 6]))


def test_marginalize():
    d = distributions.Discrete(
        [[1, 2, 3], [4, 5, 6]],
        [[0.2, 0.2, 0.6],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0]],
        ["a", "b"]
    )
    m = d.marginalize("b")
    assert np.array_equal(m.probabilities, np.array([1.0, 0.0, 0.0]))


def test_reorder():
    d = distributions.Discrete(
        [[1, 2, 3], [4, 5, 6]],
        [[0.2, 0.2, 0.6],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0]],
        ["a", "b"]
    )
    d = d.reorder(["b", "a"])
    for _ in range(10):
        assert d.sample()[1] == 1
