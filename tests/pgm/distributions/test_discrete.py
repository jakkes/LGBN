import numpy as np
import pgm.distributions as distributions


def test_discrete():
    d = distributions.Discrete([[1, 2, 3]], [1.0, 0.0, 0.0])
    assert np.array_equal(d.sample(), np.array([1]))
    assert d.sample().shape == (1, )
    assert d.sample(5).shape == (5, 1)


def test_multivariate():
    d = distributions.Discrete(
        [[1, 2, 3], [4, 5, 6]],
        [[0, 0, 0],
        [0, 0, 1],
        [0, 0, 0]]
    )
    assert np.array_equal(d.sample(), np.array([2, 6]))


def test_marginalize():
    d = distributions.Discrete(
        [[1, 2, 3], [4, 5, 6]],
        [[0.2, 0.2, 0.6],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0]]
    )
    m = d.marginalize(1)
    assert np.array_equal(m.probabilities, np.array([1.0, 0.0, 0.0]))


def test_probability():
    d = distributions.Discrete(
        [[1, 2, 3], [4, 5, 6]],
        [[0.2, 0.2, 0.6],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0]]
    )
    assert d.probability(np.array([1, 4])) == 0.2
    assert d.probability(np.array([1, 5])) == 0.2
    assert d.probability(np.array([1, 6])) == 0.6
    assert d.probability(np.array([2, 6])) == 0.0

    assert np.array_equal(
        d.probability(np.array([[1,4],[1,5],[1,6],[2,6]])),
        np.array([0.2, 0.2, 0.6, 0.0])
    )

    assert d.probability(np.array([0.5, 3.5])) == 0.2
    assert d.probability(np.array([0.5, 5.5])) == 0.6
    assert d.probability(np.array([0.5, 6.5])) == 0.6
