import numpy as np

import pgm.distributions as distributions


def test_by_evidence():
    c = distributions.conditional.Discrete(
        [[1, 2, 3]],
        [[4, 5, 6]],
        np.array([
            np.array([1, 0, 0]),
            np.array([0, 1, 0]),
            np.array([0, 0, 1])
        ], dtype=object)
    )
    d = c.by_evidence(np.array([5]))
    assert np.array_equal(d.sample(), np.array([2]))


def test_marginalization():
    c = distributions.conditional.Discrete(
        [[1, 2, 3]],
        [[4, 5, 6]],
        np.array([
            np.array([1, 0, 0]),
            np.array([0, 1, 0]),
            np.array([0, 0, 1])
        ])
    )
    d = distributions.Discrete(
        [[4, 5, 6]],
        np.array([0, 0, 1])
    )
    e = c.marginalize(d)
    assert np.array_equal(e.sample(), np.array([3]))
