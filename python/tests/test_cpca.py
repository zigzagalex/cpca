# python/tests/test_cpca.py
import numpy as np
import cpca


def test_shape():
    X = np.random.randn(20, 5)
    T, L, s, pve, cum = cpca.pca(X)
    assert T.shape == (20, 5)
    assert L.shape == (5, 5)
    np.testing.assert_almost_equal(cum[-1], 1.0, decimal=6)


def test_known_decomposition():
    # Matrix with perfect 1D structure: every row is a scaled version of [1, 2]
    X = np.array([[1, 2], [2, 4], [3, 6], [4, 8], [5, 10]], dtype=np.float64)

    T, L, s, pve, cum = cpca.pca(X)

    # All variance explained by 1 component
    assert T.shape == (5, 2)
    assert L.shape == (2, 2)
    assert s.shape == (2,)
    assert np.allclose(pve[0], 1.0, rtol=1e-5)
    assert np.allclose(pve[1], 0.0, atol=1e-10)
    assert np.allclose(cum[-1], 1.0, rtol=1e-6)


def test_large_matrix():
    np.random.seed(42)
    X = np.random.randn(1000, 100)
    T, L, s, pve, cum = cpca.pca(X)

    assert T.shape == (1000, 100)
    assert L.shape == (100, 100)
    assert s.shape == (100,)
    assert pve.shape == (100,)
    assert cum.shape == (100,)
    assert np.all(cum <= 1.0 + 1e-6)
    assert cum[-1] > 0.999  # should explain ~100% variance
