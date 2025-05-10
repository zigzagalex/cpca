# python/tests/test_cpca.py
import numpy as np
import cpca

def test_shape():
    X = np.random.randn(20, 5)
    T, L, s, pve, cum = cpca.pca(X)
    assert T.shape == (20, 5)
    assert L.shape == (5, 5)
    np.testing.assert_almost_equal(cum[-1], 1.0, decimal=6)