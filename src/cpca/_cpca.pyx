# cython: language_level=3
# src/cpca/_cpca.pyx
import numpy as np
cimport numpy as np

from libc.stdlib cimport malloc, free

cdef extern from "_c/cpca.h":
    cdef struct PCAResult:
        int m
        int n
        int k
        double* PCAdirections
        double* scores
        double* loadings
        double* stddev
        double* pve
        double* cum

    PCAResult cpca(double* A, int m, int n)
    void free_pca(PCAResult* R)

# ---------- Python-visible API ---------- #

def pca(np.ndarray[np.double_t, ndim=2, mode="c"] A):
    """
    Thin wrapper around the C cpca() routine.
    Accepts a 2-D numpy array (double, C-contiguous).
    Returns (scores, loadings, stddev, pve, cum).
    """
    if A.dtype != np.double:
        A = np.ascontiguousarray(A, dtype=np.double)
    if not A.flags["C_CONTIGUOUS"]:
        A = np.ascontiguousarray(A)

    cdef int m = A.shape[0]
    cdef int n = A.shape[1]

    # call the C code
    cdef PCAResult res = cpca(<double*>A.data, m, n)

    try:
        # numpy views of the raw buffers
        scores = np.frombuffer(<double[:res.m*res.k]> res.scores, dtype=np.double)\
                    .reshape(res.m, res.k).copy()
        loadings = np.frombuffer(<double[:res.n*res.k]> res.loadings, dtype=np.double)\
                    .reshape(res.n, res.k).copy()
        stddev = np.frombuffer(<double[:res.k]> res.stddev, dtype=np.double).copy()
        pve = np.frombuffer(<double[:res.k]> res.pve, dtype=np.double).copy()
        cum = np.frombuffer(<double[:res.k]> res.cum, dtype=np.double).copy()
    finally:
        # ALWAYS free C memory even if something blows up
        free_pca(&res)

    return scores, loadings, stddev, pve, cum