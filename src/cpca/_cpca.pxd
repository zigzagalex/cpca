# src/cpca/_cpca.pxd
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