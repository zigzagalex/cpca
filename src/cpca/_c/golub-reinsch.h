// In svd.h
#ifndef SVD_H
#define SVD_H

typedef struct {
    int m;      // number of rows of the original matrix A
    int n;      // number of columns of the original matrix A
    int k;      // min of m,n 
    double *U;  // Left singular vectors, stored in row-major (size: m x m)
    double *S;  // Singular values (vector of length k)
    double *V;  // Right singular vectors, stored in row-major (size: n x n)
} SVDResult;

// Compute the SVD of an m x n matrix A (given in row-major order) using your
// Golub-Reinsch algorithm. The tolerance (epsilon) is used for convergence checks.
SVDResult golub_reinsch_svd(int m, int n, const double *A, double epsilon);

free_svd(SVDResult *svd);

#endif

