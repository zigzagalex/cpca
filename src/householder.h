#ifndef BIDIOG_H
#define BIDIOG_H

// Perform Householder bidiagonalization on matrix A (m x n).
// On exit, A is overwritten with the bidiagonal matrix B, and U, V are filled with the
// products of Householder reflectors such that A_original = U * B * Vᵀ.
//
// Parameters:
//   m    - number of rows in A
//   n    - number of columns in A
//   A    - pointer to the matrix A (stored row-major)
//   U    - pointer to an m x m matrix (should be pre-allocated)
//   V    - pointer to an n x n matrix (should be pre-allocated)
void householder_bidiag(int m, int n, double *A, double *U, double *V);

#endif