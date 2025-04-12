#ifndef BIDIOG_H
#define BIDIOG_H

// Perform Householder bidiagonalization on matrix A (m x n).
// On exit, A is overwritten with the bidiagonal matrix B, and U, V are filled with the
// products of Householder reflectors such that A_original = U * B * Vᵀ.

// Struct to hold the result of the bidiagonalization
typedef struct {
    int m;
    int n;
    double *B;  // bidiagonal matrix, size m x n
    double *U;  // orthogonal left matrix, size m x m
    double *V;  // orthogonal right matrix, size n x n
} BidiagResult;

// Main function for computing bidiagonal form via Householder transformation
// Parameters:
//   m    - number of rows in A
//   n    - number of columns in A
//   A    - pointer to the matrix A (stored row-major)

BidiagResult householder_bidiag(int m, int n, const double *A_input);

#endif
