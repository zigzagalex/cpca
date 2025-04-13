#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <float.h>
#include <math.h>
#include <cblas.h>
#include "householder.h"  // Contains BidiagResult and householder_bidiag()

#define MAX_ITER 1000

typedef struct {
    int m;      // number of rows of the original matrix A
    int n;      // number of columns of the original matrix A
    int k;      // min(m,n)
    double *U;  // Left singular vectors, stored in row-major (size: m x m)
    double *S;  // Singular values (vector of length k)
    double *V;  // Right singular vectors, stored in row-major (size: n x n)
} SVDResult;

// Compute a Givens rotation that zeroes out b given a and b,
// returning cosine and sine values in c and s respectively.
void givens_rotation(double a, double b, double *c, double *s) {
    if (fabs(b) < 1e-16) {
        *c = 1.0;
        *s = 0.0;
    } else if (fabs(b) > fabs(a)) {
        double r = hypot(a, b);
        *c = a / r;
        *s = -b / r;
    } else {
        double t = -b / a;
        *c = 1.0 / sqrt(1.0 + t * t);
        *s = *c * t;
    }
}

// Applies a Givens rotation (c, s) to columns i and j of matrix M (row-major)
// M is rows x cols, and the rotation is applied across all rows
void apply_givens_to_cols(double *M, int rows, int cols, int i, int j, double c, double s) {
    for (int row = 0; row < rows; row++) {
        double temp_i = M[row * cols + i];
        double temp_j = M[row * cols + j];
        M[row * cols + i] = c * temp_i - s * temp_j;
        M[row * cols + j] = s * temp_i + c * temp_j;
    }
}

// Compute the implicit shift mu using the bottom 2x2 submatrix of the bidiagonal matrix.
// Here, d[k-1] and d[k] are the two diagonal entries and e[k-1] is the off-diagonal element.
double compute_shift(int k, int l, double *d, double *e) {
    // Use the bottom-right 2x2 block; here k is the current index (with 1 <= l < k <= n-1)
    double dk1 = d[k - 1];
    double dk = d[k];
    double ek = e[k - 1];
    double delta = (dk1 - dk) / 2.0;
    double sign = (delta >= 0) ? 1.0 : -1.0;
    double mu = dk - (ek * ek) / (delta + sign * sqrt(delta * delta + ek * ek));
    return mu;
}

// Sanity checking orthogonality of a matrix 
bool is_orthogonal(const double *Q, int n, double tol) {
    double *QtQ = (double *)calloc(n * n, sizeof(double));
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                n, n, n,
                1.0, Q, n,
                     Q, n,
                0.0, QtQ, n);
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double expected = (i == j) ? 1.0 : 0.0;
            if (fabs(QtQ[i * n + j] - expected) > tol) {
                free(QtQ);
                return false;
            }
        }
    }

    free(QtQ);
    return true;
}

// Blocking
void find_active_block(double *d, double *e, int *p, int *q, int k_dim) {
    int start = 0;
    while (start < k_dim - 1 && e[start] == 0) start++;
    int end = start;
    while (end < k_dim - 1 && e[end] != 0) end++;
    *p = start;
    *q = end;
}

/*
  Golub–Reinsch SVD for a bidiagonal matrix (assumed square, dimension n).
  
  Inputs:
    n       - dimension of the bidiagonal matrix.
    d       - array of length n holding the diagonal elements.
    e       - array of length n-1 holding the superdiagonal elements.
              (We assume e[n-1] is not used; you can set it to 0.)
    *A      - m x n input Matrix
    epsilon - machine precision threshold for convergence.
    
  After running, d will hold the singular values (not necessarily sorted),
  and U, V will contain the accumulated rotations such that the original
  bidiagonal matrix B ≈ U * Σ * V^T.
*/
SVDResult golub_reinsch_svd(int m, int n, const double *A, double epsilon) {
    // Bidiagonalize A
    BidiagResult bidiag = householder_bidiag(m, n, A);
    int k_dim = (m < n) ? m : n;

    // Allocate arrays for the diagonal (d) and the superdiagonal (e) of the bidiagonal matrix.
    double *d = (double *)malloc(sizeof(double) * k_dim);
    double *e = NULL;
    if (k_dim > 1) {
        e = (double *)malloc(sizeof(double) * (k_dim- 1));
    }

    // Allocate memory for the matrices
    double *U = (double *)malloc(sizeof(double) * m * m);
    double *V = (double *)malloc(sizeof(double) * n * n);
    memcpy(U, bidiag.U, sizeof(double) * m * m);  // copy result from bidiag
    memcpy(V, bidiag.V, sizeof(double) * n * n);
    double *S = (double *)malloc(sizeof(double) * k_dim);

    // Checking orthogonality 
    printf("U orthogonal? %s\n", is_orthogonal(U, m, 1e-10) ? "YES" : "NOPE");
    printf("V orthogonal? %s\n", is_orthogonal(V, n, 1e-10) ? "YES" : "NOPE");


    // Extract the diagonal and superdiagonal from bidiag.B.
    // We assume that bidiag.B is stored in row-major order as an m x n matrix.
    // For indices i = 0,...,k-1, the diagonal element is at B[i * n + i].
    // For i = 0,...,k-2, the superdiagonal element is at B[i * n + i + 1].
    for (int i = 0; i < k_dim; i++) {
        d[i] = bidiag.B[i * n + i];
        if (i < k_dim - 1 && e != NULL) {
            e[i] = bidiag.B[i * n + i + 1];
        }
    }

    // Process the bidiagonal matrix block-by-block.
    // 2b
    int p = 0, q = 0;
    find_active_block(d, e, &p, &q, k_dim);
    // p: start index; q: first index where off-diagonal is zero (or q == k_dim if never zero)
    while (p < k_dim) {
        // If the block is trivial (a 1x1 block), skip it.
        if (p == q) {
            p = q + 1;
            if (p < k_dim)
                find_active_block(d, e, &p, &q, k_dim);
            continue;
        }
        // Process active block [p, q] with implicit QR steps.
        int iter = 0;
        while (true) {
            // Zero out any tiny off-diagonals within the block.
            for (int i = p; i < q; i++) {
                if (fabs(e[i]) <= epsilon * (fabs(d[i]) + fabs(d[i+1])))
                    e[i] = 0.0;
            }
            // Check if the block has split (i.e. an off-diagonal is exactly zero).
            int split_found = -1;
            for (int i = p; i < q; i++) {
                if (e[i] == 0.0) {
                    split_found = i;
                    break;
                }
            }
            // If the block has split, break out to re-assess the active blocks.
            if (split_found != -1)
                break;

            if (iter++ >= MAX_ITER) {
                fprintf(stderr, "SVD failed to converge for block [%d, %d] after %d iterations\n", p, q, MAX_ITER);
                exit(1);
            }

            // Compute the implicit shift mu using the bottom-right 2x2 submatrix of the block.
            // Here we use indices (q-1, q) since q is the last index in the active block.
            double mu = compute_shift(q, p, d, e);

            // Perform one QR sweep over the block.
            double x = d[p] - mu;
            double z = e[p];
            double c, s;
            for (int i = p; i < q; i++) {
                givens_rotation(x, z, &c, &s);
                
                // Apply right rotation to update d[i] and e[i].
                double temp_d = d[i];
                double temp_e = e[i];
                d[i] = c * temp_d - s * temp_e;
                e[i] = s * temp_d + c * temp_e;
                
                // If applicable, update the next diagonal element.
                if (i + 1 <= q) {
                    double temp_d_next = d[i + 1];
                    d[i + 1] = c * temp_d_next;
                }
                
                // Prepare values for the next rotation.
                x = d[i];
                // For all but the last element, use e[i+1] for the next z.
                z = (i < q - 1) ? e[i + 1] : 0.0;
                
                // Accumulate the Givens rotation into the singular vector matrices.
                apply_givens_to_cols(V, n, n, i, i + 1, c, s);
                apply_givens_to_cols(U, m, m, i, i + 1, c, s);
            }
            // Final adjustment: if the last computed z is small, set the corresponding off-diagonal to 0.
            if (fabs(z) <= epsilon * (fabs(d[q - 1]) + fabs(d[q])))
                e[q - 1] = 0.0;
        } // End processing for current block.

        // Recompute the next active block.
        find_active_block(d, e, &p, &q, k_dim);
    } // End processing of all blocks.


    // Post-process: make singular values nonnegative.
    for (int i = 0; i < k_dim; i++) {
        if (d[i] < 0) {
            d[i] = -d[i];
            for (int j = 0; j < n; j++) {
                V[j*n + i] = -V[j*n + i];
            }
        }
    }

        // Store singular values in S.
    for (int i = 0; i < k_dim; i++) {
        S[i] = d[i];
    }

    // Free temporary resources.
    free(bidiag.B);
    free(d);
    if (e != NULL) free(e);

    SVDResult result = {m,n,k_dim,U,S,V};
    return result;
}
