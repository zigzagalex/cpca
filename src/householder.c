#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cblas.h>

// Initialize an identity matrix of size n x n.
void init_identity(double *M, int n) {
    for (int i = 0; i < n * n; i++)
        M[i] = 0.0;
    for (int i = 0; i < n; i++)
        M[i * n + i] = 1.0;
}

typedef struct {
    int m;
    int n;
    double *B;  // bidiagonal matrix (m x n)
    double *U;  // m x m
    double *V;  // n x n
} BidiagResult;


BidiagResult householder_bidiag(int m, int n, double *A_input) {
    int min_mn = (m < n) ? m : n;

    // Allocate result matrices
    double *A = (double *)malloc(sizeof(double) * m * n); // Will become B
    double *U = (double *)malloc(sizeof(double) * m * m);
    double *V = (double *)malloc(sizeof(double) * n * n);

    // Copy input matrix so we don't overwrite user input
    for (int i = 0; i < m * n; i++) A[i] = A_input[i];

    // Initialize U and V as identity matrices.
    init_identity(U, m);
    init_identity(V, n);

    // Loop over each column (and row) for bidiagonalization.
    for (int i = 0; i < min_mn; i++) {
        // ========= Left Householder (to zero out below A[i][i]) =========
        int len = m - i;  // Length of vector for column i.
        double *v = (double *)malloc(len * sizeof(double));

        // Extract the vector from A (column i, rows i to m-1).
        for (int k = 0; k < len; k++) {
            v[k] = A[(i+k)*n + i];
        }

        // Compute the norm of v.
        double norm_v = cblas_dnrm2(len, v, 1);
        // Determine sign (to avoid cancellation).
        double sign = (v[0] >= 0.0) ? 1.0 : -1.0;
        // Modify the first element.
        v[0] += sign * norm_v;
        // Compute beta = 2 / (v^T v).
        double vtv = cblas_ddot(len, v, 1, v, 1);
        double beta = (vtv == 0.0) ? 0.0 : 2.0 / vtv;

        // Apply the Householder reflector to submatrix A[i:m, i:n].
        for (int j = i; j < n; j++) {
            // Compute dot = v^T * (column j of submatrix).
            double dot = cblas_ddot(len, v, 1, &A[i*n + j], n);
            for (int k = 0; k < len; k++) {
                A[(i+k)*n + j] -= beta * v[k] * dot;
            }
        }
        
        // Accumulate the reflector in U (apply H = I - beta*v*v^T to U from the right).
        // Only the part from row i to m-1 is affected.
        for (int r = 0; r < m; r++) {
            double dot = cblas_ddot(len, &U[r*m + i], 1, v, 1);
                for (int k = 0; k < len; k++) {
                    U[r*m + i + k] -= beta * dot * v[k];
                }
            }

        free(v);

        // ========= Right Householder (to zero out A[i][i+2:n] if possible) =========
        if (i < n - 1) {
            int len2 = n - i - 1;  // Length for row reflector.
            double *v2 = (double *)malloc(len2 * sizeof(double));

            // Extract row i elements from column i+1 to n-1.
            for (int j = 0; j < len2; j++) {
                v2[j] = A[i*n + (i+1+j)];
            }

            double norm_v2 = cblas_dnrm2(len2, v2, 1);
            double sign2 = (v2[0] >= 0.0) ? 1.0 : -1.0;
            v2[0] += sign2 * norm_v2;
            double v2tv2 = cblas_ddot(len2, v2, 1, v2, 1);
            double beta2 = (v2tv2 == 0.0) ? 0.0 : 2.0 / v2tv2;

            // Apply the reflector to submatrix A[i:m, i+1:n].
            for (int k = i; k < m; k++) {
                double dot = cblas_ddot(len2, &A[k*n + (i+1)], 1, v2, 1);
                for (int j = 0; j < len2; j++) {
                    A[k*n + (i+1+j)] -= beta2 * dot * v2[j];
                }
            }

            // Accumulate the reflector in V.
            // Update V = V * (I - beta2 * [0 I_len2]*v2*v2^T*[0 I_len2]^T)
            // Only columns i+1 to n-1 are affected.
            for (int r = 0; r < n; r++) {
                double dot = cblas_ddot(len2, &V[r*n + (i+1)], 1, v2, 1);
                for (int j = 0; j < len2; j++) {
                    V[r*n + (i+1+j)] -= beta2 * dot * v2[j];
                }
            }

            free(v2);
        }
    }

    // At this point:
    // - Matrix A is overwritten with the upper bidiagonal matrix B.
    //   (Diagonal elements in A[i][i], and superdiagonals in A[i][i+1].)
    // - U and V are the accumulations of the Householder reflectors,
    //   so that the original A satisfies: A_original = U * B * V^T.

    BidiagResult result = {m,n,A,U,V};
    return result;
    
}
