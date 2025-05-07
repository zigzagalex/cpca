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


/* 
    Bidiagonalization of an m x n matrix using Householder transformations i.e. piviting so that the column matches the respective basis vector. 

    Inputs:

    m: int number of rows in matrix
    n: int number of columns in matrix
    *A_input: input matrix in row major form 

    Output: 

    BidiagResult: struct containing  
        int m: number of rows in matrix
        int n: number of columns in matrix
        double *B: bidiagonal matrix (m x n)
        double *U: orthogonal m x m matrix containing all the transformations to the left (columns)
        double *V;  orthogonal n x n matrix containing all the transformations to the right (rows)

    
    Basic steps in algorithms: Set B = A. For each column k from 0 to min(m,n) repeat 1-6
    1. Let v = B[k:,k] i.e. [0,0,0,b_k, b_k+1,...,b_min(m,n)] for column k
    2. Let norm_v = l_2_norm(v)
    3. Update v[0] = v[0]+sgn(v[0])*norm_v
    4. Compute beta =  2 / (v^T v)
    5. Left Householder (to zero out below B[k,k])
        5.1 Apply the Householder reflector to submatrix B[k:m, k:n]: Compute dot = v^T * (column j of submatrix for j from k to n-1)
        5.2 Accumulate the reflector in U (apply U = U * (I - beta*v*v^T to U from the right)).
    6. Right Householder (to zero out right of B[k, k+1])
        6.1 Apply the Householder reflector to submatrix B[k, k+1:n-1] 
        6.2 Accumulate the reflecter in V (V = V * (I - beta2 * [0 I_len2]*v2*v2^T*[0 I_len2]^T))


*/

BidiagResult householder_bidiag(int m, int n, double *A_input) {
    int min_mn = (m < n) ? m : n; // (condition) ? if_true : if_false

    // Allocate result matrices
    double *B = (double *)malloc(sizeof(double) * m * n); 
    double *U = (double *)malloc(sizeof(double) * m * m);
    double *V = (double *)malloc(sizeof(double) * n * n);

    // Copy input matrix so we don't overwrite user input
    for (int i = 0; i < m * n; i++) B[i] = A_input[i];

    // Initialize U and V as identity matrices.
    init_identity(U, m);
    init_identity(V, n);

    // Loop over each column (and row) for bidiagonalization.
    for (int k = 0; k < min_mn; k++) {
        // ========= Left Householder (to zero out below B[k][k]) =========
        int len = m - k;  // Length of vector for column k.
        double *v = (double *)malloc(len * sizeof(double));

        // Extract the vector from B (column k, rows k to m-1).
        for (int i = 0; i < len; i++) {
            v[i] = B[(k+i)*n + k];
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

        // Apply the Householder reflector to submatrix B[k:m, k:n].
        for (int j = k; j < n; j++) {
            // Compute dot = v^T * (column j of submatrix).
            double dot = cblas_ddot(len, v, 1, &B[k*n + j], n);
            for (int l = 0; l < len; l++) {
                B[(k+l)*n + j] -= beta * v[l] * dot;
            }
        }
        
        // Accumulate the reflector in U (apply H = I - beta*v*v^T to U from the right).
        // Only the part from row i to m-1 is affected.
        for (int r = 0; r < m; r++) {
            double dot = cblas_ddot(len, &U[r*m + k], 1, v, 1);
                for (int l = 0; l < len; l++) {
                    U[r*m + k + l] -= beta * dot * v[l];
                }
            }

        free(v);

        // ========= Right Householder (to zero out right of B[k, k+1] if possible) =========
        if (k < n - 1) {
            int len2 = n - k - 1;  // Length for row reflector.
            double *v2 = (double *)malloc(len2 * sizeof(double));

            // Extract row i elements from column k+1 to n-1.
            for (int j = 0; j < len2; j++) {
                v2[j] = B[k*n + (k+1+j)];
            }

            double norm_v2 = cblas_dnrm2(len2, v2, 1);
            double sign2 = (v2[0] >= 0.0) ? 1.0 : -1.0;
            v2[0] += sign2 * norm_v2;
            double v2tv2 = cblas_ddot(len2, v2, 1, v2, 1);
            double beta2 = (v2tv2 == 0.0) ? 0.0 : 2.0 / v2tv2;

            // Apply the reflector to submatrix B[k:m, k+1:n].
            for (int l = k; l < m; l++) {
                double dot = cblas_ddot(len2, &B[l*n + (k+1)], 1, v2, 1);
                for (int j = 0; j < len2; j++) {
                    B[l*n + (k+1+j)] -= beta2 * dot * v2[j];
                }
            }

            // Accumulate the reflector in V.
            // Update V = V * (I - beta2 * [0 I_len2]*v2*v2^T*[0 I_len2]^T)
            // Only columns i+1 to n-1 are affected.
            for (int r = 0; r < n; r++) {
                double dot = cblas_ddot(len2, &V[r*n + (k+1)], 1, v2, 1);
                for (int j = 0; j < len2; j++) {
                    V[r*n + (k+1+j)] -= beta2 * dot * v2[j];
                }
            }

            free(v2);
        }
    }

    BidiagResult result = {m,n,B,U,V};
    return result;
    
}
