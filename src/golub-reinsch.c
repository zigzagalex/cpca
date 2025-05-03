#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <float.h>
#include <math.h>
#include <cblas.h>
#include "householder.h"  // Contains BidiagResult and householder_bidiag()

#define MAX_ITER 1000

// STRUCTS
typedef struct {
    int m;      // number of rows of the original matrix A
    int n;      // number of columns of the original matrix A
    int k;      // min of m, n
    double *U;  // Left singular vectors, stored in row-major (size: m x m)
    double *S;  // Singular values (vector of length k)
    double *V;  // Right singular vectors, stored in row-major (size: n x n)
} SVDResult;


// HELPER FUNCTIONS
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

// Apply a Givens rotation (c, s) to columns i and j of matrix M (row-major)
void apply_givens_to_cols(double *M, int rows, int cols, int i, int j, double c, double s) {
    for (int row = 0; row < rows; ++row) {
        double tmp_i = M[row * cols + i];
        double tmp_j = M[row * cols + j];
        M[row * cols + i] =  c * tmp_i - s * tmp_j;
        M[row * cols + j] =  s * tmp_i + c * tmp_j;
    }
}

// Apply a Givens rotation (c, s) to rows i and j of matrix M (row‑major).
void apply_givens_to_rows(double *M, int rows, int cols, int i, int j, double c, double s) {
    for (int col = 0; col < cols; ++col) {
        double tmp_i = M[i * cols + col];
        double tmp_j = M[j * cols + col];
        M[i * cols + col] =  c * tmp_i - s * tmp_j;
        M[j * cols + col] =  s * tmp_i + c * tmp_j;
    }
}

// Compute the implicit Wilkinson shift μ using the bottom‑right 2×2 block
// of BᵀB, returning √λ (because the algorithm works on B, not BᵀB).
double compute_shift(int n, int q, double *B) {
    if (q < 1) return 0.0;
    // k is the last *diagonal* index of the active block. 
    double f = B[(q-1) * n + (q-1)];   
    double g = B[(q-1) * n + q];  
    double h = B[q*n + q]; 

    double a11 = f*f + g*g;
    double a12 = f*g;
    double a22 = h*h + g*g;

    double tr   = a11 + a22;
    double det  = a11*a22 - a12*a12;
    double disc = sqrt(fmax(0.0, tr*tr*0.25 - det));

    double λ1 = 0.5 * tr + disc;
    double λ2 = 0.5 * tr - disc;

    /* Choose the eigen‑value closer to a22 (the bottom‑right element) */
    double λ = (fabs(λ1 - a22) < fabs(λ2 - a22)) ? λ1 : λ2;
    return sqrt(fmax(0.0, λ));
}

// Checking orthogonality of a matrix 
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

// Finding active block
void find_active_block(double *B, int n, int *p, int *q, int min_mn) {
    int i = 0;
    while (i < min_mn - 1 && B[i*n+i+1] == 0) i++;
    int j = i;
    while (j < min_mn - 1 && B[j*n+j+1] != 0) j++;
    *p = i;
    *q = j;
}

/*
    Golub–Reinsch SVD for a bidiagonal matrix.

    Inputs: 
        m: number of rows in matrix 
        n: number of columns in matrix
        *A: m x n matrix
        epsilon: percision for setting close to zero elements to zero for convergence
    
    Output: 
        SVDResult: struct containing
            m: number of rows in matrix 
            n: number of columns in matrix
            U: m x n orthogonal matrix containing accumulated Householder and Givens rotations from the left (eigenvectors of A^T*A)
            S: n x n diagonal matrix containing singular values (eigenvalues of A^T*A)
            V: n x n orthogonal matrix containing accumulated Householder and Givens rotations from the right (eigenvectors of A*A^T)
    
  
    Steps in algorithm: 
    1. Compute A = U' * B * V'^T using Householder reflections so that U', V' orthogonal and B is bidiagonal
    2. Loop through these steps for i from 0 to min(m,n): 
        2.1 If B[i,i+1] < epsilon * (abs(B[i,i])+abs(B[i+1, i+1])) set B[i, i+1]=0
        2.2 Find biggest q s.t. B[i,i+1]=0 for any q<=i<=n if not found set 0
        2.3 Break if q = 0 (i.e. B is already diagonal)
        2.4 Find smallest p s.t. B[i,i+1]!=0 for any in p+1<=i<=q-1 if not found set 0
        2.5 For any p+1<=i<=q-1 if B[i,i]=0 apply a left Givens rotation to zero out B[i,i+1] and update U' and V'
        2.6 Else apply the Gholub Kahan step for the submatrix B2 from [p+1,p+1] to [q-1, 1-1] and update U' and V'
            2.6.1 Set C = lower 2x2 matrix of B2^T*B2
            2.6.2 Compute eigenvalues lambda_1, lambda_2 of C and set mu= lambda_? s.t. the eigenvalue is closer to C[2,2]
            2.6.3 Set alpha = B[p+1,p+1]^2-mu and beta = B[p+1,p+1]*B[p+1,p+2]
            2.6.4 Propagate the bulge with a nother left ant right givens rotation pushing the bulge down and right one row
*/
SVDResult golub_reinsch_svd(int m, int n, const double *A, double epsilon) {
    // Bidiagonalize A
    BidiagResult bidiag = householder_bidiag(m, n, A);
    int min_mn = (m < n) ? m : n;

    // Convenience pointers
    double *B = bidiag.B;
    double *U = bidiag.U;
    double *V = bidiag.V;

    // 2.2-2.6
    int p, q;
    find_active_block(B, n, &p, &q, min_mn);

    while(p<min_mn && q!=0){
        // If the block is trivial (a 1x1 block), skip it.
        if (p == q) {
            p = q + 1;
            if (p < min_mn)
                find_active_block(B, n, &p, &q, min_mn);
            continue;
        }
        // 2.1
        int iter = 0;
        while (true) {
            // Zero out any epsilon small off-diagonals within the block.
            for (int i = p; i < q; i++) {
                if (fabs(B[i*n+i+1]) <= epsilon * (fabs(B[i*n+i]) + fabs(B[(i+1)*n+i+1])))
                    B[i*n+i+1] = 0.0;
            }
            // Check if the block has split (i.e. an off-diagonal is exactly zero).
            int split_found = -1;
            for (int i = p; i < q; i++) {
                if (B[i*n+i+1] == 0.0) {
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

            // 2.5
            // Handling pivots with zero diagonal element
            bool handled_zero = false;
            for (int i=p; i<q;i++){
                if (B[i*n+i]==0){
                    double c, s;
                    givens_rotation(0.0, B[i*n + i + 1], &c, &s);
                    // Apply givens left rotation to B and U: L*B and L*U
                    apply_givens_to_rows(B, min_mn, n, i, i+1, c, s);
                    apply_givens_to_rows(U, m, m, i, i+1,  c, -s);
                    handled_zero = true;
                    break;
                }
            }
            if (handled_zero) continue;

            // 2.6 
            // Standard Golub‑Kahan step on the (p,q) block
            // Get mu
            double mu = compute_shift(n, q, B);

            // Set alpha and beta
            double alpha = B[p*n + p];
            double beta  = B[p*n + p + 1];
            
            // Find c and s such that the quation holds
            double c, s;
            // B*R(c,s) where Givens rotation that acts on columns i, i+1
            givens_rotation(alpha*alpha - mu*mu, alpha*beta, &c, &s);

            // Update V with V*R and B*G
            if (p+1 < n) {
                apply_givens_to_cols(B, min_mn, n, p, p+1, c, s);
                apply_givens_to_cols(V, n, n, p, p+1, c, s);
            }
            
            // Propagate the bulge 
            for (int k = p; k < q; ++k) {
                // left rotation to push bulge down one row 
                givens_rotation(B[k*n + k], B[(k+1)*n + k], &c, &s);
                apply_givens_to_rows(B, min_mn, n, k, k+1, c, s);
                apply_givens_to_rows(U, m, m, k, k+1, c, -s);

                // right rotation to push bulge right one column 
                if (k < q - 1 && k + 2 < n) {
                    givens_rotation(B[k*n + k + 1], B[k*n + k + 2], &c, &s);
                    apply_givens_to_cols(B, min_mn, n, k+1, k+2, c, s);
                    apply_givens_to_cols(V, n, n, k+1, k+2, c, s);
                }
            }
        }
        // Re‑identify the next active block 
        find_active_block(B, n, &p, &q, min_mn);
    }

    // Sanity checks (optional – comment out in production) 
    printf("U orthogonal?  %s\n", is_orthogonal(U, m, 1e-10) ? "yes" : "no");
    printf("V orthogonal?  %s\n", is_orthogonal(V, n, 1e-10) ? "yes" : "no");


    // Copy the singular values (absolute diagonal of B)
    double *S = (double *)malloc(sizeof(double) * min_mn);
    if (!S) {
        fprintf(stderr, "Out of memory for S.\n");
        exit(EXIT_FAILURE);
    }
    // Postprocessing for signs
    for (int i=0; i<min_mn;i++){
        if (B[i*n + i] < 0.0) {
            S[i] = -B[i*n + i];      /* positive sigma_i */
        for (int r = 0; r < n; ++r)  /* or m if you choose U */
            V[r*n + i] = -V[r*n + i];
        } else {
            S[i] =  B[i*n + i];
        }
    }

    int k = min_mn;

    // Free
    free(B);

    SVDResult result = {m,n,k,U,S,V};
    return result;
}
