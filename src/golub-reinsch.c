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
double compute_shift(int m, int n, int k, int l, double *B) {
    // Use the bottom-right 2x2 block; here k is the current index (with 1 <= l < k <= n-1)
    double a = B[(k - 1)*n+(k-1)];
    double b = B[k*n+k];
    double d = B[(k - 1)*n+k];
    // Compute eigenvalues of [a,b,0,d] and set mu to the cosest one to d
    //
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
void find_active_block(double *B, int *p, int *q, int min_mn) {
    int i = 0;
    while (i < min_mn - 1 && B[i*n+i+1] == 0) i++;
    int j = i;
    while (j < min_mn - 1 && B[j*n+j+1] != 0) j++;
    *p = start;
    *q = end;
}

/*
    Golub–Reinsch SVD for a bidiagonal matrix (assumed square, dimension n).

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
            2.6.4
*/
SVDResult golub_reinsch_svd(int m, int n, const double *A, double epsilon) {
    // Bidiagonalize A
    BidiagResult bidiag = householder_bidiag(m, n, A);
    int min_mn = (m < n) ? m : n;

    // Allocate memory for the matrices
    double *U = (double *)malloc(sizeof(double) * m * m);
    double *V = (double *)malloc(sizeof(double) * n * n);
    memcpy(U, bidiag.U, sizeof(double) * m * m);  // copy result from bidiag
    memcpy(V, bidiag.V, sizeof(double) * n * n);
    double *S = (double *)malloc(sizeof(double) * min_mn);

    // Checking orthogonality 
    printf("U orthogonal? %s\n", is_orthogonal(U, m, 1e-10) ? "YES" : "NOPE");
    printf("V orthogonal? %s\n", is_orthogonal(V, n, 1e-10) ? "YES" : "NOPE");


    // 2.2-2.5
    int p = 0;
    int q = 0;
    find_active_block(bidiag.B, &p, &q, min_mn);

    while(p<min_mn && q!=0){
        // If the block is trivial (a 1x1 block), skip it.
        if (p == q) {
            p = q + 1;
            if (p < min_mn)
                find_active_block(bidiag.B, &p, &q, min_mn);
            continue;
        }
        // 2.1
        int iter = 0;
        while (true) {
            // Zero out any epsilon small off-diagonals within the block.
            for (int i = p; i < q; i++) {
                if (fabs(bidiag.B[i*n+i+1]) <= epsilon * (fabs(bidiag.B[i*n+i]) + fabs(bidiag.B[(i+1)*n+i+1])))
                    bidiag.B[i*n+i+1] = 0.0;
            }
            // Check if the block has split (i.e. an off-diagonal is exactly zero).
            int split_found = -1;
            for (int i = p; i < q; i++) {
                if (bidiag.B[i*n+i+1] == 0.0) {
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
            for (int i=p; i<q;i++){
                if (bidiag.B[i*n+i]==0){
                    // Apply Givens rotation to zero out B[i*n+i+1] and break to reassess blocks
                }
                else {
                    // 2.6
                    // get mu
                    // set alpha and beta
                    // find c and s such that the quation holds
                    // B*R(c,s) where Givens rotation that acts on columns i, i+1
                    // Update V with V*R
                    // set alpha and beta to B[i,i] and B[i,i+1]
                    // find c and s such that the quation holds
                    // R(c,-s)*B where Givens rotation that acts on rows i, i+1
                    // Update U with R*U
                    // update alpha and beta to B[i,i+1] and B[i,i+2]
                }
            }

    }

    // Free temporary resources.


    SVDResult result = {m,n,min_mn,U,S,V};
    return result;
}
