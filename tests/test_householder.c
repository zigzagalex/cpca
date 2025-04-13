#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <cblas.h> 
#include "householder.h"  // contains BidiagResult + householder_bidiag()

#define TOL 1e-6

// Utility function: Checks if two doubles are nearly equal.
int nearly_equal(double a, double b, double tol) {
    return fabs(a - b) < tol;
}

// Debugging print functions
void print_matrix(const char *name, const double *M, int rows, int cols) {
    printf("%s:\n", name);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%10.4f ", M[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void print_vector(const char *name, const double *V, int len) {
    printf("%s:\n", name);
    for (int i = 0; i < len; i++) {
        printf("%10.4f ", V[i]);
    }
    printf("\n\n");
}


/* ------------------------------------------------------------------------
 * Test 1: Zero Column Test
 * ------------------------------------------------------------------------ */
void test_zero_column() {
    int m = 3, n = 2;
    double A[6] = {
        0.0, 1.0,
        0.0, 2.0,
        0.0, 3.0
    };

    BidiagResult result = householder_bidiag(m, n, A);

    // Check that the (0,0) entry in B is still 0 (since original column was all zero)
    assert(nearly_equal(result.B[0], 0.0, TOL));

    free(result.B);
    free(result.U);
    free(result.V);

    printf("Test 'test_zero_column' passed.\n");
}

/* ------------------------------------------------------------------------
 * Test 2: 1x1 Matrix Test
 * ------------------------------------------------------------------------ */
void test_1x1() {
    int m = 1, n = 1;
    double A[1] = { 42.0 };

    BidiagResult result = householder_bidiag(m, n, A);

    assert(nearly_equal(fabs(result.B[0]), 42.0, TOL));
    assert(nearly_equal(fabs(result.U[0]), 1.0, TOL));  // U should be [-1] or [1]
    assert(nearly_equal(fabs(result.V[0]), 1.0, TOL));  // V should be [1]

    free(result.B);
    free(result.U);
    free(result.V);

    printf("Test 'test_1x1' passed.\n");
}

/* ------------------------------------------------------------------------
 * Test 3: Identity Matrix Test
 * ------------------------------------------------------------------------ */
void test_identity() {
    int m = 4, n = 4;
    double A[16] = {0};

    // Identity matrix
    for (int i = 0; i < 4; i++) {
        A[i * 4 + i] = 1.0;
    }

    BidiagResult result = householder_bidiag(m, n, A);

    // Expect main diagonal ≈ 1.0, everything else ≈ 0.0
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            double val = result.B[i * n + j];
            if (i == j) {
                assert(nearly_equal(fabs(val), 1.0, TOL));
            } else if (i == j - 1) {
                // superdiagonal is allowed to be nonzero in bidiagonal
                continue;
            } else {
                assert(nearly_equal(val, 0.0, TOL));
            }
        }
    }

    free(result.B);
    free(result.U);
    free(result.V);

    printf("Test 'test_identity' passed.\n");
}

/* ------------------------------------------------------------------------
 * Test 4: Reconstruction of A using the results up to TOLARANCE
 * ------------------------------------------------------------------------ */
void test_bidiag_reconstruction() {
    int m = 4, n = 3;
    double A[12] = {
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
        10.0, 11.0, 12.0
    };

    BidiagResult result = householder_bidiag(m, n, A);

    // U: m x m, B: m x n, V: n x n → we want A_rec = U * B * Vᵀ

    // Step 1: T = B * Vᵀ (B: m x n, Vᵀ: n x n → T: m x n)
    double *T = (double *)calloc(m * n, sizeof(double));
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                m, n, n,
                1.0, result.B, n,
                     result.V, n,
                0.0, T, n);

    // Step 2: A_rec = U * T (U: m x m, T: m x n → A_rec: m x n)
    double *A_rec = (double *)calloc(m * n, sizeof(double));
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                m, n, m,
                1.0, result.U, m,
                     T, n,
                0.0, A_rec, n);

    // Compare A_rec to original A
    double error = 0.0;
    for (int i = 0; i < m * n; i++) {
        double diff = A[i] - A_rec[i];
        error += diff * diff;
    }
    error = sqrt(error);

    print_matrix("Original A", A, m, n);
    print_matrix("Reconstructed A", A_rec, m, n);
    printf("Bidiagonal reconstruction error (Frobenius norm): %e\n", error);

    assert(error < TOL);

    free(result.B);
    free(result.U);
    free(result.V);
    free(T);
    free(A_rec);

    printf("Test 'test_bidiag_reconstruction' passed.\n");
}

/* ------------------------------------------------------------------------
 * main()
 * ------------------------------------------------------------------------ */
int main() {
    printf("Running unit tests for Householder bidiagonalization...\n");
    test_zero_column();
    test_1x1();
    test_identity();
    test_bidiag_reconstruction();  
    printf("All tests passed.\n");
    return 0;
}
