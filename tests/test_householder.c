#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include "householder.h"  // contains BidiagResult + householder_bidiag()

#define TOL 1e-6

// Utility function: Checks if two doubles are nearly equal.
int nearly_equal(double a, double b, double tol) {
    return fabs(a - b) < tol;
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
 * main()
 * ------------------------------------------------------------------------ */
int main() {
    printf("Running unit tests for Householder bidiagonalization...\n");
    test_zero_column();
    test_1x1();
    test_identity();
    printf("All tests passed.\n");
    return 0;
}
