#include <stdio.h>
#include <cblas.h>

int main() {
    // A: 2x3 matrix
    double A[6] = {
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0
    };

    // B: 3x2 matrix
    double B[6] = {
        7.0,  8.0,
        9.0, 10.0,
        11.0, 12.0
    };

    // C: Result will be 2x2
    double C[4] = {0};

    // cblas_dgemm params:
    // Order: Row-major
    // TransA: No
    // TransB: No
    // M: rows of A (and C)
    // N: cols of B (and C)
    // K: cols of A / rows of B
    // alpha: 1.0 (scaling for A*B)
    // A, lda: matrix A and its leading dimension (3)
    // B, ldb: matrix B and its leading dimension (2)
    // beta: 0.0 (scaling for existing C)
    // C, ldc: output matrix and its leading dimension (2)

    cblas_dgemm(
        CblasRowMajor, // storing 2d matrix as rows in memory
        CblasNoTrans,  // no transpose
        CblasNoTrans,
        2, 2, 3,
        1.0, A, 3,
        B, 2,
        0.0, C, 2
    );

    printf("Result matrix C:\n");
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            printf("%.1f ", C[i * 2 + j]);
        }
        printf("\n");
    }

    return 0;
}
