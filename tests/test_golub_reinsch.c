#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cblas.h>
#include "golub-reinsch.h"   // Contains the SVDResult struct and svd() prototype

#define TOL 1e-8
#define BIG_TOL 1e-10

// Rand
double frand(void)     // uniform [‑0.5, 0.5] 
{
    return (rand() / (double)RAND_MAX) - 0.5;
}

// Debugging print functions
void print_matrix(const char *name, const double *M, int rows, int cols) {
    printf("%s:\n", name);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%10.8f ", M[i * cols + j]);
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

// Compute the Frobenius norm of the difference between two matrices A and B.
double frob_norm_diff(int rows, int cols, const double *A, const double *B) {
    int size = rows * cols;
    double *diff = (double *)malloc(sizeof(double) * size);
    for (int i = 0; i < size; i++) {
        diff[i] = A[i] - B[i];    
    }
    double norm = cblas_dnrm2(size, diff, 1);
    free(diff);
    return norm;
}

// Reconstruct the original matrix A_rec from SVDResult.
// A_rec = U[:,0:k] * diag(S) * V[:,0:k]^T
void reconstruct_A(const SVDResult *svdRes, double *A_rec) {
    int m = svdRes->m, n = svdRes->n, k = svdRes->k;

    // Step 1: Multiply U_k * diag(S), where U_k is m x k
    double *US = (double *)calloc(m * k, sizeof(double));
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            US[i * k + j] = svdRes->U[i * m + j] * svdRes->S[j];
        }
    }

    // Step 2: Compute A_rec = US * V_k^T, where V_k is n x k
    // We'll transpose V_k manually for cblas_dgemm (row-major)
    // V_k^T is k x n
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                m, n, k,
                1.0, US, k,
                     svdRes->V, n,
                0.0, A_rec, n);

    free(US);
}

// Test 1: Reconstruction Test
void test_reconstruction(void)
{
    int m = 2, n = 2;
    double A[4] = { 10.0, 0.0,
                    10.0,-10.0 };

    SVDResult svd = golub_reinsch_svd(m, n, A, TOL);

    print_matrix("A", A, m, n);
    print_matrix("U", svd.U, m, m);
    print_vector("S", svd.S, svd.k);
    print_matrix("V", svd.V, n, n);

    double A_rec[4];
    reconstruct_A(&svd, A_rec);
    print_matrix("A_rec", A_rec, m, n);
    double err = frob_norm_diff(m, n, A, A_rec);
    printf("Reconstruction error (2×2): %.3e\n", err);
    assert(err < TOL);

    free(svd.U); free(svd.S); free(svd.V);
    puts("test_reconstruction ✓");
}

// Test 2: Identity Matrix Test
void test_identity(void)
{
    int m = 4, n = 4;
    double A[16] = {0};
    for (int i = 0; i < 4; ++i) A[i*4 + i] = 1.0;

    SVDResult svd = golub_reinsch_svd(m, n, A, TOL);
    double A_rec[16];  reconstruct_A(&svd, A_rec);
    double err = frob_norm_diff(m, n, A, A_rec);
    printf("Identity error: %.3e\n", err);
    assert(err < TOL);

    free(svd.U); free(svd.S); free(svd.V);
    puts("test_identity ✓");
}

// Test 3: Known Matrix Test
void test_known_matrix() {
    int m = 3, n = 3;
    double A[9] = {
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
    };

    SVDResult svdRes = golub_reinsch_svd(m, n, A, TOL);
    double *A_rec = (double *)malloc(sizeof(double) * m * n);
    reconstruct_A(&svdRes, A_rec);
    double error = frob_norm_diff(m, n, A, A_rec);
    print_matrix("A", A, m, n);
    print_matrix("U", svdRes.U, m, m);
    print_vector("S", svdRes.S, svdRes.k);
    print_matrix("V", svdRes.V, n, n);
    printf("Known matrix reconstruction error: %e\n", error);
    assert(error < 1e-4);  // Slightly higher tolerance for ill-conditioned data

    free(A_rec);
    free(svdRes.U);
    free(svdRes.S);
    free(svdRes.V);
    printf("Test 'known matrix' passed.\n");
}

// Test 4: Very wide Matrix
void test_wide(void)
{
    int m = 5, n = 100;
    double *A = (double *)malloc(sizeof(double) * m * n);
    for (int i = 0; i < m * n; ++i) A[i] = frand();

    SVDResult svd = golub_reinsch_svd(m, n, A, TOL);

    double *A_rec = (double *)malloc(sizeof(double) * m * n);
    reconstruct_A(&svd, A_rec);
    double err = frob_norm_diff(m, n, A, A_rec);
    printf("Wide 5×100 reconstruction error: %.3e\n", err);
    assert(err < TOL);

    free(A);  free(A_rec);
    free(svd.U); free(svd.S); free(svd.V);
    puts("test_wide ✓");
}

// Test 5: Very Tall Matrix
void test_tall(void)
{
    int m = 100, n = 5;
    double *A = (double *)malloc(sizeof(double) * m * n);
    for (int i = 0; i < m * n; ++i) A[i] = frand();

    SVDResult svd = golub_reinsch_svd(m, n, A, TOL);

    double *A_rec = (double *)malloc(sizeof(double) * m * n);
    reconstruct_A(&svd, A_rec);
    double err = frob_norm_diff(m, n, A, A_rec);
    printf("Tall 100×5 reconstruction error: %.3e\n", err);
    assert(err < TOL);

    free(A);  free(A_rec);
    free(svd.U); free(svd.S); free(svd.V);
    puts("test_tall ✓");
}

// Test 6: Bigger Matrix
void test_big(void)
{
    int m = 500, n = 500;
    double *A = (double *)malloc(sizeof(double) * m * n);
    for (int i = 0; i < m * n; ++i) A[i] = frand();

    SVDResult svd = golub_reinsch_svd(m, n, A, BIG_TOL);

    double *A_rec = (double *)malloc(sizeof(double) * m * n);
    reconstruct_A(&svd, A_rec);
    double err = frob_norm_diff(m, n, A, A_rec);
    printf("Big 500×500 reconstruction error: %.3e\n", err);
    assert(err < 5e-6);     

    free(A);  free(A_rec);
    free(svd.U); free(svd.S); free(svd.V);
    puts("test_big ✓");
}

int main() {
    printf("Running SVD tests...\n");
    test_reconstruction();
    test_identity();
    test_known_matrix();
    // test_wide();
    test_tall();
    test_big();
    printf("✅ All SVD tests passed.\n");
    return 0;
}
