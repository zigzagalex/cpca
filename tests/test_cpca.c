#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <cblas.h>

#include "cpca.h"        /* cpca(), PCAResult             */
#include "cpca_utils.h"   /* pca_pretty_print(), recon …   */

#define TOL 1e-10

double frob_norm_diff(int rows,int cols,const double *A,const double *B)
{
    int size = rows*cols;
    double *diff = malloc(sizeof(double)*size);
    for (int i=0;i<size;i++) diff[i]=A[i]-B[i];
    double nrm = cblas_dnrm2(size,diff,1);
    free(diff);  return nrm;
}

// Test 1: 2×2 easy matrix (already mean-zero)
void test_pca_small(void)
{
    int m=2,n=2;
    double A[4] = {-1.0,  2.0,
                    1.0, -2.0};  

    PCAResult p = cpca(A, m, n);

    // Reconstruct and compare 
    double X_rec[4];
    pca_reconstruct(&p, X_rec);
    double err = frob_norm_diff(m,n,A,X_rec);
    printf("small 2×2 reconstruction err = %.3e\n", err);
    assert(err < TOL);

    puts("Pretty print for small matrix:");
    pca_pretty_print(&p);

    free_pca(&p);
    puts("test_pca_small ✓\n");
}

// Test 2: classic toy data (10×2)
void test_pca_jolliffe(void)
{
    int m=10,n=2;
    double X[20] = { 2.5,2.4,  0.5,0.7,  2.2,2.9,  1.9,2.2,  3.1,3.0,
                     2.3,2.7,  2.0,1.6,  1.0,1.1,  1.5,1.6,  1.1,0.9 };

    // copy because cpca() centres in-place
    double *A = malloc(sizeof(double)*m*n);
    memcpy(A, X, sizeof(double)*m*n);

    PCAResult p = cpca(A, m, n);

    // In R first PC explains > 96 %
    assert(p.pve[0] > 0.96);

    puts("Pretty print for Jolliffe toy data:");
    pca_pretty_print(&p);

    free(A);
    free_pca(&p);
    puts("test_pca_jolliffe ✓\n");
}

int main(void)
{
    puts("Running PCA tests …");
    test_pca_small();
    test_pca_jolliffe();
    puts("✅  All PCA tests passed.");
    return 0;
}
