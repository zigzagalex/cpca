#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cblas.h>
#include "cpca_utils.h"

// Pretty print                                                */
void pca_pretty_print(const PCAResult *p)
{
    int k = p->k;
    int n = p->n;

    // 1. Importance of components table
    puts("\nImportance of components:");
    // header 
    printf("%-20s", "");
    printf("       PC1");
    for (int j = 1; j < k; ++j) printf("       PC%-5d", j+1);
    puts("");

    // Std dev 
    printf("%-20s", "  Standard deviation   ");
    for (int j = 0; j < k; ++j) printf("%10.4f", p->stddev[j]);
    puts("");

    // PVE 
    printf("%-20s", "  Proportion Variance  ");
    for (int j = 0; j < k; ++j) printf("%10.4f", p->pve[j]);
    puts("");

    // Cumulative
    printf("%-20s", "  Cumulative Proportion");
    for (int j = 0; j < k; ++j) printf("%10.4f", p->cum[j]);
    puts("\n");

    // 2. Loadings (rotation)
    puts("Rotation (loadings):");
    // col headers
    printf("%-8s", ""); 
    for (int j = 0; j < k; ++j) printf("   PC%-5d", j+1);
    puts("");
    // each variable row
    for (int r = 0; r < n; ++r) {
        printf("Var%-4d", r+1);
        for (int c = 0; c < k; ++c)
            printf("%10.4f", p->loadings[r*k + c]);
        puts("");
    }
    puts("");
}

// X_rec = scores  *  PCAdirections^T
void pca_reconstruct(const PCAResult *p, double *X_rec)
{
    int m = p->m, n = p->n, k = p->k;

    // scores: m×k      PCAdirections^T: k×n  → X_rec m×n
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                m, n, k,
                1.0, p->scores,       k,
                     p->PCAdirections, n,
                0.0, X_rec,           n);
}
