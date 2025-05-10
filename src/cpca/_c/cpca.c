#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <float.h>
#include <math.h>
#include <cblas.h>
#include "golub-reinsch.h"

#define epsilon 1e-10

typedef struct {
    int m, n, k;          // rows, cols, # components (k = min(m,n))
    double *PCAdirections;// V  (nxn) matrix axes that slice through the data cloud so you see the biggest spread
    double *scores;       // T  (m × k)
    double *loadings;     // L  (n × k)
    double *stddev;       // s_k (length k)
    double *pve;          // proportion of variance explained (length k)
    double *cum;          // cumulative PVE (length k)
} PCAResult;


void center(double *A, int m, int n){
    // Column centering the matrix A for PCA on covariance
    // Optionally standerdize it for PCA on correlations
    double *mean = (double *)malloc(n * sizeof *mean);
    if (!mean) return;

    for (int j = 0; j < n; ++j) {
        double sum = 0.0;
        for (int i = 0; i < m; ++i)
            sum += A[i*n + j];
        mean[j] = sum / (double)m;    
    }
    for (size_t j = 0; j < n; ++j)
        for (size_t i = 0; i < m; ++i)
            A[i*n + j] -= mean[j];

    free(mean);
}


// Scores = U(:,1:k) * diag(S)
double *scores_from_us(const double *U, const double *S, int m, int k) {
    double *T = malloc((size_t)m * k * sizeof *T);
    if (!T) return NULL;
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < k; ++j)
            T[i*k + j] = U[i*m + j] * S[j];   
    return T;
}

// Loadings L = V · Σ / √(N−1)
void loadings(const double *V, const double *S, int n, int k, int m, double **loadings_out){
    const double inv_sqrt_nm1 = 1.0 / sqrt((double)(m - 1));
    double *L = malloc((size_t)n * k * sizeof *L);
    if (!L) return NULL;
    for (int j = 0; j < n; ++j)
        for (int c = 0; c < k; ++c)
            L[j*k + c] = V[j*n + c] * S[c] * inv_sqrt_nm1;
    *loadings_out = L;
}

// Standard deviation of each PC s_k = σ_k / √(N−1)
void std_PC(const double *S, int k, int m, double **stddev_out){
    const double inv_sqrt_nm1 = 1.0 / sqrt((double)(m - 1));
    double *stddev = malloc((size_t)k * sizeof *stddev);
    if (!stddev) return NULL;
    for (int c = 0; c < k; ++c)
        stddev[c] = S[c] * inv_sqrt_nm1;
    *stddev_out = stddev;
}

// Proportion of variance explained p_k = σ_k² / Σ_j σ_j² and commulatative proportion of variance explained c_k = Σ_{j ≤ k} p_j
void pve(const double *S, int k, int m, double **pve_out, double **cum_out){
    double *pve  = malloc((size_t)k * sizeof *pve);
    double *cum  = malloc((size_t)k * sizeof *cum);
    if (!pve || !cum) return NULL;
    double total = 0.0;
    for (int c = 0; c < k; ++c) total += S[c]*S[c];

    double running = 0.0;
    for (int c = 0; c < k; ++c) {
        pve[c] = (S[c]*S[c]) / total;
        running += pve[c];
        cum[c] = running;
    }
    *pve_out = pve;
    *cum_out = cum;
}

void free_pca(PCAResult *R) {
    free(R->PCAdirections);
    free(R->scores);
    free(R->loadings);
    free(R->stddev);
    free(R->pve);
    free(R->cum);
}

PCAResult cpca(double *A, int m, int n){

    // Center A on columns
    center(A, m, n);

    SVDResult svd = golub_reinsch_svd(m, n, A, epsilon);
    int k = svd.k;
    R.k = k;

    PCAResult R = {m, n, k, NULL,NULL,NULL,NULL,NULL,NULL};

    // Return the principle directions V
    R.PCAdirections = malloc((size_t)n * n * sizeof *R.PCAdirections);
    if (!R.PCAdirections) goto FAIL;
    memcpy(R.PCAdirections, svd.V, (size_t)n * n * sizeof *svd.V);
    // Compute the standerdised scores with 
    R.scores = scores_from_us(svd.U, svd.S, m, k);
    // Compute the loadings 
    loadings(svd.V, svd.S, n, k, m, &R.loadings);
    // Compute the standard deviation
    std_PC(svd.S, n, k, m, &R.stddev);
    // Compute the proportion of variance explained (eigenvalues)
    pve(svd.V, svd.S, n, k, m, &R.pve, &R.cum);

    /* alloc failure check */
    if (!R.scores || !R.loadings || !R.stddev || !R.pve || !R.cum)
        goto FAIL;

    free_svd(&svd);
    return R;

FAIL:
    free_pca(&R);
    free_svd(&svd);
    fprintf(stderr,"cpca: out of memory\n");
    exit(EXIT_FAILURE);

}

