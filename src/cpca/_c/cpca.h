#ifndef CPCA_H
#define CPCA_H

typedef struct {
    int m, n, k;          // rows, cols, # components (k = min(m,n))
    double *PCAdirections;// V  (nxn) matrix axes that slice through the data cloud so you see the biggest spread
    double *scores;       // T  (m × k)
    double *loadings;     // L  (n × k)
    double *stddev;       // s_k (length k)
    double *pve;          // proportion of variance explained (length k)
    double *cum;          // cumulative PVE (length k)
} PCAResult;

PCAResult cpca(double *A, int m, int n);

void free_pca(PCAResult *R);

#endif