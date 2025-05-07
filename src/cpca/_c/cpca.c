#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <float.h>
#include <math.h>
#include <cblas.h>
#include "golub-reinsch.h"

#define epsilon 1e-10

void cpca(double *A){
    int m = 0;
    int n = 0;

    // Center A

    golub_reinsch_svd(m, n, A, epsilon);

    // Return Results
    // Return the principle directions V
    // Compute the standerdised scores with U*sqrt(n-1)
    // Compute the loadings V*S/sqrt(n-1)
    // Compute the standard deviation
    // Compute the proportion of variance explained (eigenvalues)
    // Compute the commulatative proportion

}

