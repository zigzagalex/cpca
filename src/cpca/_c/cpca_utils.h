#ifndef CPCA_UTILS_H
#define CPCA_UTILS_H

#include "golub-reinsch.h"   
#include "cpca.h"           

// R-style summary of a PCAResult
void pca_pretty_print(const PCAResult *p);

// helper: X_rec = scores * t(PCAdirections)  
void pca_reconstruct(const PCAResult *p, double *X_rec);

#endif