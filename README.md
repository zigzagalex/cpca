# cpca
A C implementation of PCA packaged for python.

This is a personal project to learn 4 things: 
1. Matrix operations and the CBLAS library in C
2. Linear dimensionality reduction using PCA through different SVD algorithms.
3. How to package C code for python using cython. 
4. How to visualize matrix algorithms in python. 

### What is PCA?
Principle component analysis is linear non-parametric (assumes no distribution) transformation of data. It helps rotate the axes of the dataset in such a way that the axes express the data "better". In the real world we often don't know the dynamics and correlations of the systems we are measuring. So we choose variables, in a heuristic way. The goal of PCA is to minimize the redundancy of variables measured by covariance and maximize the signal measured by variance. This is what we mean by "better". We assume with this definition that the directions with the largest variences in our measurement vector space contain the dynamics of interest.

In other words we want to rotate our data matrix A to a set of basis vectors such that our goal is accomplished. For this we use the singular value decomposition (SVD) of a matrix with real elements. 

The output of PCA is: 
* Vectors specifying the principle components 
* The variance explained by the principle components in proportion to total variance of the dataset
* Loadings of principle components i.e. how much every variable contributes to a principle component.

### How can PCA be done with SVD?
Any real m * n matrix $`A`$ can be decomposed into $`A=U*\Sigma*V^T`$. Where $`U`$ (n*n) and $`V`$ (m*m) are orthogonal matrices and $`\Sigma`$ (n*m)is a diagonal matrix containing the singular values of $`A`$ (i.e. the eigenvalues of $`cov(A)`$). In other words any matrix can be decomposed into a rotation, a stretch and another rotation. 

For PCA $`A`$ must be centered with respect to the columns. 

The outputs of PCA can be computed from the SVD $`A`$: 
* Vectors specifying the principle components: $`V`$
* The variance explained by the principle components in proportion to total variance of the dataset: $`p_k = \sigma_k^2 / \Sigma_j \sigma_j^2`$
* Loadings: $`L = V · \Sigma / \sqrt(N−1)`$

SVD algorithm implemented: Golub Reinsch with Golub Kahan step [1] but there are many more to choose from. 

### What could be improved? 
* In the Golub Reinsch algorithm two operations could be parallelized. First the Givens rotations could be computed in parallel for more than one element since the rotations act on the i,i+1 rows/columns and don't touch the other ones. Second the individual subblocks could be identified in one loop and simultaniously be transformed using Givens rotations. This again because the rotations acting on one block will not affect the others by definition of the blocks and the rotation. Furthermore memory management could be improved by setting less copies and doing operations on A directly.
* Memory management inside the C functions.


### How is the C function packaged for Python?


### References
[1] Bernstein, D. S. (2007). Handbook of Linear Algebra. In Discrete Mathematics and its Applications. CRC Press. See especially Chapter 45 by Alan Kaylor Cline and Inderjit S. Dhillon.  
[2] StackExchange. (n.d.). Relationship between SVD and PCA: How to use SVD to perform PCA. Cross Validated. Retrieved from https://stats.stackexchange.com/questions/134282  
[3] Smith, K. W. (2015). Cython: A Guide for Python Programmers. O’Reilly Media.  
[4] Shlens, J. (2014). A Tutorial on Principal Component Analysis (Version 3). Retrieved from https://arxiv.org/abs/1404.1100  
