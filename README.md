# cpca
A C implementation of PCA packaged for python.

This is a personal project to learn 4 things: 
1. Matrix operations and the CBLAS library in C
2. Linear dimensionality reduction using PCA through different SVD algorithms.
3. How to package C code for python using cython. 
4. How to visualize matrix algorithms in python. 

What is PCA?

How can PCA be done with SVD?

What SVD algorithm is implemented: Golub Reinsch with Golub Kahan step (reference)

What could be improved? 
* In the Golub Reinsch algorithm two operations could be parallelized. First the Givens rotations could be computed in parallel for more than one element since the rotations act on the i,i+1 rows/columns and don't touch the other ones. Second the individual subblocks could be identified in one loop and simultaniously be transformed using Givens rotations. This again because the rotations acting on one block will not affect the others by definition of the blocks and the rotation. Furthermore memory management could be improved by setting less copies and doing operations on A directly.



### References
[1] Handbook of Linear Algebra - Discrete Mathematics and its Applications (2007) in particular Chapter
[2] https://stats.stackexchange.com/questions/134282/relationship-between-svd-and-pca-how-to-use-svd-to-perform-pca
[3] Cython - A guide for Python Programmers by Kurt W. Smith (2015) O'Reily

