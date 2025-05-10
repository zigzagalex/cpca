#include <stdio.h>
#include <cblas.h>
#include <math.h>

void print_matrix(const double *A, int rows, int cols) {
    for (int i=0; i<rows; i++){
        for (int j=0; j<cols; j++){
            printf("%10.4f ", A[i*cols+j]);
        }
        printf("\n");
    }
    printf("\n");
}


int main() {
    /* 
    Questions:
    1. What is a float complex?
    2. Should I use flattened arrays? Yes, because 2d arrays must have defined size at initialization of the variable i.e. you cant pass double [][] into a function, only double [m][n] with m,n fixed. 
    3. What is the difference between float and double?
    

    */
    int n = 4;

    double x[4] = {1,2,3,4};
    double y[4] = {4,5,6,7};

    // double z[4] = x + y; trying to just add two arrays will not work because arrays decay to pointers
    // so adding to addresses together doesn't make sense

    double a = 2;
    int incx = 1;
    int incy = 1;

    print_matrix(y, 1, 4);

    // general cblas_?func(**args) where ? is data type (s,d,c,z)

    // O(n) operations

    // adding vectors y = y + a*x
    cblas_daxpy(n, a, x, incx, y, incy);

    // scale vector x = a*x
    cblas_dscal(n, a, x, incx);

    // copy vector z <- x
    double z[4];
    int incz = 1;
    cblas_dcopy(n, x, incx, z, incz);


    // adding vectors of different sizes: just adds the values of indeces extending the vector if neccessary with zeros
    double v[5] = {1,2,3,4,5};
    cblas_daxpy(n,a,v,incx,x,incy);
    print_matrix(x, 1, 4);
    print_matrix(v, 1, 5);

    // dot product x^t * y
    double l = cblas_ddot(n,x,incx,y,incy);
    printf("%f l\n", l);

    // dot product of wrong dimensions
    double k = cblas_ddot(n,x,incx, v, incy);
    printf("%f k\n", k);

    // 2-norm of a vector
    double nn = cblas_dnrm2(n,x,incx);
    printf("2-norm of x is: %.10f\n", nn);

    // inf-norm 
    int idx = cblas_idamax(n, x, incx); // index of max |x[i]|
    double in = fabs(x[idx]);           // the actual value
    printf("inf-norm of x is: %.10f\n", in);



    return 0;
};