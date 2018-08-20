/* Copyright (c) 2011-2012 University of A Coruña
 *
 * CUSIMANN: An optimized simulated annealing software for GPUs
 *
 * Authors: A.M. Ferreiro, J.A. García, J.G. López-Salas, C. Vázquez 
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 * LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 * OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 * WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

/*
 * This sample minimizes the Shekel Functions (
 * http://www-optima.amp.i.kyoto-u.ac.jp/member/student/hedar/Hedar_files/TestGO_files/Page2354.htm ).
 * See supplied whitepaper for more explanations.
 */


#include "cusimann.cuh"
#include "nelderMead.h"

typedef struct {
	real *A, *c;
} SHEKEL_data;

template <class T>
class Shekel {
public:
	__host__ __device__ T operator() (const T *x, unsigned int n, void *f_data) const
	{
		SHEKEL_data shekel_data;
		shekel_data = *((SHEKEL_data*)f_data);

		real *A = shekel_data.A;
		real *c = shekel_data.c;
	
		T f_x = 0.0f, aux;

		int i, j;
		for(i=0;i<10;i++) {

			aux = 0.0f;
			for(j=0;j<4;j++)
				aux += pow(x[j]-A[pos2Dto1D(i,j,4)],2);
	
			f_x +=  1.0f/(aux+c[i]);
		}
	
		return - f_x;
	}
};

double f_nelderMead(unsigned int n, const double *x, double *grad, void *f_data){
	return Shekel<double>()(x,n,f_data);
}

int main() {
	real T_0 = 1000, T_min = 0.01;
	const unsigned int n = 4, N = 100;
	const real rho = 0.99;
	size_t sizeFD = n * sizeof(real);
	real *lb, *ub, *cusimann_minimum = (real*)malloc(sizeFD), f_cusimann_minimum;
	lb = (real*)malloc(sizeFD);
	unsigned int i;
	for(i=0;i<n;i++)
		lb[i] = -5; 
	ub = (real*)malloc(sizeFD);
	for(i=0;i<n;i++)
		ub[i] = 15;


	real A[10*4] = {	4, 4, 4, 4,
						1, 1, 1, 1,
						8, 8, 8, 8,
						6, 6, 6, 6,
						3, 7, 3, 7,
						2, 9, 2, 9,
						5, 5, 3, 3,
						8, 1, 8, 1,
						6, 2, 6, 2,
						7,3.6,7,3.6
					 };

	real c[10] = { 0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5 };
	
	// struct in host memory
	SHEKEL_data shekel_data;
	shekel_data.A = A;
	shekel_data.c = c;

	// Assemble the device structure in host memory first
	SHEKEL_data *d_Array_shekel_data, d_shekel_data[1];

	real *d_A, *d_c;
	cutilSafeCall( cudaMalloc((void**)&d_A, 4*10 * sizeof(real)) );
	cutilSafeCall( cudaMemcpy(d_A, shekel_data.A, 4*10 * sizeof(real), cudaMemcpyHostToDevice ) );

	cutilSafeCall( cudaMalloc((void**)&d_c, 10 * sizeof(real)) );
	cutilSafeCall( cudaMemcpy(d_c, shekel_data.c, 10 * sizeof(real), cudaMemcpyHostToDevice ) );

	d_shekel_data[0].A = d_A;
	d_shekel_data[0].c = d_c;

	// Then copy that host memory version to device memory
	cutilSafeCall( cudaMalloc ( (void**) &d_Array_shekel_data, 1*sizeof(SHEKEL_data) ) );
	cutilSafeCall( cudaMemcpy(d_Array_shekel_data, d_shekel_data, 1*sizeof(SHEKEL_data), cudaMemcpyHostToDevice ) );
	


	unsigned int n_threads_per_block = 256;
	unsigned int n_blocks = 64;

	cusimann_optimize(n_threads_per_block, n_blocks, T_0, T_min, N, rho, n, lb, ub, Shekel<real>(), d_Array_shekel_data, cusimann_minimum, &f_cusimann_minimum);

	printf("cusimann_minimum = [");
	for(i=0;i<n;i++)
		printf(" %f", cusimann_minimum[i]);
	printf(" ]\n");
	printf("f(simann_minimum) = %f\n", f_cusimann_minimum);

	double f_nelderMead_minimum;
	double *nelderMead_minimum = (double*)malloc(n*sizeof(double));
	nelderMead_optimize(n, lb, ub, cusimann_minimum, f_nelderMead, &shekel_data, nelderMead_minimum, &f_nelderMead_minimum);

	printf("nelderMead_minimum = [");
	for(i=0;i<n;i++)
		printf(" %f", nelderMead_minimum[i]);
	printf(" ]\n");
	printf("f(nelderMead_minimum) = %lf\n", f_nelderMead_minimum);

	free(lb);
	free(ub);
	free(cusimann_minimum);
	free(nelderMead_minimum);
	
	return EXIT_SUCCESS;
}
