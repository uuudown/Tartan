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
 * This sample minimizes the Modified Langerman function (
 * http://www.it.lut.fi/ip/evo/functions/node15.html ).
 * See supplied whitepaper for more explanations.
 */


#include "cusimann.cuh"
#include "nelderMead.h"

#define M_PI_CUDA 3.14159265358979323846f  /* pi */

typedef struct {
	real *A, *c;
} LANGERMAN_data;

template <class T>
class Langerman {
public:
	__host__ __device__ T operator() (const T *x, unsigned int n, void *f_data) const
	{
		LANGERMAN_data langerman_data;
		langerman_data = *((LANGERMAN_data*)f_data);

		real *A = langerman_data.A;
		real *c = langerman_data.c;
	
		T f_x = 0.0f;
		T aux;
	
		int i, j;
		for(i=0;i<5;i++) {

			aux = 0.0f;
			for(j=0;j<n;j++)
				aux += pow(x[j]-A[pos2Dto1D(i,j,10)],2) ;
		
			f_x +=  c[i] * exp(-1.0f/M_PI_CUDA * aux) * cos(M_PI_CUDA * aux);
		}
	
		return - f_x;
	}
};

double f_nelderMead(unsigned int n, const double *x, double *grad, void *f_data){
	return Langerman<double>()(x,n,f_data);
}

int main() {
	real T_0 = 1000, T_min = 0.1; 
	const unsigned int n = 5, N = 400;
	const real rho = 0.99;
	size_t sizeFD = n * sizeof(real);
	real *lb, *ub, *cusimann_minimum = (real*)malloc(sizeFD), f_cusimann_minimum;
	lb = (real*)malloc(sizeFD);
	unsigned int i;
	for(i=0;i<n;i++)
		lb[i] = -10; 
	ub = (real*)malloc(sizeFD);
	for(i=0;i<n;i++)
		ub[i] = 10;


	real A[5*10] = { 9.681, 0.667, 4.783, 9.095, 3.517, 9.325, 6.544, 0.211, 5.122, 2.020, 
					  9.400, 2.041, 3.788, 7.931, 2.882, 2.672, 3.568, 1.284, 7.033, 7.374,
					  8.025, 9.152, 5.114, 7.621, 4.564, 4.711, 2.996, 6.126, 0.734, 4.982,
					  2.196, 0.415, 5.649, 6.979, 9.510, 9.166, 6.304, 6.054, 9.377, 1.426,
					  8.074, 8.777, 3.467, 1.863, 6.708, 6.349, 4.534, 0.276, 7.633, 1.567 };
	real c[5] = { 0.806, 0.517, 0.100, 0.908, 0.965 }; 

	// struct in host memory
 	LANGERMAN_data langerman_data;
 	langerman_data.A = A;
 	langerman_data.c = c;

	// Assemble the device structure in host memory first
	LANGERMAN_data *d_Array_langerman_data, d_langerman_data[1];

	real *d_A, *d_c;
	cutilSafeCall( cudaMalloc((void**)&d_A, 5*10 * sizeof(real)) );
	cutilSafeCall( cudaMemcpy(d_A, langerman_data.A, 5*10 * sizeof(real), cudaMemcpyHostToDevice ) );

	cutilSafeCall( cudaMalloc((void**)&d_c, 5 * sizeof(real)) );
	cutilSafeCall( cudaMemcpy(d_c, langerman_data.c, 5 * sizeof(real), cudaMemcpyHostToDevice ) );

	d_langerman_data[0].A = d_A;
	d_langerman_data[0].c = d_c;

	// Then copy that host memory version to device memory
	cutilSafeCall( cudaMalloc ( (void**) &d_Array_langerman_data, 1*sizeof(LANGERMAN_data) ) );
	cutilSafeCall( cudaMemcpy(d_Array_langerman_data, d_langerman_data, 1*sizeof(LANGERMAN_data), cudaMemcpyHostToDevice ) );
	



	unsigned int n_threads_per_block = 512;
	unsigned int n_blocks = 64;

	cusimann_optimize(n_threads_per_block, n_blocks, T_0, T_min, N, rho, n, lb, ub, Langerman<real>(), d_Array_langerman_data, cusimann_minimum, &f_cusimann_minimum);

	printf("cusimann_minimum = [");
	for(i=0;i<n;i++)
		printf(" %f", cusimann_minimum[i]);
	printf(" ]\n");
	printf("f(simann_minimum) = %lf\n", f_cusimann_minimum);

	double f_nelderMead_minimum;
	double *nelderMead_minimum = (double*)malloc(n*sizeof(double));
	nelderMead_optimize(n, lb, ub, cusimann_minimum, f_nelderMead, &langerman_data, nelderMead_minimum, &f_nelderMead_minimum);

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
