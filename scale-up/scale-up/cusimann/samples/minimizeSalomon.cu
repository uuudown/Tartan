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
 * This sample minimizes the Salomon's function (
 * http://www.it.lut.fi/ip/evo/functions/node12.html ).
 * See supplied whitepaper for more explanations.
 */


#include "cusimann.cuh"
#include "nelderMead.h"

#define M_PI_CUDA 3.14159265358979323846f  /* pi */

template <class T>
class Salomon {
public:
	__host__ __device__ T operator() (const T *x, unsigned int n, void *f_data) const
	{
		T f_x = 0.0f;
		T root = 0.0f;
	
		int i;
		for(i=0;i<n;i++)
			root += pow(x[i],2);
		root = sqrt(root);

		f_x = -cos(2.0f*M_PI_CUDA*root) + 0.1f*root + 1.0f;
	
		return f_x;
	}
};

double f_nelderMead(unsigned int n, const double *x, double *grad, void *f_data){
	return Salomon<double>()(x,n,f_data);
}

int main() {
	real T_0 = 100000, T_min = 0.001;
	const unsigned int n = 5, N = 400;
	const real rho = 0.999;
	size_t sizeFD = n * sizeof(real);
	real *lb, *ub, *cusimann_minimum = (real*)malloc(sizeFD), f_cusimann_minimum;
	lb = (real*)malloc(sizeFD);
	unsigned int i;
	for(i=0;i<n;i++)
		lb[i] = -4; 
	ub = (real*)malloc(sizeFD);
	for(i=0;i<n;i++)
		ub[i] = 4;


	unsigned int n_threads_per_block = 512;
	unsigned int n_blocks = 64;

	cusimann_optimize(n_threads_per_block, n_blocks, T_0, T_min, N, rho, n, lb, ub, Salomon<real>(), NULL, cusimann_minimum, &f_cusimann_minimum);

	printf("cusimann_minimum = [");
	for(i=0;i<n;i++)
		printf(" %f", cusimann_minimum[i]);
	printf(" ]\n");
	printf("f(cusimann_minimum) = %lf\n", f_cusimann_minimum);

	double f_nelderMead_minimum;
	double *nelderMead_minimum = (double*)malloc(n*sizeof(double));
	nelderMead_optimize(n, lb, ub, cusimann_minimum, f_nelderMead, NULL, nelderMead_minimum, &f_nelderMead_minimum);

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
