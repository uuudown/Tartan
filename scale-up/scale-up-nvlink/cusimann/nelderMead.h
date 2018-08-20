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
 * This file is the file header containing the NLOPT
 * Nelder Mead Algorithm.
 * See supplied whitepaper for more explanations.
 */


#ifndef NELDERMEAD_H
#define NELDERMEAD_H

#include <stdio.h>
#include <stdlib.h>
#include "nlopt/include/nlopt.h"

void nelderMead_optimize(unsigned int n, real *lb, real *ub, real *startPoint, double (*f)(unsigned int n, const double *x, double *grad, void *f_data), void *f_data, double *minimum, double *minf) {

	//nlopt requires calculations in double, so we convert to double precision lb, ub and startPoint
	double *lb_, *ub_;
	size_t size = n * sizeof(double);
	lb_ = (double*)malloc(size); ub_ = (double*)malloc(size);
	unsigned int c;
	for(c=0;c<n;c++) {
		lb_[c] = lb[c];
		ub_[c] = ub[c];
		minimum[c] = startPoint[c];
	}

	nlopt_opt opt;

	//opt = nlopt_create(NLOPT_LN_NELDERMEAD, n);
	opt = nlopt_create(NLOPT_LN_SBPLX, n);
	//opt = nlopt_create(NLOPT_LN_COBYLA, n); 
	//opt = nlopt_create(NLOPT_LN_PRAXIS, n); 
	nlopt_set_lower_bounds(opt,lb_);
	nlopt_set_upper_bounds(opt,ub_);

	nlopt_set_min_objective(opt, f, f_data);

	//nlopt_set_ftol_abs(opt, 1e-15);
	//nlopt_set_xtol_rel(opt, 1e-15);
	
	nlopt_set_xtol_abs1(opt, 1e-15);

	int resul;
	
	resul = nlopt_optimize(opt,minimum,minf);
	if (resul<0){
		printf("Error: nlopt failed, %d\n", resul); exit(-1);
	}

	free(lb_); free(ub_);
	nlopt_destroy(opt);	

}

#endif // NELDERMEAD_H
