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
 * This file is the file header containing all the CUSIMANN 
 * functions.
 * See supplied whitepaper for more explanations.
 */


#ifndef CUSIMANN_CUH
#define CUSIMANN_CUH

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand_kernel.h>
#include <cutil_inline.h>
#include <math.h>
#include <ctime>
#include <nccl.h>
#include "configuration.h"

#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>


#ifdef MULTIGPU
#include <omp.h>
#endif

#ifndef DOUBLE_PRECISION
	#define NCCLTYPE ncclFloat
#else
        #define NCCLTYPE ncclDouble
#endif

#define CURAND_CALL(x) do { if ((x) != CURAND_STATUS_SUCCESS) { \
							printf("Error at %s:%d\n",__FILE__,__LINE__); \
							return EXIT_FAILURE;} } while(0)

#define NCCLCHECK(cmd) do { \
    ncclResult_t r=cmd;\
    if (r!=ncclSuccess){\
        printf("Failed, NCCL error %s: %d '%s'\n", __FILE__, __LINE__, ncclGetErrorString(r));\
        exit(EXIT_FAILURE);\
    }\
} while(0)



				//J:row, K: column, NK: number of columns
#define pos2Dto1D(J, K, NK) ( (J)*(NK) + (K) )

//n_MAX: maximum dimension of the search space
#define CUSIMANN_n_MAX 1000

__constant__ unsigned int CUSIMANN_n, CUSIMANN_N;
__constant__ real CUSIMANN_LB[CUSIMANN_n_MAX], CUSIMANN_UB[CUSIMANN_n_MAX];
__constant__ real CUSIMANN_BEST_POINT[CUSIMANN_n_MAX];
__constant__ real CUSIMANN_F_BEST_POINT;


void checkCUDAError(const char *msg) {
	cudaError_t err = cudaGetLastError();
	if( cudaSuccess != err) {
		fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
		exit(EXIT_FAILURE);
	}       
}

__global__ void cusimann_generateStartPoints(real *d_startPoints, unsigned int root, real *d_deltas) {
	const int tid = blockDim.x * blockIdx.x + threadIdx.x;
	
	int c;
	for(c=0;c<CUSIMANN_n;c++)
		d_startPoints[pos2Dto1D(tid,c,CUSIMANN_n)] = CUSIMANN_LB[c] + (((int)(tid/powf(root,c))) % root) * d_deltas[c];

}

__global__ void setup_kernel(curandState *state, time_t seed){
	const int tid = blockDim.x * blockIdx.x + threadIdx.x;
	curand_init(seed, tid, 0, &state[tid]);
}


template<unsigned short int isFirstTime, class F>
__global__ void cusimann_kernel(curandState *state, real T, unsigned int pos_min_f_points, real *d_points, real *d_f_points, void *f_data, F f) {
	
	const int global_tid = blockDim.x * blockIdx.x + threadIdx.x;

	real f_actual, f_neighbor;
	int c;

	// This if is evaluated at compile time
	if (isFirstTime)
		f_actual = f(&(d_points[pos2Dto1D(global_tid,0,CUSIMANN_n)]),CUSIMANN_n,f_data);
	else {	
	
		for(c=0;c<CUSIMANN_n;c++)
			d_points[pos2Dto1D(global_tid,c,CUSIMANN_n)] = CUSIMANN_BEST_POINT[c];
		
		
		f_actual = CUSIMANN_F_BEST_POINT;
	}

	curandState localState = state[global_tid];
		
	//sa loop
	real u, actual;
	int i,d;
	for(i=0;i<CUSIMANN_N;i++){
		
		//begin compute neighbour
		u = curand_uniform(&localState);
		d = u * CUSIMANN_n;

		if(d>=CUSIMANN_n)
			d--;
		
		actual = d_points[pos2Dto1D(global_tid,d,CUSIMANN_n)];
		
		u = curand_uniform(&localState);
		d_points[pos2Dto1D(global_tid,d,CUSIMANN_n)] = (u<0.5f) ? 2.0f*(actual-CUSIMANN_LB[d])*u+CUSIMANN_LB[d] : (2.0f*u-1.0f)*(CUSIMANN_UB[d]-actual) + actual;
		
		f_neighbor = f(&(d_points[pos2Dto1D(global_tid,0,CUSIMANN_n)]),CUSIMANN_n,f_data);
		//end compute neighbour
		
		u = curand_uniform(&localState);

		if ( u <= exp( -(f_neighbor-f_actual)/T ) )
			f_actual = f_neighbor;
		else
			d_points[pos2Dto1D(global_tid,d,CUSIMANN_n)] = actual;
	
	}
	state[global_tid] = localState;

	d_f_points[global_tid] = f_actual;
}

#ifndef MULTIGPU
template<class F>
int cusimann_optimize(unsigned int n_threads_per_block, unsigned int n_blocks, real T_0, real T_min, unsigned int N, real rho, unsigned int n, real *lb, real *ub, F f, void *f_data, real *cusimann_minimum, real *f_cusimann_minimum) {

	real T;

	size_t sizeFD = n * sizeof(real);
	cutilSafeCall(cudaMemcpyToSymbol("CUSIMANN_n", &n, sizeof(n)));
	cutilSafeCall(cudaMemcpyToSymbol("CUSIMANN_N", &N, sizeof(N)));
	cutilSafeCall(cudaMemcpyToSymbol("CUSIMANN_LB", lb, sizeFD));
	cutilSafeCall(cudaMemcpyToSymbol("CUSIMANN_UB", ub, sizeFD));
	
	const unsigned int NThreads = n_threads_per_block * n_blocks; //total number of threads

	real *d_points; real *d_f_points;
	cutilSafeCall(cudaMalloc((void**)&d_points, NThreads*n*sizeof(real)));
	cutilSafeCall(cudaMalloc((void**)&d_f_points, NThreads*sizeof(real)));

	unsigned int pos_min_d_f_points=0; 
	
	//begin: generate start points
	unsigned int root = (unsigned int)ceilf( powf(NThreads,1/(real)n) );
	real *h_deltas = (real*)malloc(n*sizeof(real));
	
	unsigned int c;
	for(c=0;c<n;c++)
		h_deltas[c] = (ub[c]-lb[c])/(root-1);
	
	real *d_deltas;
	cutilSafeCall( cudaMalloc( (void**)&d_deltas, n*sizeof(real) ) );
	cutilSafeCall( cudaMemcpy(d_deltas, h_deltas, n*sizeof(real), cudaMemcpyHostToDevice ) );
	
	cusimann_generateStartPoints<<<n_blocks,n_threads_per_block>>>(d_points, root, d_deltas);
	cutilSafeCall( cudaFree(d_deltas) );
	free(h_deltas);
	//end: generate start points
	
	//begin: random number generator initialization
	curandState *devStates;
	cutilSafeCall( cudaMalloc((void**)&devStates, NThreads*sizeof(curandState)) );

	setup_kernel<<<n_blocks,n_threads_per_block>>>(devStates, time(NULL));
	//end: random number generator initialization


	//begin: sa

	thrust::device_ptr<real> dt_f_points(d_f_points);

	// first time
	T = T_0;
	cusimann_kernel<1><<<n_blocks,n_threads_per_block>>>(devStates, T, pos_min_d_f_points, d_points, d_f_points, f_data, f);
	checkCUDAError("Program aborted: ");

	pos_min_d_f_points = thrust::min_element(dt_f_points, dt_f_points + NThreads) - dt_f_points;

	cutilSafeCall( cudaMemcpyToSymbol("CUSIMANN_BEST_POINT",&(d_points[pos2Dto1D(pos_min_d_f_points,0,n)]),n*sizeof(real),0,cudaMemcpyDeviceToDevice) );
	cutilSafeCall( cudaMemcpyToSymbol("CUSIMANN_F_BEST_POINT",&(d_f_points[pos_min_d_f_points]),sizeof(real),0,cudaMemcpyDeviceToDevice) );

	T *= rho;
	// end first time

	// next times
	do {

		cusimann_kernel<0><<<n_blocks,n_threads_per_block>>>(devStates, T, pos_min_d_f_points, d_points, d_f_points, f_data, f);

		pos_min_d_f_points = thrust::min_element(dt_f_points, dt_f_points + NThreads) - dt_f_points;

		cutilSafeCall( cudaMemcpyToSymbol("CUSIMANN_BEST_POINT",&(d_points[pos2Dto1D(pos_min_d_f_points,0,n)]),n*sizeof(real),0,cudaMemcpyDeviceToDevice) );
		cutilSafeCall( cudaMemcpyToSymbol("CUSIMANN_F_BEST_POINT",&(d_f_points[pos_min_d_f_points]),sizeof(real),0,cudaMemcpyDeviceToDevice) );

		T *= rho;
	} while(T>T_min);
	// end next times

	//end: sa

	cutilSafeCall(cudaMemcpy(cusimann_minimum, &(d_points[pos2Dto1D(pos_min_d_f_points,0,n)]), n*sizeof(real), cudaMemcpyDeviceToHost ));

	cutilSafeCall(cudaMemcpy(f_cusimann_minimum, &(d_f_points[pos_min_d_f_points]), sizeof(real), cudaMemcpyDeviceToHost ));

	//begin: CLEAN UP
	cutilSafeCall(cudaFree(d_points));
	cutilSafeCall(cudaFree(d_f_points));
	cutilSafeCall(cudaFree(devStates));
	//cutilSafeCall(cudaDeviceReset());
	//end: CLEAN UP
	return 1;
}
#else // MULTIGPU
static void reduceMultiGPU(real *vector, unsigned int size, unsigned int *position) {
	unsigned int i;

	real minimum = vector[0];
	*position = 0;

	for(i=1;i<size;i++) {
		if (vector[i] < minimum) {
			minimum = vector[i];
			*position = i;
		}	
	}

}

template<class F>
int cusimann_optimize(unsigned int n_threads_per_block, unsigned int n_blocks, real T_0, real T_min, unsigned int N, real rho, unsigned int n, real *lb, real *ub, F f, void *f_data, real *cusimann_minimum, real *f_cusimann_minimum) {


	size_t sizeFD = n * sizeof(real);

	const unsigned int NThreads = n_threads_per_block * n_blocks; //total number of threads

	unsigned int root = (unsigned int)ceilf( powf(NThreads,1/(real)n) );
	real *h_deltas = (real*)malloc(n*sizeof(real));
	
	unsigned int c;
	for(c=0;c<n;c++)
		h_deltas[c] = (ub[c]-lb[c])/(root-1);
	
	int num_gpus = 0;
	cudaGetDeviceCount(&num_gpus);
	if (num_gpus < 1) {
		printf("No CUDA capable devices were detected\n");
		return 1;
	}
	omp_set_num_threads(num_gpus);

	//Begin: memory allocation for reducing the results of all GPUs
	real *h_best_points, *h_f_best_points;
	
	h_best_points = (real*)malloc(num_gpus*n*sizeof(real)); // h_best_points [num_gpus][n]
	h_f_best_points = (real*)malloc(num_gpus*sizeof(real)); // h_f_best_points[n]

	unsigned int pos_minimum_f_best_points;
	//End: memory allocation for reducing the results of all GPUs

	ncclUniqueId id;
    ncclGetUniqueId(&id);

	#pragma omp parallel
	{

		real T;
	
		unsigned int cpu_thread_id = omp_get_thread_num();
		unsigned int num_cpu_threads = omp_get_num_threads();
		unsigned int tid = cpu_thread_id;
		cutilSafeCall(cudaSetDevice(cpu_thread_id));

		int gpu_id = -1;
		cutilSafeCall(cudaGetDevice(&gpu_id));
		printf("CPU thread %d (of %d) uses CUDA device %d\n", cpu_thread_id, num_cpu_threads, gpu_id);
		

		ncclComm_t comm;
		NCCLCHECK(ncclCommInitRank(&comm, num_gpus, id, gpu_id));
		
		if (cpu_thread_id == 0)
		{
			cutilSafeCall(cudaMemcpyToSymbol(CUSIMANN_n, &n, sizeof(n)));
			cutilSafeCall(cudaMemcpyToSymbol(CUSIMANN_N, &N, sizeof(N)));
			cutilSafeCall(cudaMemcpyToSymbol(CUSIMANN_LB, lb, sizeFD));
			cutilSafeCall(cudaMemcpyToSymbol(CUSIMANN_UB, ub, sizeFD));
		}

		real * CUSIMANN_n_ptr;
		real * CUSIMANN_N_ptr;
		real * CUSIMANN_LB_ptr;
		real * CUSIMANN_UB_ptr;

		cudaGetSymbolAddress((void**)&CUSIMANN_n_ptr, CUSIMANN_n);
		cudaGetSymbolAddress((void**)&CUSIMANN_N_ptr, CUSIMANN_N);
		cudaGetSymbolAddress((void**)&CUSIMANN_LB_ptr, CUSIMANN_LB);
		cudaGetSymbolAddress((void**)&CUSIMANN_UB_ptr, CUSIMANN_UB);

		NCCLCHECK(ncclBcast(CUSIMANN_n_ptr, 1, NCCLTYPE, 0, comm, 0));
		NCCLCHECK(ncclBcast(CUSIMANN_N_ptr, 1, NCCLTYPE, 0, comm, 0));		
		NCCLCHECK(ncclBcast(CUSIMANN_LB_ptr, n, NCCLTYPE, 0, comm, 0));
		NCCLCHECK(ncclBcast(CUSIMANN_UB_ptr, n, NCCLTYPE, 0, comm, 0));


		real *d_points; real *d_f_points;
		cutilSafeCall(cudaMalloc((void**)&d_points, NThreads*n*sizeof(real)));
		cutilSafeCall(cudaMalloc((void**)&d_f_points, NThreads*sizeof(real)));

		real *d_deltas;
		cutilSafeCall( cudaMalloc( (void**)&d_deltas, n*sizeof(real) ) );

		
		
		if (cpu_thread_id == 0) 
		{
			cutilSafeCall( cudaMemcpy(d_deltas, h_deltas, n*sizeof(real), cudaMemcpyHostToDevice ) );
		}
	
		NCCLCHECK(ncclBcast(d_deltas, n, NCCLTYPE, 0, comm, 0));
	
		cusimann_generateStartPoints<<<n_blocks,n_threads_per_block>>>(d_points, root, d_deltas);

		cutilSafeCall( cudaFree(d_deltas) );

	
		//begin: random number generator initialization
		curandState *devStates;
		cutilSafeCall( cudaMalloc((void**)&devStates, NThreads*sizeof(curandState)) );

		
		setup_kernel<<<n_blocks,n_threads_per_block>>>(devStates, cpu_thread_id+time(NULL));
	
		//end: random number generator initialization


		//begin: sa

		thrust::device_ptr<real> dt_f_points(d_f_points);

		unsigned int pos_min_d_f_points=0; 

		// first time
		T = T_0;
		
		cusimann_kernel<1><<<n_blocks,n_threads_per_block>>>(devStates, T, pos_min_d_f_points, d_points, d_f_points, f_data, f);
		
		checkCUDAError("Program aborted: ");


		pos_min_d_f_points = thrust::min_element(dt_f_points, dt_f_points + NThreads) - dt_f_points;

		
		//cutilSafeCall(cudaMemcpy(&(h_best_points[pos2Dto1D(cpu_thread_id,0,n)]), &(d_points[pos2Dto1D(pos_min_d_f_points,0,n)]), n*sizeof(real), cudaMemcpyDeviceToHost ));
		cutilSafeCall(cudaMemcpy(&(h_f_best_points[cpu_thread_id]), &(d_f_points[pos_min_d_f_points]), sizeof(real), cudaMemcpyDeviceToHost ));		
		


		#pragma omp barrier
		//begin: reduce multigpu
		if (cpu_thread_id == 0)
			reduceMultiGPU(h_f_best_points, num_gpus, &pos_minimum_f_best_points);
		#pragma omp barrier
		//end: reduce multigpu

		
		//cutilSafeCall( cudaMemcpyToSymbol(CUSIMANN_BEST_POINT,&(h_best_points[pos2Dto1D(pos_minimum_f_best_points,0,n)]),n*sizeof(real)) );
		//cutilSafeCall( cudaMemcpyToSymbol(CUSIMANN_F_BEST_POINT,&(h_f_best_points[pos_minimum_f_best_points]),sizeof(real) ) );
		real * CUSIMANN_BEST_POINT_ptr;
		real * CUSIMANN_F_BEST_POINT_ptr;			
		cudaGetSymbolAddress((void**)&CUSIMANN_BEST_POINT_ptr, CUSIMANN_BEST_POINT);
		cudaGetSymbolAddress((void**)&CUSIMANN_F_BEST_POINT_ptr, CUSIMANN_F_BEST_POINT);
		if (cpu_thread_id == pos_minimum_f_best_points)
		{
			
			cudaMemcpy(CUSIMANN_BEST_POINT_ptr,  &(d_points[pos2Dto1D(pos_min_d_f_points,0,n)]), n*sizeof(real), cudaMemcpyDeviceToDevice);
			cudaMemcpy(CUSIMANN_F_BEST_POINT_ptr, &(d_f_points[pos_min_d_f_points]), sizeof(real), cudaMemcpyDeviceToDevice);
			
		}
		
		NCCLCHECK(ncclBcast(CUSIMANN_BEST_POINT_ptr, n, NCCLTYPE, pos_minimum_f_best_points, comm, 0));
		NCCLCHECK(ncclBcast(CUSIMANN_F_BEST_POINT_ptr, 1, NCCLTYPE, pos_minimum_f_best_points, comm, 0));
		

		#pragma omp barrier

		T *= rho;
		// end first time

		// next times
		do {
			
			cusimann_kernel<0><<<n_blocks,n_threads_per_block>>>(devStates, T, pos_min_d_f_points, d_points, d_f_points, f_data, f);
			
			pos_min_d_f_points = thrust::min_element(dt_f_points, dt_f_points + NThreads) - dt_f_points;

			
			//cutilSafeCall(cudaMemcpy(&(h_best_points[pos2Dto1D(cpu_thread_id,0,n)]), &(d_points[pos2Dto1D(pos_min_d_f_points,0,n)]), n*sizeof(real), cudaMemcpyDeviceToHost ));
			cutilSafeCall(cudaMemcpy(&(h_f_best_points[cpu_thread_id]), &(d_f_points[pos_min_d_f_points]), sizeof(real), cudaMemcpyDeviceToHost ));
			
			#pragma omp barrier
			//begin: reduce multigpu
			if (cpu_thread_id == 0)
				reduceMultiGPU(h_f_best_points, num_gpus, &pos_minimum_f_best_points);
			#pragma omp barrier
			//end: reduce multigpu


			//cutilSafeCall( cudaMemcpyToSymbol(CUSIMANN_BEST_POINT,&(h_best_points[pos2Dto1D(pos_minimum_f_best_points,0,n)]),n*sizeof(real)) );
			//cutilSafeCall( cudaMemcpyToSymbol(CUSIMANN_F_BEST_POINT,&(h_f_best_points[pos_minimum_f_best_points]),sizeof(real) ) );
			real * CUSIMANN_BEST_POINT_ptr;
                	real * CUSIMANN_F_BEST_POINT_ptr;
			cudaGetSymbolAddress((void**)&CUSIMANN_BEST_POINT_ptr, CUSIMANN_BEST_POINT);
                        cudaGetSymbolAddress((void**)&CUSIMANN_F_BEST_POINT_ptr, CUSIMANN_F_BEST_POINT);

			if (cpu_thread_id == pos_minimum_f_best_points)
                	{
                    	cudaMemcpy(CUSIMANN_BEST_POINT_ptr,  &(d_points[pos2Dto1D(pos_min_d_f_points,0,n)]), n*sizeof(real), cudaMemcpyDeviceToDevice);
                    	cudaMemcpy(CUSIMANN_F_BEST_POINT_ptr, &(d_f_points[pos_min_d_f_points]), sizeof(real), cudaMemcpyDeviceToDevice);
                	}
                	NCCLCHECK(ncclBcast(CUSIMANN_BEST_POINT_ptr, n, NCCLTYPE, pos_minimum_f_best_points, comm, 0));
                	NCCLCHECK(ncclBcast(CUSIMANN_F_BEST_POINT_ptr, 1, NCCLTYPE, pos_minimum_f_best_points, comm, 0));
			#pragma omp barrier
			T *= rho;

		} while(T>T_min);
	
		if (cpu_thread_id == pos_minimum_f_best_points)
		{
			cutilSafeCall(cudaMemcpy(&(h_best_points[pos2Dto1D(pos_minimum_f_best_points,0,n)]), &(d_points[pos2Dto1D(pos_min_d_f_points,0,n)]), n*sizeof(real), cudaMemcpyDeviceToHost ));
            cutilSafeCall(cudaMemcpy(&(h_f_best_points[pos_minimum_f_best_points]), &(d_f_points[pos_min_d_f_points]), sizeof(real), cudaMemcpyDeviceToHost ));
		}
		// end next times

		//end: sa

		//begin: CLEAN UP
		cutilSafeCall(cudaFree(d_points));
		cutilSafeCall(cudaFree(d_f_points));
		cutilSafeCall(cudaFree(devStates));

		//cutilSafeCall(cudaDeviceReset());
		//end: CLEAN UP
	}

	memcpy(cusimann_minimum, &(h_best_points[pos2Dto1D(pos_minimum_f_best_points,0,n)]), n*sizeof(real));
	*f_cusimann_minimum = h_f_best_points[pos_minimum_f_best_points];

	free(h_deltas);
	free(h_best_points);
	free(h_f_best_points);
	//std::cout << "Communication Time: " << comm_time << "s." << std::endl; 
	return cudaThreadExit();
}


#endif // MULTIGPU

#endif // CUSIMANN_CUH
