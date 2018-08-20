//////////////////////////////////////////////////////////////////////////////////
//                                                                              //
//  trueke                                                                      //
//  A multi-GPU implementation of the exchange Monte Carlo method.              //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
//                                                                              //
//  Copyright © 2015 Cristobal A. Navarro, Wei Huang.                           //
//                                                                              //
//  This file is part of trueke.                                                //
//  trueke is free software: you can redistribute it and/or modify              //
//  it under the terms of the GNU General Public License as published by        //
//  the Free Software Foundation, either version 3 of the License, or           //
//  (at your option) any later version.                                         //
//                                                                              //
//  trueke is distributed in the hope that it will be useful,                   //
//  but WITHOUT ANY WARRANTY; without even the implied warranty of              //
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the               //
//  GNU General Public License for more details.                                //
//                                                                              //
//  You should have received a copy of the GNU General Public License           //
//  along with trueke.  If not, see <http://www.gnu.org/licenses/>.             //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////

/* comment MEASURE if you just want to check simulation performance. */
#define MEASURE

/* warning: changing the block dimensions can lead to undefined behavior. */
#define WARPSIZE 32
#define BX	32
#define BY	8
#define	BZ	4
#define BLOCKSIZE1D 512
#define Q_DIST 2000

#include <stdio.h>
#include <cuda.h>
#include <time.h>
#include <sys/time.h>
#include <syslog.h>
#include <string>
#include <curand_kernel.h>
/* include MTGP host helper functions */
#include <curand_mtgp32_host.h>
/* include MTGP pre-computed parameter sets */
#include <curand_mtgp32dc_p_11213.h>
/* Utilities and system includes */
#include <helper_cuda.h>
#include <helper_functions.h>
#include <omp.h>
#include <nvml.h>

// Error checking
#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while(0)

/* local includes */
#include "cpcg.cuh"
#include "structs.h"
#include "heap.h"
#include "kmetropolis.cuh"
#include "kprng.cuh"
#include "reduction.cuh"
#include "cputools.h"
#include "tools.cuh"
#include "setup.h"
#include "pt.h"
#include "adapt.h"

int main(int argc, char **argv){
	/* openmp thread values */
	int tid, nt, r, a, b; 
    double atime, stime;
	printf("\n**************** trueke : multi-GPU Parallel Tempering for 3D Random Field Ising Model****************\n\n");
	/* setup handles the variables */
	setup_t s;
    /* init */
	adapt_init(&s, argc, argv);
	/* find good temperature distribution */
	atime = adapt(&s);
	printf("ok: total time %.2f secs\n", atime);
	// initialization takes care of memory allocation
	init(&s, argc, argv);	
	/* measure time */
	sdkResetTimer(&(s.gtimer));
	sdkStartTimer(&(s.gtimer));
	#pragma omp parallel private(tid, nt, r, a, b) shared(s)
	{
		/* main simulation */ 
		for(int i = 0; i < s.realizations; i++){
			/* multi-GPU PT simulation */
			/* set the thread */
			threadset(&s, &tid, &nt, &r);
			a = tid * r;
			b = a + r;
			if(tid == 0){
				printf("[realization %i of %i]\n", i+1, s.realizations); fflush(stdout);
			}
			/* reset some data at each realization*/
			reset(&s, tid, a, b);
			/* distribution for H */
			hdist(&s, tid, a, b);
			kernel_reset<float><<< (b-a + BLOCKSIZE1D - 1)/BLOCKSIZE1D, BLOCKSIZE1D, 0, s.rstream[a] >>> (s.dE[tid], b-a, 0.0f);
			cudaDeviceSynchronize();	cudaCheckErrors("kernel_reset dE");
			/* up values */
			//#pragma omp barrier
			//#pragma omp barrier
			//if(tid == 0){
				//printarray<float>(s.exE, s.R, "exE");
			//}
			//#pragma omp barrier
			/* equilibration */
			equilibration(&s, tid, a, b);
			//#pragma omp barrier
			//ptenergies(&s, tid, a, b);
			//#pragma omp barrier
			//if(tid == 0){
			//	printarray<float>(s.exE, s.R, "exE");
			//}
			/* parallel tempering */
			pt(&s, tid, a, b);
			#ifdef MEASURE
				/* accumulate realization statistics */
				accum_realization_statistics( &s, tid, a, b, s.realizations );
			#endif
			#pragma omp barrier
		}
	}
#ifdef MEASURE
	/* write physical results */
	physical_results(&s);
#endif
	/* total time */
	sdkStopTimer(&(s.gtimer));
    stime = sdkGetTimerValue(&(s.gtimer))/1000.0f;
	printf("ok: total time = adaptive + simulation = %.2fh + %.2fh = %.2fh\n", atime/3600.0, stime/3600.0, (atime + stime)/3600.0);

	/* free memory */
	freemem(&s);
    freegpus(&s);
}	
