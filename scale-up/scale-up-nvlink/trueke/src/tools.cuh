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
#ifndef _TOOLS_H_
#define _TOOLS_H_


#include <nccl.h>

#define NCCLCHECK(cmd) do { \
    ncclResult_t r=cmd;\
    if (r!=ncclSuccess){\
        printf("Failed, NCCL error %s: %d '%s'\n", __FILE__, __LINE__, ncclGetErrorString(r));\
        exit(EXIT_FAILURE);\
    }\
} while(0) \



using namespace std;

/* print array function */
template <typename T>
void printarray(T *a, int n, const char *name);
/* print fragmented array */
template <typename T>
void printarrayfrag(T **a, int n, int *m, const char *name);
/* print indexed fragmented array */
template <typename T>
void printindexarrayfrag(T **a, findex** ind, int n, int *m, const char *name);

/* forward function declarations */
void threadset(setup_t *s, int *tid, int *nt, int *r);

/* reset functions */
void reset_realization_statistics( setup_t *s, int tid, int a, int b);
void reset_mcmc_statistics( setup_t *s, int tid, int a, int b);
void reset_block_statistics( setup_t *s, int tid, int a, int b);
void reset(setup_t *s, int tid, int a, int b);
void reset_gpudata(setup_t *s, int tid, int a, int b);

/* statistical averaging functions */
void accum_block_statistics( obset_t *obstable, int tid, int a, int b);
void accum_realization_statistics( setup_t *s, int a, int b, int realizations );
void accum_mcmc_statistics( setup_t *s, int tid, int a, int b); 

/* exchange temperatures function */
void extemp(setup_t *s, int r1, int r2);

/* check nvidia nvml result */
int nvml_check(nvmlReturn_t r, const char* mesg);

/* functions for moving fragmented structs */
findex_t fgetleft(setup_t *s, findex_t frag);
findex_t fgetright(setup_t *s, findex_t frag);
void fgoleft(setup_t *s, findex_t *frag);
void fgoright(setup_t *s, findex_t *frag);
void fshiftleft(setup_t *s, findex_t *frag);
void fshiftright(setup_t *s, findex_t *frag);

unsigned int devseed(){
    int data;
    FILE *fp;
    fp = fopen("/dev/urandom", "r");
    int bytes = fread(&data, sizeof(int), 1, fp);
    fclose(fp);
    //printf("seed = %u\n", data);
    //getchar();
    return data;
}

/* compare the utilization of two gpus */
int compgpu(const void *a, const void *b){
	return ( ((gpu*)a)->u - ((gpu*)b)->u );
}

/* compare two floats */
int floatcomp(const void* elem1, const void* elem2){
    if(*(const float*)elem1 < *(const float*)elem2)
        return -1;
    return *(const float*)elem1 > *(const float*)elem2;
}

/* check nvml result */
int nvml_check(nvmlReturn_t r, const char* mesg){
	if(r != NVML_SUCCESS){
		if(r == NVML_ERROR_UNINITIALIZED)
			fprintf(stderr, "nvml error: %s: NVML_ERROR_UNINITIALIZED\n", mesg);
		else if(r == NVML_ERROR_INVALID_ARGUMENT)
			fprintf(stderr, "nvml error: %s: NVML_ERROR_INVALID_ARGUMENT\n", mesg);
		else if(r == NVML_ERROR_NOT_SUPPORTED)
			fprintf(stderr, "nvml error: %s: NVML_ERROR_NOT_SUPPORTED\n", mesg);
		else if(r == NVML_ERROR_GPU_IS_LOST)
			fprintf(stderr, "nvml error: %s: NVML_ERROR_GPU_IS_LOST\n", mesg);
		else if(r == NVML_ERROR_UNKNOWN)
			fprintf(stderr, "nvml error: %s: NVML_ERROR_UNKNOWN\n", mesg);
		else
			fprintf(stderr, "nvml error: %s: code not listed\n");
		return 0;
	}
	return 1;
}

/* array reset */
template < typename T >
void reset_array(T *a, int n, T val){
	//printf("reseting array \n"); fflush(stdout);
	for(int i=0; i<n; ++i){
		a[i] = val;
	}
}
	
/* per realization reset */
void reset(setup_t *s, int tid, int a, int b){
    #pragma omp barrier
#ifdef MEASURE
	/* reset block statistics */
	reset_block_statistics( s, tid, a, b);
#endif
    #pragma omp barrier
	/* reset gpu data */
	reset_gpudata(s, tid, a, b);
    #pragma omp barrier

	/* reset ex counters */
	reset_array<float>((float*)(s->ex + tid*(b-a)), b-a, 0.0f);
    #pragma omp barrier

	/* reset average ex counters */
	reset_array<float>((float*)(s->avex + tid*(b-a)), b-a, 0.0f);
    #pragma omp barrier

	/* reset index arrays */
	for(int i = a; i < b; ++i){
		s->rts[i] = s->trs[i] = i;
		s->exE[i] = 0.0;
	}
    #pragma omp barrier
}

/* set the thread indices for location */
void threadset(setup_t *s, int *tid, int *nt, int *r){
	/* get thread id */
	*tid = omp_get_thread_num();
	/* number of threads */
	*nt = omp_get_num_threads();
	/* 'r' replicas for each GPU */
	*r = (s->R)/(*nt);
	/* set the device */
	//printf("tid = %i   ngpus = %i\n   gpus[%i].i = %i\n", *tid, s->ngpus, s->gpus[*tid].i);
	checkCudaErrors(cudaSetDevice( s->gpus[*tid].i ));
}

/* set the thread indices for location */
void adapt_threadset(setup_t *s, int *tid, int *nt, int *r){
	/* get thread id */
	*tid = omp_get_thread_num();
	/* number of threads */
	*nt = omp_get_num_threads();
	/* 'r' replicas for each GPU */
	//*r = (s->R)/(*nt);
    /* get the number of replicas for the actual GPU */
    *r = s->gpur[*tid];
	/* set the device */
	//printf("tid = %i   ngpus = %i\n   gpus[%i].i = %i\n", *tid, s->ngpus, s->gpus[*tid].i);
	checkCudaErrors(cudaSetDevice( s->gpus[*tid].i ));
}

/* reset gpu data structures */
void reset_gpudata(setup_t *s, int tid, int a, int b){
    // getting new need
    #pragma omp barrier
    if(tid == 0){
        // choose a new seed from the sequential PRNG
        s->seed = gpu_pcg32_random_r(&s->hpcgs, &s->hpcgi);
        printf("[%lu]\n", s->seed);
    }
    #pragma omp barrier
	//printf("tid=%i     a=%i    b=%i\n", tid, a, b); fflush(stdout);
	for(int k = a; k < b; ++k){
		/* up spins */
		kernel_reset<int><<< s->lgrid, s->lblock, 0, s->rstream[k] >>>(s->dlat[k], s->N, 1);
		cudaCheckErrors("kernel: reset spins up");

		/* doing a per-realizaton reset only works if seed is different each time */
        // LATEST
        kernel_gpupcg_setup<<<s->prng_grid, s->prng_block, 0, s->rstream[k] >>>(s->pcga[k], s->pcgb[k], s->N/4, s->seed + s->N/4 * k, k);
		cudaCheckErrors("kernel: prng reset");
	}
    #pragma omp barrier
	cudaDeviceSynchronize();
	cudaCheckErrors("kernel realization resets");
}

int adapt_globalk(setup_t *s, int tid, int k){
	int acc = 0;
	for(int i=0; i<tid; ++i){
		acc += s->gpur[i];
	}
	return k + acc;
}


/* reset gpu data structures */
void adapt_reset_gpudata(setup_t *s, int tid){
    //printf("adapt_reset\n");
    #pragma omp barrier
    if(tid == 0){
        // choose a new seed from the sequential PRNG
        s->seed = gpu_pcg32_random_r(&s->hpcgs, &s->hpcgi);
        printf("[%lu]\n", s->seed);
    }
    #pragma omp barrier
	for(int k = 0; k < s->gpur[tid]; ++k){
		/* up spins */
		kernel_reset<int><<< s->lgrid, s->lblock, 0, s->arstream[tid][k] >>>(s->mdlat[tid][k], s->N, 1);
        //printf("replica %i   %i:\n", tid, k);
        //cudaDeviceSynchronize();
        //printsomespins(s, 10);
        //printsomeh(s, 20);
        //printM(s, tid, k);
		cudaCheckErrors("kernel: reset spins up");
		/* random spins */
		//kernel_reset_random<<< s->prng_grid, s->prng_block, 0, s->rstream[k] >>>(s->dlat[k], s->N, s->dstates[k]);
		//cudaCheckErrors("kernel: reset spins random");
		/* doing a per-realizaton reset only works if seed is different each time */
		//kernel_gpupcg_setup<<<s->prng_grid, s->prng_block, 0, s->arstream[tid][k] >>>(s->apcga[tid][k], s->apcgb[tid][k], s->N/4, s->seed , (unsigned long long)((s->R/s->ngpus)*tid + k));
		//printf("tid=%i   N=%i   N/4 = %i  R = %i  seed = %lu   k = %lu \n", tid, s->N, s->N/4, s->R, s->seed + (unsigned long long)(s->N/4 * (s->rpool[tid]*tid + k)), (s->rpool[tid]*tid + k));
		//printf("tid=%i   N=%i   N/4 = %i  R = %i  seed = %lu   k = %lu \n", tid, s->N, s->N/4, s->R, s->seed + (unsigned long long)(s->N/4 * adapt_globalk(s, tid, k)), adapt_globalk(s, tid, k));
        // LATEST
		kernel_gpupcg_setup<<<s->prng_grid, s->prng_block, 0, s->arstream[tid][k] >>>(s->apcga[tid][k], s->apcgb[tid][k], s->N/4, s->seed + (unsigned long long)(s->N/4 * adapt_globalk(s, tid, k)), adapt_globalk(s, tid, k));
		cudaCheckErrors("kernel: prng reset");
	}
    #pragma omp barrier
	cudaDeviceSynchronize();
	cudaCheckErrors("kernel realization resets");
}

/* system call */
int run_sys_call(char *buffer){
    int res;
    res = system(buffer);
    if ( WEXITSTATUS(res) != 0 ) {
                syslog(LOG_CRIT," System call failed.\n");
                syslog(LOG_CRIT," %s\n",buffer);
    }
    return res;
}

/* make the output folders */
#ifdef MEASURE
void make_output_folders( const char *obs, const char *plot ){
	char command[256];
	sprintf(command, "mkdir -p %s/", obs);
	run_sys_call(command);
	sprintf(command, "mkdir -p %s/", plot);
	run_sys_call(command);
}
#endif

/* print average measures */
#ifdef MEASURE
void print_realization_statistics( setup_t *s ){
	for(int i = 0; i < s->R; ++i){
		printf("T[%i]=%.2f", i, s->T[i]);
		for(int j=0; j<NUM_PHYSICAL_VALUES; ++j){
			printf(" ,%s=[%.3f,%.3f,%.3f%%,%.2f]", symbols[j], s->obstable[i].rdata[j].mean, s->obstable[i].rdata[j].stdev, 100.0 * s->obstable[i].rdata[j].stdev / s->obstable[i].rdata[j].mean, s->obstable[i].rdata[j].correlation);
		}
		printf("\n");
	}
}
#endif

void write_parameters(setup_t* s){
	FILE *fw;
    fw = fopen(string(string(s->obsfolder) + "/parameters.txt").c_str(), "w");
    if(!fw){
            fprintf(stderr, "error opening file %s for writing\n", "parameters.txt");
            exit(1);
    }
    fprintf(fw, "launch parameters:\n");
    fprintf(fw, "bin/trueke -l %i %i     -t %f %f      -a %i %i %i %i    -h %f \
            -s %i %i %i %i %i %i    -br %i %i    -z %lu     -g %i\n", s->L, s->Ro,
        s->TR, s->dT, s->atrials, s->ains, s->apts, s->ams, s->h, s->pts,
        s->mzone, s->ds, s->ms, s->fs, s->period, s->blocks, s->realizations,
        s->oseed, s->ngpus);
    fclose(fw);
}
/* write the realization statistics */
#ifdef MEASURE
void write_realization_statistics( setup_t* s){
	FILE *fw;
	for(int j = 0; j < NUM_PHYSICAL_VALUES; j++){
		fw = fopen(string(string(s->obsfolder) + "/" + string(filenames[j])).c_str(), "w");
		//printf("plot file: %s\n", string(string(s->obsfolder) + "/" + string(filenames[j])).c_str());
		if(!fw){
				fprintf(stderr, "error opening file %s for writing\n", filenames[j]);
				exit(1);
		}
		fprintf(fw, "#T\t\t%s\t\tbstdev\t\tbsterr\t\tbcorr\t\trstdev\t\trsterr\t\trcorr\n", symbols[j]);
		/* temperature order */
		for(int i = 0; i < s->R; ++i){
			fprintf(fw, "%e\t%e\t%e\t%e\t%e\t%e\t%e\t%e\n", s->T[i], 
															s->obstable[i].rdata[j].mean, 
															s->obstable[i].rdata[j].avbstdev, 
															abs(s->obstable[i].rdata[j].avbstdev/sqrt(s->obstable[i].rdata[j].n)),
															abs(s->obstable[i].rdata[j].avbcorrelation), 
															s->obstable[i].rdata[j].stdev, 
															abs(s->obstable[i].rdata[j].stdev/sqrt(s->obstable[i].rdata[j].n)),
															s->obstable[i].rdata[j].correlation );

			fflush(fw);	
			//printf("writing replica %i\n", s->trs[i]);
		}
		fclose(fw);
	}
}
#endif

/* write the binder as : Qfrac = <m^4>/<m^2>^2 */
#ifdef MEASURE
void write_binder( setup_t *s ){
	FILE *fw;
	double x, y, dx, dy;
	fw = fopen(string(string(s->obsfolder) + "/" + string("binder.dat")).c_str(), "w");
	if(!fw){
		fprintf(stderr, "error opening file %s for writing\n", "binder.dat");
		exit(1);
	}
	fprintf(fw, "#T          B           dB\n");
	for(int r=0; r<s->R; ++r){
		x = s->obstable[r].rdata[QUADM_POS].mean;
		y = s->obstable[r].rdata[SQM_POS].mean;
		dx = s->obstable[r].rdata[QUADM_POS].stdev / sqrt((double)s->obstable[r].rdata[QUADM_POS].n);
		dy = s->obstable[r].rdata[SQM_POS].stdev / sqrt((double)s->obstable[r].rdata[SQM_POS].n);
		fprintf(fw, "%e\t%e\t%e\n", s->T[r], 	s->obstable[r].rdata[QUADM_POS].mean / (s->obstable[r].rdata[SQM_POS].mean * s->obstable[r].rdata[SQM_POS].mean),  (dx*y - dy*x) / (y*y));
	}
	fclose(fw);
}

void write_specific_heat( setup_t *s ){
	FILE *fw;
	double rsqT, dx, dy;
	fw = fopen(string(string(s->obsfolder) + "/" + string("Zspecific_heat.dat")).c_str(), "w");
	if(!fw){
		fprintf(stderr, "error opening file %s for writing\n", "Zspecific_heat.dat");
		exit(1);
	}
	fprintf(fw, "#T          C         dC\n");
	for(int r=0; r<s->R; ++r){
		rsqT = 1.0/((double)s->T[r] * (double)s->T[r]);
		dx = s->obstable[r].rdata[SQE_POS].stdev / sqrt((double)s->obstable[r].rdata[SQE_POS].n);
		dy = s->obstable[r].rdata[ZSQE_POS].stdev / sqrt((double)s->obstable[r].rdata[ZSQE_POS].n);
		fprintf(fw, "%e\t%e\t%e\n", s->T[r], 	(double)s->N * (s->obstable[r].rdata[SQE_POS].mean - s->obstable[r].rdata[ZSQE_POS].mean) * rsqT, abs(dx-dy));
	}
	fclose(fw);
}

void write_susceptibility( setup_t *s ){
	FILE *fw;
	double rT, dx, dy;
	fw = fopen(string(string(s->obsfolder) + "/" + string("Zsusceptibility.dat")).c_str(), "w");
	if(!fw){
		fprintf(stderr, "error opening file %s for writing\n", "Zsusceptibility.dat");
		exit(1);
	}
	fprintf(fw, "#T          X           dX\n");
	for(int r=0; r<s->R; ++r){
		rT = 1.0/((double)s->T[r]);
		dx = s->obstable[r].rdata[SQM_POS].stdev / sqrt((double)s->obstable[r].rdata[SQM_POS].n);
		dy = s->obstable[r].rdata[ZSQM_POS].stdev / sqrt((double)s->obstable[r].rdata[ZSQM_POS].n);
		fprintf(fw, "%e\t%e\t%e\n", s->T[r], 	(double)s->N * (s->obstable[r].rdata[SQM_POS].mean - s->obstable[r].rdata[ZSQM_POS].mean) * rT, abs(dx-dy));
	}
	fclose(fw);
}
#endif

/* print the magnetic field H */
void printH(int *h, int N){
	for(int i=0; i < N; i++)
		printf("%i \n", h[i]);
}

/* accumulate a variance step, for statistics */
void variance_step(double x,  int *n, double *mean, double *w1, double *w2, const double x1, double *lastx){
	
	*n = *n + 1;
	double d = x - *mean;
	double lastmean = *mean;

	*mean 	+= d/(*n);
	*w2 	+= d*(x-(*mean));

	// correlation
	*w1 += (x - *mean) * (*lastx - *mean) + (*n - 2) * (pow(*mean, 2.0) - pow(lastmean, 2.0)) + ((2.0*(*n-1)*(lastmean)) - x1 - *lastx) * (lastmean - *mean);
	*lastx = x;
}

/* reset per realization statistics */
void reset_realization_statistics( setup_t *s, int R){
	/* we do not need to use replica or temp order, it's just a reset */
	for(int r=0; r<R; ++r){
		for(int j=0; j<NUM_PHYSICAL_VALUES; j++){
			s->obstable[r].rdata[j].mean = 0.0;
			s->obstable[r].rdata[j].avbstdev = 0.0;
			s->obstable[r].rdata[j].avbcorrelation = 0.0;
			s->obstable[r].rdata[j].stdev = 0.0;
			s->obstable[r].rdata[j].correlation = 0.0;
			s->obstable[r].rdata[j].n = 0.0;
			s->obstable[r].rdata[j].w1 = 0.0;
			s->obstable[r].rdata[j].w2 = 0.0;
			s->obstable[r].rdata[j].x1 = 0.0;
			s->obstable[r].rdata[j].lastx = 0.0;
		}
	}
}

/* reset per block statistics */
void reset_block_statistics( setup_t *s, int tid, int a, int b){
	int q;
	for(int k = a; k < b; ++k){
		q = s->rts[k];
		for(int j=0; j<NUM_PHYSICAL_VALUES; j++){
			s->obstable[q].bdata[j].mean = 0.0;
			s->obstable[q].bdata[j].n = 0.0;
			s->obstable[q].bdata[j].w1 = 0.0;
			s->obstable[q].bdata[j].w2 = 0.0;
			s->obstable[q].bdata[j].x1 = 0.0;
			s->obstable[q].bdata[j].lastx = 0.0;
		}
	}
}

/* reset per mcmc statistics */
void reset_mcmc_statistics( setup_t *s, int tid, int a, int b){
	int q;
	for(int k = a; k < b; ++k){
		q = s->rts[k];
		s->obstable[q].mdata.E 		= 0.0;
		s->obstable[q].mdata.sqE 	= 0.0;
		s->obstable[q].mdata.M 		= 0.0;
		s->obstable[q].mdata.sqM 	= 0.0;
		s->obstable[q].mdata.quadM 	= 0.0;
		s->obstable[q].mdata.F		= 0.0;
	}
}

int getgpuid(){
	int gpu;
	cudaGetDevice(&gpu);
	return gpu;
}

/* measured mcmc simulation */
void simulation(setup_t *s, int tid, int a, int b){
	for(int i = 0; i < s->fs; i++){
		/* use replica order */
		for(int k = a; k < b; ++k){
			kernel_metropolis<<< s->mcgrid, s->mcblock, 0, s->rstream[k] >>>(s->N, s->L, s->dlat[k], s->dH[tid], s->h, -2.0f/s->T[s->trs[k]], s->pcga[k], s->pcgb[k], 0);
		}
		cudaDeviceSynchronize();
		cudaCheckErrors("simulation: kernel metropolis white launch");
		for(int k = a; k < b; ++k){
			kernel_metropolis<<< s->mcgrid, s->mcblock, 0, s->rstream[k] >>>(s->N, s->L, s->dlat[k], s->dH[tid], s->h, -2.0f/s->T[s->trs[k]], s->pcga[k], s->pcgb[k], 1);
		}
		cudaDeviceSynchronize();
		cudaCheckErrors("simulation: kernel metropolis black launch");
		
#ifdef MEASURE
		/* accumulate mcmc statistics */
		accum_mcmc_statistics(s, tid, a, b);
#endif
	}
}

/* accum per block measures. this code runs inside an openmp parallel pragma */
#ifdef MEASURE
void accum_mcmc_statistics( setup_t *s, int tid, int a, int b){
	double E, M, F;
	double invN = 1.0/(double)s->N;
	double onethird = 0.333333333333333333333;
	//int L = s->L;
	/* quick reset of the device reduction variables */
	cudaDeviceSynchronize();
	cudaCheckErrors("beginning accum_mcmc_statistics");
	kernel_reset<float><<< (b-a + BLOCKSIZE1D - 1)/BLOCKSIZE1D, BLOCKSIZE1D, 0, s->rstream[a] >>> (s->dE[tid], b-a, 0.0f);
	kernel_reset<int><<< (b-a + BLOCKSIZE1D - 1)/BLOCKSIZE1D, BLOCKSIZE1D, 0, s->rstream[a + (1 % (b-a))] >>> (s->dM[tid], b-a, 0);
	kernel_reset<float3><<< (b-a + BLOCKSIZE1D - 1)/BLOCKSIZE1D, BLOCKSIZE1D, 0, s->rstream[a + (2 % (b-a))] >>> (s->dF1[tid], b-a, make_float3(0.0f, 0.0f, 0.0f));
	kernel_reset<float3><<< (b-a + BLOCKSIZE1D - 1)/BLOCKSIZE1D, BLOCKSIZE1D, 0, s->rstream[a + (3 % (b-a))] >>> (s->dF2[tid], b-a, make_float3(0.0f, 0.0f, 0.0f));
	cudaDeviceSynchronize();	
	cudaCheckErrors("after resets accum_mcmc_statistics");
	/* use replica order */
	for(int k = a; k < b; ++k){
		redenergy(s, tid, a, b, k);
		redmagnetization(s, tid, a, b, k);
		redcorrlen(s, tid, a, b, k);
	}
	/* sync all streams for all reductions */
	cudaDeviceSynchronize();	
	cudaCheckErrors("E, M, F reductions");

	/* copy reductions to host side */
	cudaMemcpyAsync(s->E + a, s->dE[tid], (b-a)*sizeof(float), cudaMemcpyDeviceToHost, s->rstream[a]);
	cudaMemcpyAsync(s->M + a, s->dM[tid], (b-a)*sizeof(int), cudaMemcpyDeviceToHost, s->rstream[a + (1 % (b-a))]);
	cudaMemcpyAsync(s->F1 + a, s->dF1[tid], (b-a)*sizeof(float3), cudaMemcpyDeviceToHost, s->rstream[a + (2 % (b-a))]);
	cudaMemcpyAsync(s->F2 + a, s->dF2[tid], (b-a)*sizeof(float3), cudaMemcpyDeviceToHost, s->rstream[a + (3 % (b-a))]);
	cudaDeviceSynchronize();	
	cudaCheckErrors("cuda memcpys accum_mcmc_statistics");
    #pragma omp barrier
	/* accumulate (write) data in temp order, but read from replica order */
	for(int k = a; k < b; ++k){
		int q = s->trs[k];
		/* read in replica order */
		//printf("mcmc:R%i   E = %f    M = %f\n", k, E, M);

		/* test CPU measurements with double and see any real improvements for using double precision */
		//cudaMemcpy(s->hlat[k], s->dlat[k], sizeof(int)*s->N, cudaMemcpyDeviceToHost);
		//E 	= (double)compute_E(s->hlat[k], s->hH, s->h, L, L, L)/(double)(s->N);
		//M 	= abs((double)reduceM(s->hlat[k], L, L, L))/(double)(s->N);
		//F 	= (double)compute_F(s->hlat[k], L, L, L)/(double)(s->N);

		//double cE = compute_E(s->hlat[k], s->hH, s->h, s->L)/(double)s->N;
		//printf("cpuE = %f     gpuE = %f\n", cE, E);
		//getchar();
		//double cpuF = compute_F(s->hlat[k], s->L);
		//printf("sF1[%i] = (%f, %f, %f)     sF2[%i] = (%f, %f, %f)\n", k, s->F1[k].x, s->F1[k].y, s->F1[k].z, k, s->F2[k].x, s->F2[k].y, s->F2[k].z);

		/* GPU measurement */
		E = (double)s->E[k]/(double)s->N;
		M = abs((double)s->M[k]/(double)s->N);	
		F = (double)((onethird * invN * invN)*(s->F1[k].x * s->F1[k].x + s->F1[k].y * s->F1[k].y + s->F1[k].z * s->F1[k].z + s->F2[k].x * s->F2[k].x + s->F2[k].y * s->F2[k].y + s->F2[k].z * s->F2[k].z));
		//printf("mcmc[%i]      E = %f   M = %f   F = %f\n", q, E, M, F);
		//printf("m = ");
		//for(int i = a; i < b; ++i){
		//	printf("[%f] ", s->T[s->rts[i]]);
		//}
		//printf("\n");
		//printarray<int>(s->rts, b-a, "m");
		//getchar();
		/* write in temp order */
		s->obstable[q].mdata.E		+= E;
		s->obstable[q].mdata.sqE 	+= E*E;
		s->obstable[q].mdata.M		+= M;
		s->obstable[q].mdata.sqM 	+= M*M;
		s->obstable[q].mdata.quadM	+= M*M*M*M;
		//double qm1 = M*M*M*M;
        //double qm2 = quad(M);
        //printf(" qm1 = %.30f               qm2 = %.30f\n", qm1, qm2);
        //getchar();
		s->obstable[q].mdata.F		+= F;
	}
    #pragma omp barrier
	//for(int i = a; i < b; ++i){
	//	printf("obstable[%i].mdata.E = %f\n", i, s->obstable[i].mdata.E);
	//}
}
#endif

/* accumulate per block statistics */
#ifdef MEASURE
void accum_block_statistics( setup_t *s, int tid, int a, int b ){
	int q;
	double invsteps = 1.0/(double)s->fs;
	/* array of observables, ommiting the special ones */
	double values[NUM_PHYSICAL_VALUES - NUM_SPECIAL];
	double N = (double)s->N;
	/* use temp order */
    #pragma omp barrier
	for(int k = a; k < b; ++k){
		q = s->trs[k];
		values[E_POS] 	= s->obstable[q].mdata.E * invsteps;
        //printf("tid=%i k=%i T=%f  E=%f\n", tid, k, s->T[q], values[E_POS]); 
		values[M_POS] 	= s->obstable[q].mdata.M * invsteps;
		values[SQE_POS] = s->obstable[q].mdata.sqE * invsteps;
		values[SQM_POS] = s->obstable[q].mdata.sqM * invsteps;
		values[QUADM_POS] = s->obstable[q].mdata.quadM * invsteps;
		values[Xd_POS] 	= N * (values[M_POS] * values[M_POS]) * invsteps;
		values[F_POS] 	= s->obstable[q].mdata.F * invsteps;
		for(int j=0; j<NUM_PHYSICAL_VALUES - NUM_SPECIAL; j++){
			if( s->obstable[q].bdata[j].n == 0 ){
				s->obstable[q].bdata[j].x1 = values[j];
			}
			variance_step(	values[j], 	&(s->obstable[q].bdata[j].n), 		&(s->obstable[q].bdata[j].mean),	&(s->obstable[q].bdata[j].w1), &(s->obstable[q].bdata[j].w2), 	s->obstable[q].bdata[j].x1, 		&(s->obstable[q].bdata[j].lastx) );
		}
	}
    #pragma omp barrier
	//for(int k = a; k < b; ++k){
	//	printf("R%i  MEAN BLOCK        E = %f       M = %f\n\n", k, s->obstable[k].bdata[E_POS].mean, s->obstable[k].bdata[M_POS].mean);
	//}
}
#endif

/* accum realization statistics */
#ifdef MEASURE
void accum_realization_statistics( setup_t *s, int tid, int a, int b, int realizations ){
    #pragma omp barrier
	for(int k = a; k < b; ++k){
		/* traverse replicas, but access in temp order */
		int q = s->trs[k];
		/* the special observables are ommited here */
		for(int j=0; j<NUM_PHYSICAL_VALUES - NUM_SPECIAL; ++j){
			if( s->obstable[q].rdata[j].n == 0 ){
				s->obstable[q].rdata[j].x1 = s->obstable[q].bdata[j].mean;
			}
			variance_step(s->obstable[q].bdata[j].mean, &(s->obstable[q].rdata[j].n), &(s->obstable[q].rdata[j].mean),&(s->obstable[q].rdata[j].w1), &(s->obstable[q].rdata[j].w2),
							s->obstable[q].rdata[j].x1,&(s->obstable[q].rdata[j].lastx) );

			s->obstable[q].rdata[j].avbstdev 				+= sqrt( s->obstable[q].bdata[j].w2 / ((double)s->obstable[q].bdata[j].n - 1.0)) / (double)realizations;
			s->obstable[q].rdata[j].avbcorrelation 		+= (s->obstable[q].bdata[j].w1 / s->obstable[q].bdata[j].w2) / (double)realizations;
		}
		/* auxiliary variables for computing observables */
		double A, sqA, N, L, val, F, T;
		L = (double)s->L;
		N = (double)s->N;
		T = (double)s->T[q];


        #pragma omp barrier
		/* specific heat */
		A = s->obstable[q].bdata[E_POS].mean;
		sqA = s->obstable[q].bdata[SQE_POS].mean;
		val = N * (sqA - A*A) / (T*T); 
		if( s->obstable[q].rdata[C_POS].n == 0 ){
			s->obstable[q].rdata[C_POS].x1 = val;
		}
        //printf("tid=%i   k=%i  T=%f    C=%.10f\n", tid, k, T, val);
        //getchar();

		variance_step(val, &(s->obstable[q].rdata[C_POS].n), &(s->obstable[q].rdata[C_POS].mean), &(s->obstable[q].rdata[C_POS].w1), &(s->obstable[q].rdata[C_POS].w2), s->obstable[q].rdata[C_POS].x1, 
						&(s->obstable[q].rdata[C_POS].lastx));

		
		/* susceptibility */
		A = s->obstable[q].bdata[M_POS].mean;
		sqA = s->obstable[q].bdata[SQM_POS].mean;
		val = N * (sqA - A*A) / T; 
		if( s->obstable[q].rdata[X_POS].n == 0 ){
			s->obstable[q].rdata[X_POS].x1 = val;
		}
		variance_step(val, &(s->obstable[q].rdata[X_POS].n), &(s->obstable[q].rdata[X_POS].mean), &(s->obstable[q].rdata[X_POS].w1), &(s->obstable[q].rdata[X_POS].w2), s->obstable[q].rdata[X_POS].x1, 
						&(s->obstable[q].rdata[X_POS].lastx));

		/* corrlen */
		sqA = s->obstable[q].bdata[SQM_POS].mean;
		F = s->obstable[q].bdata[F_POS].mean;
		//printf("sqM = %f        F = %f          sqM/F = %f\n", sqA, F, sqA/F);
		if( sqA >= F ){
			val = sqrt( sqA/F - 1.0 ) / L;
			if( s->obstable[q].rdata[CORRLEN_POS].n == 0 ){
				s->obstable[q].rdata[CORRLEN_POS].x1 = val;
			}
			variance_step(val, &(s->obstable[q].rdata[CORRLEN_POS].n), &(s->obstable[q].rdata[CORRLEN_POS].mean), &(s->obstable[q].rdata[CORRLEN_POS].w1), &(s->obstable[q].rdata[CORRLEN_POS].w2),	
							s->obstable[q].rdata[CORRLEN_POS].x1, &(s->obstable[q].rdata[CORRLEN_POS].lastx));
		}					

		/* ZSQE */
		val = pow(s->obstable[q].bdata[E_POS].mean, 2.0);
		if( s->obstable[q].rdata[ZSQE_POS].n == 0 ){
			s->obstable[q].rdata[ZSQE_POS].x1 = val;
		}
		variance_step(val, &(s->obstable[q].rdata[ZSQE_POS].n), &(s->obstable[q].rdata[ZSQE_POS].mean), &(s->obstable[q].rdata[ZSQE_POS].w1), &(s->obstable[q].rdata[ZSQE_POS].w2),	
						s->obstable[q].rdata[ZSQE_POS].x1, &(s->obstable[q].rdata[ZSQE_POS].lastx));

		/* ZSQM */
		val = pow(s->obstable[q].bdata[M_POS].mean, 2.0);
		if( s->obstable[q].rdata[ZSQM_POS].n == 0 ){
			s->obstable[q].rdata[ZSQM_POS].x1 = val;
		}
		variance_step(val, &(s->obstable[q].rdata[ZSQM_POS].n), &(s->obstable[q].rdata[ZSQM_POS].mean), &(s->obstable[q].rdata[ZSQM_POS].w1), &(s->obstable[q].rdata[ZSQM_POS].w2),	
						s->obstable[q].rdata[ZSQM_POS].x1, &(s->obstable[q].rdata[ZSQM_POS].lastx));

		/* exchange rates */
		val = s->avex[k];
		if(s->obstable[q].rdata[EXCHANGE_POS].n == 0){
			s->obstable[q].rdata[EXCHANGE_POS].x1 = val;
		}
		variance_step(val, &(s->obstable[q].rdata[EXCHANGE_POS].n), &(s->obstable[q].rdata[EXCHANGE_POS].mean), &(s->obstable[q].rdata[EXCHANGE_POS].w1), &(s->obstable[q].rdata[EXCHANGE_POS].w2),
							s->obstable[q].rdata[EXCHANGE_POS].x1, &(s->obstable[q].rdata[EXCHANGE_POS].lastx));
	}
    #pragma omp barrier
}
#endif

/* produce realization statistics */
void make_realization_statistics( setup_t *s ){
	/* use replica order */
	for(int i=0; i<s->R; ++i){
		for(int j=0; j<NUM_PHYSICAL_VALUES; j++){
			s->obstable[i].rdata[j].stdev 			= sqrt(s->obstable[i].rdata[j].w2 / ((double)s->obstable[i].rdata[j].n - 1.0));
			s->obstable[i].rdata[j].correlation 	= abs((s->obstable[i].rdata[j].w1 / s->obstable[i].rdata[j].w2));
		}
	}
}

/* CPU random configuration for the lattice */
void random_configuration(int N, int* lat, uint64_t *state, uint64_t *seq){
	for(int i=0; i < N; i++){
		if(gpu_rand01(state, seq) >= 0.5f)
			lat[i] =  1;
		else
			lat[i] = -1;
	}	
}

/* CPU one configuration for the lattice */
void one_configuration(int N, int* lat){
	for(int i=0; i < N; i++){
			lat[i] =  1;
	}	
}

/* CPU random magnetic field H */
void random_Hi(int N, int* Hlat, uint64_t *state, uint64_t *seq){
	for(int i=0; i < N; i++){
		if(gpu_rand01(state, seq) >= 0.5)
			Hlat[i] =  1;
		else
			Hlat[i] = -1;
	}	
}

/* equilibration */
void equilibration(setup_t *s, int tid, int a, int b){
		if(tid == 0){ printf("equilibration......0%%"); fflush(stdout);}
		for(int i = 0; i < s->ds; i++){
			/* replica order */
			for(int k = a; k < b; ++k){
				//kernel_metropolis<<< s->mcgrid, s->mcblock, 0, s->rstream[k] >>>(s->N, s->L, s->dlat[k], s->dH[tid], s->h, -2.0f/s->T[s->trs[k]], s->dstates[k], 0);
				kernel_metropolis<<< s->mcgrid, s->mcblock, 0, s->rstream[k] >>>(s->N, s->L, s->dlat[k], s->dH[tid], s->h, -2.0f/s->T[s->trs[k]], s->pcga[k], s->pcgb[k], 0);
			}
			cudaDeviceSynchronize();
			cudaCheckErrors("equilibration: kernel metropolis white");
			for(int k = a; k < b; ++k){
				//kernel_metropolis<<< s->mcgrid, s->mcblock, 0, s->rstream[k] >>>(s->N, s->L, s->dlat[k], s->dH[tid], s->h, -2.0f/s->T[s->trs[k]], s->dstates[k], 1);
				kernel_metropolis<<< s->mcgrid, s->mcblock, 0, s->rstream[k] >>>(s->N, s->L, s->dlat[k], s->dH[tid], s->h, -2.0f/s->T[s->trs[k]], s->pcga[k], s->pcgb[k], 1);
			}
			cudaDeviceSynchronize();
			cudaCheckErrors("equilibration: kernel metropolis black");
			if( tid == 0 ){ printf("\requilibration......%i%%", (100*(i+1))/s->ds); fflush(stdout); }
		}
		if(tid == 0){printf("\n"); fflush(stdout);}
}

/* metropolis */
void metropolis(setup_t *s, int tid, int a, int b, int ms){
	for(int i = 0; i < ms; ++i){
		/* replica order => use m to access temperature */
		//printf("\n");
		for(int k = a; k < b; ++k){
			//printf("0 - simulating R%i --> T%i=%f\n", k, s->trs[k], s->T[s->trs[k]]);
			kernel_metropolis<<< s->mcgrid, s->mcblock, 0, s->rstream[k] >>>(s->N, s->L, s->dlat[k], s->dH[tid], s->h, -2.0f/s->T[s->trs[k]], s->pcga[k], s->pcgb[k], 0);
		}
		cudaDeviceSynchronize(); 
		cudaCheckErrors("mcmc: kernel metropolis white launch");
		//printf("\n");
		for(int k = a; k < b; ++k){
			//printf("1 - simulating R%i --> T%i=%f\n", k, s->trs[k], s->T[s->trs[k]]);
			kernel_metropolis<<< s->mcgrid, s->mcblock, 0, s->rstream[k] >>>(s->N, s->L, s->dlat[k], s->dH[tid], s->h, -2.0f/s->T[s->trs[k]], s->pcga[k], s->pcgb[k], 1);
		}
		cudaDeviceSynchronize(); 
		cudaCheckErrors("mcmc: kernel metropolis black launch");
	}
}

/* metropolis */
void adapt_metropolis(setup_t *s, int tid, int ms){
	//printf("simulando papa\n"); fflush(stdout);
	for(int i = 0; i < ms; ++i){
        //printf("adapt_metro1 jejejej\n"); fflush(stdout);
		/* replica order => use m to access temperature */
		for(int k = 0; k < s->gpur[tid]; ++k){
            //printf("adapt_metro2 jejejej\n"); fflush(stdout);
			//printf("0 - simulating R%i --> T%i=%f\n", k, s->trs[k], s->T[s->trs[k]]); fflush(stdout);
			//getchar();
			kernel_metropolis<<< s->mcgrid, s->mcblock, 0, s->arstream[tid][k] >>>(s->N, s->L, s->mdlat[tid][k], s->dH[tid], s->h, -2.0f/s->aT[s->atrs[tid][k].f][s->atrs[tid][k].i], s->apcga[tid][k], s->apcgb[tid][k], 0);
		}
        //printf("adapt_metro3 jejejej\n"); fflush(stdout);
		cudaDeviceSynchronize();
		cudaCheckErrors("mcmc: kernel metropolis white launch");
		for(int k = 0; k < s->gpur[tid]; ++k){
			//printf("1 - simulating R%i --> T%i=%f\n", k, s->trs[k], s->T[s->trs[k]]);
			kernel_metropolis<<< s->mcgrid, s->mcblock, 0, s->arstream[tid][k] >>>(s->N, s->L, s->mdlat[tid][k], s->dH[tid], s->h, -2.0f/s->aT[s->atrs[tid][k].f][s->atrs[tid][k].i], s->apcga[tid][k], s->apcgb[tid][k], 1);
		}
		cudaDeviceSynchronize();
		cudaCheckErrors("mcmc: kernel metropolis black launch");
	}
}

template <typename T>
void printarray(T *a, int n, const char *name){
	cout << name << "\t = [";
	for(int i = 0; i < n; ++i){
		cout << a[i] << ", ";
	}
	printf("]\n");
}

/* print indexed fragmented array */
template <typename T>
void printindexarray(T *a, int *ind, int n, const char *name){
	cout << name << "\t = [";
    for(int i = 0; i < n; ++i){
        cout << a[ind[i]] << ", ";
    }
	printf("]\n");
}

/* print indexed fragmented array */
template <typename T>
void printindexarrayfrag(T **a, findex** ind, int n, int *m, const char *name){
	cout << name << "\t = [";
    for(int i = 0; i < n; ++i){
        for(int j = 0; j < m[i]; ++j){
            cout << a[ind[i][j].f][ind[i][j].i] << ", ";
        }
        printf("    ");
    }
	printf("]\n");
}

template <typename T>
void printarrayfrag(T **a, int n, int *m, const char *name){
	cout << name << "\t = [";
    for(int i = 0; i < n; ++i){
        for(int j = 0; j < m[i]; ++j){
            cout << a[i][j] << ", ";
        }
    }
	printf("]\n");
}
/* GPU random magnetic field H */
//Modified by Ang Li, 12/08/2017
//=================================================
void adapt_hdist(setup_t *s, int tid, ncclComm_t* comm)
{
    if( tid == 0 ) {
        kernel_reset_random_gpupcg<<< s->lgrid, s->lblock>>>(s->dH[tid], s->N, s->apcga[tid][0], s->apcgb[tid][0]);	
		cudaCheckErrors("prng random distribution H");
    }
#pragma omp barrier
    NCCLCHECK(ncclBcast((void*)s->dH[tid], s->N, ncclInt, 0, *comm, 0));
}

/* GPU random magnetic field H */
void hdist(setup_t *s, int tid, int a, int b, ncclComm_t* comm){
    /* generate dist in multiple GPUs */
	#pragma omp barrier
	if( tid == 0 ){
		/* we pass the first prng array from the corresponding GPU */
        kernel_reset_random_gpupcg<<< s->lgrid, s->lblock>>>(s->dH[tid], s->N, s->pcga[a], s->pcgb[a]);	
		cudaCheckErrors("prng random distribution H");
	}
	#pragma omp barrier
	/* threads tid > 0 copy with p2p access, from GPU of tid0 */ 

#pragma omp barrier
    NCCLCHECK(ncclBcast((void*)s->dH[tid], s->N, ncclInt, 0, *comm, 0));
#pragma omp barrier

}
//=================================================

/* physical results */
#ifdef MEASURE
void physical_results(setup_t *s){
	make_realization_statistics(s);
	/* write special average measures */
	write_binder(s);
	write_specific_heat(s);
	write_susceptibility(s);
	/* write average measures */
	write_realization_statistics(s);
    write_parameters(s);
}
#endif

/* free the GPUs, reset them */
void freegpus(setup_t *s){
	for(int i = 0; i < s->ngpus; ++i){
        cudaSetDevice(s->gpus[i].i);
        cudaDeviceReset();
	}
}

/* free memory for CPU and GPU */
void freemem(setup_t *s){
	for(int i = 0; i < s->R; ++i){
		//printf("feeing replica data %i.....", r); fflush(stdout);
		cudaFree(s->dlat[i]);
		free(s->hlat[i]);
		//printf("ok\n"); fflush(stdout);
	}
	for(int i = 0; i < s->ngpus; ++i){
		cudaFree(s->dH[i]);
	}
	free(s->dlat);
	free(s->hlat);
	free(s->E);
	free(s->exE);
	free(s->hH);
	free(s->M);
	free(s->F1);
	free(s->F2);
	free(s->ex);
	free(s->avex);
#ifdef MEASURE
	free(s->obstable);
#endif
}

/* move frag position one to the left */
void fgoleft(setup_t *s, findex_t *frag){
	frag->i -= 1;
	if(frag->i < 0){
		frag->f -= 1;
		frag->i = s->gpur[frag->f] - 1;
	}
    if(frag->f < 0){
        *frag = (findex_t){-1,-1};
    }
}

/* move frag position one to the right */
void fgoright(setup_t *s, findex_t *frag){
	frag->i += 1;
	if(frag->i >= s->gpur[frag->f]){
		frag->f += 1;
		frag->i = 0;
	}
    if(frag->f >= s->ngpus){
        *frag = (findex_t){-2,-2};
    }
}

/* get the left frag position */
findex_t fgetleft(setup_t *s, findex_t frag){
	findex_t out = frag;
	out.i -= 1;
	if(out.i < 0){
		out.f -= 1;
		out.i = s->gpur[out.f] - 1;
	}
    if(out.f < 0){
        out = (findex_t){-1,-1};
    }
	return out;
}

/* get the right frag position */
findex_t fgetright(setup_t *s, findex_t frag){
	findex_t out = frag;
	out.i -= 1;
	if(out.i < 0){
		out.f -= 1;
		out.i = s->gpur[out.f] - 1;
	}
    if(out.f >= s->ngpus){
        out = (findex_t){-2,-2};
    }
	return out;
}

/* move frag position one to the left */
void fshiftleft(setup_t *s, findex_t *frag){
	frag->i -= 1;
	if(frag->i < 0){
		frag->f -= 1;
		frag->i = s->gpur[frag->f] - 1;
	}
    if(frag->f < 0){
        *frag = (findex_t){s->ngpus-1, s->gpur[s->ngpus-1]};
    }
}

/* shift frag position one to the right */
void fshiftright(setup_t *s, findex_t *frag){
	frag->i += 1;
	if(frag->i >= s->gpur[frag->f]){
		frag->f += 1;
		frag->i = 0;
	}
    if(frag->f >= s->ngpus){
        *frag = (findex_t){0,0};
    }
}
#endif
