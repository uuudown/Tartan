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
#ifndef ADAPT_H
#define ADAPT_H

#include <nccl.h>

#define NCCLCHECK(cmd) do { \
    ncclResult_t r=cmd;\
    if (r!=ncclSuccess){\
        printf("Failed, NCCL error %s: %d '%s'\n", __FILE__, __LINE__, ncclGetErrorString(r));\
        exit(EXIT_FAILURE);\
    }\
} while(0) \

/* functions for adapt */
int adapt_exchange(setup_t *s, int tid, int p);
void adapt_ptenergies(setup_t *s, int tid);
void adapt_swap(setup_t *s, findex_t a, findex_t b );
float adapt( setup_t *s );
double avfragex( setup_t *s );
double minfragex( setup_t *s );
double maxfragex( setup_t *s );
void insert_temps(setup_t *s);
void rebuild_temps(setup_t *s);
void rebuild_indices(setup_t *s);

/* adapt(): adapt temperatures --> return time as float */
float adapt( setup_t *s ){
    /* file for trial data */
	sdkStartTimer(&(s->gtimer));
    FILE *fw = fopen(string(string(s->obsfolder) + string( + "/trials.dat")).c_str(), "w");
    fprintf(fw, "trial  av  min max\n");
	/* print the beginning temp */
	printarrayfrag<float>(s->aT, s->ngpus, s->gpur, "Initial temp set:\naT");
	printf("\n\n");

    //Modified by Ang Li, 12/08/2017
    //=================================================
	int nt = (s->ngpus);
    ncclUniqueId id;
    ncclGetUniqueId(&id);
    ncclComm_t* comm = new ncclComm_t[nt];
    ncclGroupStart();
    for(int i=0; i<nt; i++) {
        cudaSetDevice(i);
        NCCLCHECK(ncclCommInitRank(&comm[i],nt,id,i));
    }
    ncclGroupEnd();
    //=================================================


	/* each adaptation iteration improves the temperature distribution */
	for(int j = 0; j < s->atrials; ++j){
		int tid, r;
		/* progress printing */
		printf("[trial %i of %i]\n", j+1, s->atrials); fflush(stdout);

		/* reset timers */
		sdkResetTimer(&(s->timer));  sdkStartTimer(&(s->timer));
		/* average exchanges */
		#pragma omp parallel private(tid, nt, r) shared(comm)
		{
		    double avex = 0.0;
			/* set the thread */
			adapt_threadset(s, &tid, &nt, &r);
            /* distribution for H */
            adapt_hdist(s, tid, comm+tid);
			/* reset ex counters */
			reset_array<float>(s->aex[tid], r, 0.0f);
			/* reset average ex counters */
			reset_array<float>(s->aavex[tid], r, 0.0f);
			/* reset gpu data */
			adapt_reset_gpudata(s, tid);
			#pragma omp barrier
			/* parallel tempering */
			for(int p = 0; p < s->apts; ++p){
				/* metropolis simulation */
				adapt_metropolis(s, tid, s->ams);
				/* compute energies for exchange */
				adapt_ptenergies(s, tid);
				/* exchange phase */
				avex += (double)adapt_exchange(s, tid, p)/(double)s->apts;
                /* print status */
				if(tid == 0){
					printf("\rpt........%i%%", 100 * (p + 1)/(s->apts)); fflush(stdout);
				}
			}
		}
        sdkStopTimer(&(s->timer));

        double avex = avfragex(s) / (double)(s->R-1);
        double min = minfragex(s);
        double max = maxfragex(s);
        printf(" %.3fs ", sdkGetTimerValue(&(s->timer))/1000.0f);
        printf(" [<ex> = %.3f <min> = %.3f <max> = %.3f]\n\n", avex, min, max);
        fprintf(fw, "%i %f  %f  %f\n", j, avex, min, max); fflush(fw);
        //printarrayfrag<float>(s->aT, s->ngpus, s->gpur, "aT");
        printarrayfrag<float>(s->aex, s->ngpus, s->gpur, "aex");
        printarrayfrag<float>(s->aavex, s->ngpus, s->gpur, "aavex");
        //printindexarrayfrag<float>(s->aexE, s->arts, s->ngpus, s->gpur, "aexE");
		//printf("\n");
        /* place new temperatures at the lowest the exchange rates */
        insert_temps(s);
        /* rebuild temperatures */
        rebuild_temps(s);
        /* rebuild indices */
        rebuild_indices(s);
	}

    //Modified by Ang Li, 12/08/2017
    //=================================================
    ncclGroupStart();
    for (int i=0; i<nt; i++){
        cudaSetDevice(i);
        NCCLCHECK(ncclCommDestroy(comm[i]));
    }
    ncclGroupEnd();
    delete[] comm;
    //=================================================


    fclose(fw);
	sdkStopTimer(&(s->gtimer));
    return sdkGetTimerValue(&(s->gtimer))/1000.0f;
}

double minfragex( setup_t *s ){
    double min = 1.0;
    for(int i = 0; i < s->ngpus; ++i){
        if( i == 0 ){
            for(int j = 1; j < s->gpur[i]; ++j){
                if( s->aavex[i][j] < min )
                    min = s->aavex[i][j];
            }
        }
        else{
            for(int j = 0; j < s->gpur[i]; ++j){
                if( s->aavex[i][j] < min )
                    min = s->aavex[i][j];
            }
        }
    }
    return min;
}

double maxfragex( setup_t *s ){
    double max = 0.0;
    for(int i = 0; i < s->ngpus; ++i){
        if( i == 0 ){
            for(int j = 1; j < s->gpur[i]; ++j){
                if( s->aavex[i][j] > max )
                    max = s->aavex[i][j];
            }
        }
        else{
            for(int j = 0; j < s->gpur[i]; ++j){
                if( s->aavex[i][j] > max )
                    max = s->aavex[i][j];
            }
        }
    }
    return max;
}

double avfragex( setup_t *s ){
    double av = 0.0;
    for(int i = 0; i < s->ngpus; ++i){
        if( i == 0 ){
            for(int j = 1; j < s->gpur[i]; ++j){
                av += s->aavex[i][j] = 2.0 *  s->aex[i][j] / (double)s->apts;
            }
        }
        else{
            for(int j = 0; j < s->gpur[i]; ++j){
                av += s->aavex[i][j] = 2.0 *  s->aex[i][j] / (double)s->apts;
            }
        }
    }
    return av;
}

/* adapt_ptenergies(): adaptation exchange energies */
void adapt_ptenergies(setup_t *s, int tid){
	/* quick reset of the device reduction variables */
	int r = s->gpur[tid];
    kernel_reset<float><<< (r + BLOCKSIZE1D - 1)/BLOCKSIZE1D, BLOCKSIZE1D, 0, s->arstream[tid][0] >>> (s->dE[tid], r, 0.0f);
	cudaDeviceSynchronize();
	/* compute one energy reduction for each replica */
	//printf("adapt_ptsenergies: r = %i\n", r); fflush(stdout);
	for(int k = 0; k < r; ++k){
		/* launch reduction kernel for k-th replica */
		adapt_redenergy(s, tid, k);
	}
	cudaDeviceSynchronize();	cudaCheckErrors("kernel_redenergy");
	cudaMemcpy(s->aexE[tid], s->dE[tid], r*sizeof(float), cudaMemcpyDeviceToHost);
}

/* exchange phase */
int adapt_exchange(setup_t *s, int tid, int p){
	/* count the number of exchanges */
	int ex = 0;
	/* sync openmp threads before entering the exchange phase */
	#pragma omp barrier
	if(tid == 0){
		double delta = 0.0;
		/* a locator for fragmented data */
		findex_t fnow, fleft;
		fnow.f = s->ngpus-1;
		fnow.i = s->gpur[fnow.f]-1;
		/* traverse in reverse temperature order */
		for(int k = s->R-1; k > 0; --k){
            //printf("testing exchange for k=%i\n", k); fflush(stdout);
			/* alternate between odd and even replicas */
			if((k % 2) == (p % 2)){
				fgoleft(s, &fnow);
				continue;
			}
			fleft = fgetleft(s, fnow);
            //printf("fnow = %i  %i             fleft = %i   %i\n", fnow.f, fnow.i,fleft.f, fleft.i);
            //printarrayfrag<float>(s->aT, s->ngpus, s->gpur, "aT");
            //printarrayfrag<float>(s->aexE, s->ngpus, s->gpur, "aexE");
            delta = (1.0f/s->aT[fnow.f][fnow.i] - 1.0f/s->aT[fleft.f][fleft.i]) *
            (s->aexE[s->arts[fleft.f][fleft.i].f][s->arts[fleft.f][fleft.i].i] -
            s->aexE[s->arts[fnow.f][fnow.i].f][s->arts[fnow.f][fnow.i].i]);
            double randme = gpu_rand01(&s->hpcgs, &s->hpcgi);
            //printf("delta=%f exp(-delta) = %f      rand = %f..........", delta, exp(-delta), randme);
			//if( delta < 0.0 || randn() < exp(-delta) ){
			if( delta < 0.0 || randme < exp(-delta) ){
                //printf("YES\n"); fflush(stdout);
				/* swap temperatures */
				adapt_swap(s, fnow, fleft);
				/* global counter */
				ex++;
				/* this array is temp sorted */
				s->aex[fnow.f][fnow.i] += 1.0f;
			}
            else{
                //printf("NO\n"); fflush(stdout);
            }
			fgoleft(s, &fnow);
            //printf("\n");
		}
	}
	/* sync again */
	#pragma omp barrier
	return ex;
}

/* swap temperatures */
void adapt_swap(setup_t *s, findex_t a, findex_t b ){
	findex_t t1, t2, taux, raux;
	t1 = s->arts[a.f][a.i];
	t2 = s->arts[b.f][b.i];
	taux = s->atrs[t1.f][t1.i];
	raux = s->arts[a.f][a.i];

	/* swap rts */
	s->arts[a.f][a.i] = s->arts[b.f][b.i];
	s->arts[b.f][b.i] = raux;

	/* swap trs */
	s->atrs[t1.f][t1.i] = s->atrs[t2.f][t2.i];
	s->atrs[t2.f][t2.i] = taux;
}



/* put the new value well distributed in all GPUs */
void newtemp(setup_t *s, findex_t l){
    findex_t left = fgetleft(s, l);
    float ntemp = (s->aT[l.f][l.i] + s->aT[left.f][left.i])/2.0f;
    //printf("new Temp = (%f + %f)/2 = %f\n", s->aT[l.f][l.i], s->aT[left.f][left.i], ntemp);
    s->aT[s->fam][s->gpur[s->fam]++] = ntemp;
    s->fam = (s->fam + 1) % s->ngpus;
    /* update the number of active replicas */
    s->R++;
	//printf("new R = %i\n", s->R);
}

/* rebuild the temperatures, sorted */
void rebuild_temps(setup_t *s){
    int count  = 0;
    float *flat = (float*)malloc(sizeof(float)*s->R);
    /* flatten the temperatures */
    for(int i=0; i<s->ngpus; ++i){
        for(int j=0; j<s->gpur[i]; ++j){
            flat[count++] = s->aT[i][j];
        }
    }
    //printarray<float>(flat, s->R, "flat");
    qsort(flat, s->R, sizeof(float), floatcomp);
    /* sort them */
    //printarray<float>(flat, s->R, "flat");
    /* fragment the sorted temperatures */
    count = 0;
    for(int i=0; i<s->ngpus; ++i){
        for(int j=0; j<s->gpur[i]; ++j){
            s->aT[i][j] = flat[count++];
        }
    }
    free(flat);
}

/* insert temperatures at the "ins" lowest exchange places */
void insert_temps(setup_t *s){
    /* minheap */
    minHeap hp = initMinHeap(0);
    /* put average exchange rates in a min heap */
    for(int i=s->ngpus-1; i >= 0; --i){
        if(i > 0){
            for(int j = s->gpur[i]-1; j >= 0; --j){
                //printf("inserting [%f, {%i ,%i}] \n", s->aavex[i][j], i, j);
                insertNode(&hp, s->aavex[i][j], (findex_t){i, j});
            }
        }
        else{
            for(int j = s->gpur[i]-1; j > 0; --j){
                //printf("inserting [%f, {%i ,%i}] \n", s->aavex[i][j], i, j);
                insertNode(&hp, s->aavex[i][j], (findex_t){i, j});
            }
        }
    }
    //printf("heap has size = %i      s->ains = %i\n", hp.size, s->ains);
    /* get the lowest "ins" exchange rates */
    for(int i = 0; i < s->ains && hp.size > 0; ++i){
        node nod = popRoot(&hp);
        //printf("heap now is size  = %i:\n", hp.size);
        //levelorderTraversal(&hp);
        newtemp(s, nod.coord);
        //printarrayfrag<float>(s->aT, s->ngpus, s->gpur, "aT");
    }
    //printf("s->R = %i\n", s->R);
    //printarray<int>(s->gpur, s->ngpus, "gpur");
    //printf("sorted temps\n");
    //printarrayfrag<float>(s->aT, s->ngpus, s->gpur, "aT");
    //printarray<int>(s->rpool, s->ngpus, "rpool");
    //printarray<int>(s->gpur, s->ngpus, "gpur");

}

/* rebuild atrs and arts indices */
void rebuild_indices(setup_t *s){
    for(int k = 0; k < s->ngpus; ++k){
        for(int j = 0; j < s->gpur[k]; ++j){
            s->arts[k][j] = s->atrs[k][j] = (findex_t){k, j};
        }
    }
}

#endif
