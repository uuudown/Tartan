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
#ifndef _STRUCTS_H_
#define _STRUCTS_H_

const char *symbols[] = {"E", "M", "sqE", "sqM", "quadM", "Xd", "F", "C", "X", "CORRLEN", "EXCHANGE", "ZSQE", "ZSQM"};
const char *filenames[] = {"energy.dat", "magnetization.dat", "sqenergy.dat", "sqmagnetization.dat", "quadmagnetization.dat", "dis_susceptibility.dat", "F.dat", "specific_heat.dat", "susceptibility.dat", "corrlen.dat", "exchange.dat", "zsqe.dat", "zsqm.dat"};

/* definitions for physical values */
#define NUM_PHYSICAL_VALUES 	13
#define NUM_SPECIAL				6

/* NORMAL: the following are computed at each ptstep */
#define E_POS 			0
#define M_POS 			1
#define SQE_POS			2
#define SQM_POS 		3
#define QUADM_POS 		4
#define Xd_POS 			5
#define F_POS		 	6
/* SPECIAL: the following are computed at each realization */
#define C_POS 			7
#define X_POS 			8
#define CORRLEN_POS		9
#define EXCHANGE_POS	10
#define ZSQE_POS		11
#define ZSQM_POS		12

/* forward struct declarations */
struct realization_data;
struct block_data;
struct mc_data;
struct obset;
struct gpu;
struct setup;
struct findex;
typedef realization_data rdata_t;
typedef block_data bdata_t;
typedef mc_data mcdata_t;
typedef obset obset_t;
typedef gpu gpu_t;
typedef setup setup_t;
typedef findex findex_t;

/* realization statistics data structure */
struct realization_data{
	int n;
	double mean;
	double stdev;
	double correlation;
	double avbstdev;
	double avbcorrelation;
	double w1;
	double w2;
	double x1;
	double lastx;	

};

/* block statistics data structure */
struct block_data{
	double mean;
	int n;
	double w1;
	double w2;
	double x1;
	double lastx;	
};


/* Monte Carlo step statistics data */
struct mc_data{
	double E;
	double sqE;
	double M;
	double sqM;
	double quadM;
	double F;
};


/* obervables set of values */
struct obset{
	rdata_t		rdata[NUM_PHYSICAL_VALUES];
	bdata_t 	bdata[NUM_PHYSICAL_VALUES];
	mcdata_t	mdata;
};


/* gpu info data structure */
struct gpu{
	int i;
	int u;
	int m;
};

/* fragmented index structure */
struct findex{
	int f;
	int i;
};


/* main structure */
struct setup{
	/* arguments / parameters */
	int L, N;
	float TR, dT, h;
	/* adaptation parameters */
	int atrials, ains, apts, ams;
	/* pt and simulation parameters */
	int pts, ds, ms, fs, cs, period;
	int blocks, realizations;
	unsigned long long seed, oseed;
    int R, Ra, Ro;
	int ngpus, fam;

	/* space of computation dimensions */	
	dim3 mcgrid, mcblock, lgrid, lblock, prng_grid, prng_block;
#ifdef MEASURE
	int mzone;
#endif
	obset_t *obstable;
	/* temperature array, increasing */
	float *T, **aT;
	/* exchange array for each temp value */
	float *ex, *avex, **aex, **aavex;
	/* index array for accessing a replica's temp and viceversa */
	int *rts, *trs;
	findex_t **arts, **atrs;

    /* useful data per GPU */
    int *gpur, *rpool;

	/* energy variables */
	float *exE, *E, **dE, **aexE;
	/* magnetization variables */
	int *M, **dM;
	/* correlation length variables */
	float3 *F1, *F2, **dF1, **dF2;

	/* set of host and device lattices, dH is one copy per GPU */
	int **hlat, **dlat, *hH, **dH, ***mdlat;
	/* custom GPU struct with metadata */
	gpu_t *gpus;
	cudaStream_t *rstream, **arstream;
    uint64_t hpcgs, hpcgi;
    uint64_t **pcga, ***apcga;
    uint64_t **pcgb, ***apcgb;
#ifdef MEASURE
	const char *obsfolder;
	const char *plotfolder;
#endif
	/* generic, global and kernel timers */
	StopWatchInterface *timer;
	StopWatchInterface *gtimer;
	StopWatchInterface *ktimer;
};

#endif
