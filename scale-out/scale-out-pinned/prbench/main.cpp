/*
 * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <mpi.h>
#include <math.h>
#include <sys/time.h>
#include <getopt.h>
#include <limits.h>
#include <float.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <unistd.h>
//#include <cuda_profiler_api.h>

#ifndef __APPLE__
#include <fcntl.h> // for posix_fadvise()
#endif

#ifdef __cplusplus
#define __STDC_FORMAT_MACROS 1
#endif
#include <inttypes.h>

#include "global.h"
#include "phsort.h"
#include "utils.h"
#include "cuda_kernels.h"

#define CHUNK_SIZE	(1024*1024)
#define MAX_LINE	(256)

#define RHS_RANDOM	(0)
#define RHS_CONSTANT	(1)
#define RHS_FILE	(2)

#if LOCINT_SIZE == 8 
#define LOCINT_MAX_CHAR	(20)
#else
#define LOCINT_MAX_CHAR	(11)
#endif

#define RMAT_A	(0.57f)
#define RMAT_B	(0.19f)
#define RMAT_C	(0.19f)

// ~200 Mb buffer
// If smaller the hdd/ssd cache may have an 
// effect even with --no-rcache
#if LOCINT_SIZE == 8
#define IOINT_NUM	(1024*1024*10)
#else
#define IOINT_NUM	(1024*1024*10*2)
#endif
char IOBUF[IOINT_NUM*(LOCINT_MAX_CHAR+1)];

typedef struct {
	int	code;
	REAL	fp;
	char	*str;
} rhsv_t;

typedef struct {
	LOCINT	*u;
	LOCINT	*v;
	int64_t	ned;
} elist_t;

static int avoid_read_cache = 0;

static void usage(const char *pname) {
	
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	if (rank == 0) {
		const char *bname = rindex(pname, '/');
		if (!bname) bname = pname;
		else	    bname++;

		fprintf(stdout, 
			"Usage:\n"
			"%s [basic options] [advanced options]\n"
			"\n"
			"Basic options:\n"
			"\t-S <scale>\n"
			"\t--scale <scale>\n"
			"\t\tSpecifies the problem scale (2^scale vertices).\n"
			"\t\tDefault value: 21.\n"
			"\n"
			"\t-E <edge_factor>\n"
			"\t--edgef <edge_factor>\n"
			"\t\tSpecifies the average number of edges  per  vertex  (2^scale*edge_factor\n"
			"\t\ttotal edges).\n"
			"\t\tDefault value: 16.\n"
			"\n"
			"\t-i <niter>\n"
			"\t--iter <niter>\n"
			"\t\tSpecifies how  many  iterations  of  the  PageRank  algorithm  will  be\n"
			"\t\tperformed.\n"
			"\t\tDefault: 20.\n"
			"\n"
			"\t-k <kernel>\n"
			"\t--kstart <kernel>\n"
			"\t\tSpecifies from the kernel from which the benchmark will start (0, 1  or\n"
			"\t\t2).  Input files for the kernel must be present int the current working\n"
			"\t\tdirectory.  This option is useful to test a kernels on specific inputs.\n"
			"\t\tDefault: 0.\n"
			"\n"
			"\t-r <val>\n"
			"\t--rhs <val>\n"
			"\t\tSpecifies the type of r vector to be used for  the  PageRank  algorithm\n"
			"\t\t(kernel 3).  If this option is not specified the  vector  will	set  to\n"
			"\t\tnormalized random values (as required by  the  benchmark).   Otherwise,\n"
			"\t\ttype can be either a floating point number of  a  file	name.	In  the\n"
			"\t\tformer case all elements of r will be  set  to	\"val\"  while	in  the\n"
			"\t\tlatter	 the   values	 will	 be    read    from    file    \"val\".\n"
			"\t\tDefault: random.\n"
			"\n"
			"\t-c <damp_fact>\n"
			"\t--cfact <damp_fact>\n"
			"\t\tSpecifies the damping factor associated with  the  PageRank  algorithm.\n"
			"\t\tDefault: 0.15.\n"
			"\n"
			"\t-a <damp_vect>\n"
			"\t--avect <damp_vect>\n"
			"\t\tSpecifies the damping vector associated with  the  PageRank  algorithm.\n"
			"\t\tDefault: [1.0]*(1-damp_fact)/(2^scale).\n"
			"\n"
			"\t-f\n"
			"\t--fast\n"
			"\t\tEnables fast mode (skips disk I/O operations).\n"
			"\t\tDefault: false.\n"
			"\n"
			"\t-h\n"
			"\t--help\n"
			"\t\tShows this help.\n"
			"\n"
			"Advanced options:\n"
			"\t-s <seed>\n"
			"\t--seed <value>\n"
			"\t\tSpecify the value used to seed the generation of the kronecker graph.\n"
			"\t\tDefault: 0.\n"
			"\n"
			"\t--no-krcf\n"
			"\t\tDisables clip and flip on the generated graph.\n"
			"\t\tDefault: enabled.\n"
			"\n"
			"\t--no-krpm\n"
			"\t\tDisables vertex permutation on the generated graph.\n"
			"\t\tDefault: enabled.\n"
			"\n"
			"\t-g <file>\n"
			"\t--graph <file>\n"
			"\t\tThis option is used to perform the  benchmark  on  a  graph  read  from\n"
			"\t\t'file' rather than on a synthetic  one.   The  file  must  contain  one\n"
			"\t\tedge per line represented as a pair of separated numeric strings in the\n"
			"\t\trange [0:2^scale].  Edges do not need to be sorted.  Please  note  that\n"
			"\t\tthis  option  causes the -s,  --krcf  and  --krpm options to be ignored.\n"
			"\t\tMoreover it cannot be used together with -k specifying a kernel greater\n"
			"\t\tthan 0.\n"
			"\t\tDefault: disabled.\n",
			bname);
	}
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();
	exit(EXIT_SUCCESS);
}

static void prexit(const char *fmt, ...) {

        int rank;
        va_list ap;

        va_start(ap, fmt);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (0 == rank) vfprintf(stderr, fmt, ap);
	MPI_Barrier(MPI_COMM_WORLD);
        MPI_Finalize();
        exit(EXIT_SUCCESS);
}

static spmat_t *createSpmat(int n) {

	spmat_t *m = (spmat_t *)Malloc(sizeof(*m));

	m->firstRow = -1;
	m->lastRow = -1;
	m->intColsNum = 0;
	m->extColsNum = 0;
	m->totToSend = 0;

	m->sendNum = 0;
	m->recvNum = 0;

	m->sendNeigh = NULL;
	m->recvNeigh = NULL;
	m->sendCnts = NULL;
	m->recvCnts = NULL;
	m->sendOffs = NULL;
	m->recvOffs = NULL;

	m->rowsToSend_d = NULL;
#ifdef USE_MAPPED_SENDBUF
	m->sendBuffer_m = NULL;
#else
	m->sendBuffer_d = NULL;
#endif
	m->sendBuffer = NULL;
	m->recvBuffer = NULL;

	m->ncsr = n;
	m->nnz = (LOCINT *)Malloc(n*sizeof(LOCINT));
	m->nrows = (LOCINT *)Malloc(n*sizeof(LOCINT));

	m->roff_d = (LOCINT **)Malloc(n*sizeof(LOCINT *));
	m->rows_d = (LOCINT **)Malloc(n*sizeof(LOCINT *));
	m->cols_d = (LOCINT **)Malloc(n*sizeof(LOCINT *));
	m->vals_d = (REAL **)Malloc(n*sizeof(REAL *));

	memset(m->roff_d, 0, n*sizeof(LOCINT *));
	memset(m->rows_d, 0, n*sizeof(LOCINT *));
	memset(m->cols_d, 0, n*sizeof(LOCINT *));
	memset(m->vals_d, 0, n*sizeof(REAL *));

	m->kthr = (LOCINT (*)[2])Malloc(n*sizeof(LOCINT[2]));
	m->koff = (LOCINT (*)[2])Malloc(n*sizeof(LOCINT[2]));

	return m;
}

static void destroySpmat(spmat_t *m) {

	if (m->sendNeigh) free(m->sendNeigh);
	if (m->recvNeigh) free(m->recvNeigh);
	if (m->sendCnts) free(m->sendCnts);
	if (m->recvCnts) free(m->recvCnts);
	if (m->sendOffs) free(m->sendOffs);
	if (m->recvOffs) free(m->recvOffs);

	if (m->rowsToSend_d) CHECK_CUDA(cudaFree(m->rowsToSend_d));
#ifndef USE_MAPPED_SENDBUF
	if (m->sendBuffer_d) CHECK_CUDA(cudaFree(m->sendBuffer_d));
#endif
	if (m->sendBuffer) {
		CHECK_CUDA(cudaHostUnregister(m->sendBuffer));
		free(m->sendBuffer);
	}
	if (m->recvBuffer) {
		CHECK_CUDA(cudaHostUnregister(m->recvBuffer));
		free(m->recvBuffer);
	}

	for(int i = 0; i < m->ncsr; i++) {
		if (m->roff_d[i]) CHECK_CUDA(cudaFree(m->roff_d[i]));
		if (m->rows_d[i]) CHECK_CUDA(cudaFree(m->rows_d[i]));
		if (m->cols_d[i]) CHECK_CUDA(cudaFree(m->cols_d[i]));
		if (m->vals_d[i]) CHECK_CUDA(cudaFree(m->vals_d[i]));
	}
	free(m->roff_d);
	free(m->rows_d);
	free(m->cols_d);
	free(m->vals_d);

	free(m->kthr);
	free(m->koff);

	return;
}

static int64_t parallelReadGraph(const char *fpath, int scale, LOCINT **upptr, LOCINT **vpptr) {

	LOCINT	*u=NULL, *v=NULL;
	LOCINT	i, j;
	LOCINT	N = ((LOCINT)1) << scale;
	int64_t	n, nmax;
	size_t  size;
	int64_t off1, off2;

	int      rem;
	FILE     *fp;
	char     str[MAX_LINE];

	int rank, ntask;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &ntask);

	size = getFsize(fpath);
	rem = size % ntask;
	off1 = (size/ntask)* rank    + (( rank    > rem)?rem: rank);
	off2 = (size/ntask)*(rank+1) + (((rank+1) > rem)?rem:(rank+1));

	fp = Fopen(fpath, "r");
	if (rank < (ntask-1)) {
		fseek(fp, off2, SEEK_SET);
		fgets(str, MAX_LINE, fp);
		off2 = ftell(fp);
	}
	fseek(fp, off1, SEEK_SET);
	if (rank > 0) {
		fgets(str, MAX_LINE, fp);
		off1 = ftell(fp);
	}

	//fprintf(stdout, "Process %d off1=%lld off2=%lld size=%lld\n",
	//      rank, off1, off2, off2-off1);

	n = 0;
	nmax = CHUNK_SIZE; // must be even
	u = (LOCINT *)Malloc(nmax*sizeof(*u));
	v = (LOCINT *)Malloc(nmax*sizeof(*v));

	/* read edges from file */
	while (ftell(fp) < off2) {

		fgets(str, MAX_LINE, fp);

		char *ptr;
		for(ptr = str; (*ptr == ' ') || (*ptr == '\t'); ptr++);
		if (ptr[0] == '#') continue;

		sscanf(str, "%"PRILOC" %"PRILOC"\n", &i, &j);

		if (i >= N || j >= N) {
			fprintf(stderr,
				"[%d] found invalid edge in %s for N=%"PRILOC": (%"PRILOC", %"PRILOC")\n",
				rank, fpath, N, i, j);
			exit(EXIT_FAILURE);
		}

		if (n >= nmax) {
			nmax += CHUNK_SIZE;
			u = (LOCINT *)Realloc(u, nmax*sizeof(*u));
			v = (LOCINT *)Realloc(v, nmax*sizeof(*v));
		}
		u[n] = i;
		v[n] = j;
		n++;
	}
	fclose(fp);

	*upptr = u;
	*vpptr = v;

	return n;
}

static inline size_t bin2str(LOCINT v, char *s) {

	char *e = s;
	do {
		*e++ = '0' + (v%10);
		v /= 10;
	} while(v);
	size_t len=e-s;
	for(e--; s < e;) {
		char t = *s;
		*s++ = *e;
		*e-- = t;
	}
	return len;
}

static int64_t fwriteGraphAOS(LOCINT *ed, int64_t ned, const char *fname) {

	int64_t	i, c;
	size_t	off=0;
	FILE	*fp = Fopen(fname, "w");

	for(i=0, c=0; i < 2*ned; i++, c++) {
		if (c == IOINT_NUM) {
			fwrite(IOBUF, off, 1, fp);
			c = off = 0;
		}
		off += bin2str(ed[i], IOBUF+off);
		IOBUF[off++] = (i&1) ? '\n' : '\t';
	}
	if (off) fwrite(IOBUF, off, 1, fp);

	/* if we must wait for data to actually hit the disk */
	if (fflush(fp) != 0) {
		fprintf(stderr, "Error while executing fflush()!\n");
		exit(EXIT_FAILURE);
	}
	if (fsync(fileno(fp)) < 0) {
		fprintf(stderr, "Error while executing fflush()!\n");
		exit(EXIT_FAILURE);
	}

	i = ftell(fp);
	fclose(fp);

	return i;
}

static size_t fwriteGraphSOA(LOCINT *u, LOCINT *v, int64_t ned, const char *fname, double *wtime) {

	int64_t	i, c;
	size_t	off=0;

	double t = MPI_Wtime();
	FILE *fp = Fopen(fname, "w");
	*wtime = MPI_Wtime() - t;
	
	for(i=0, c=0; i < ned; i++, c+=2) {
		if (c >= IOINT_NUM-1) {

			t = MPI_Wtime();
			fwrite(IOBUF, off, 1, fp);
			*wtime += MPI_Wtime() - t;

			c = off = 0;
		}
		off += bin2str(u[i], IOBUF+off);
		IOBUF[off++] = '\t';
		off += bin2str(v[i], IOBUF+off);
		IOBUF[off++] = '\n';
	}
	t = MPI_Wtime();
	if (off) fwrite(IOBUF, off, 1, fp);
	
	/* if we must wait for data to actually hit the disk */
	if (fflush(fp) != 0) {
		fprintf(stderr, "Error while executing fflush()!\n");
		exit(EXIT_FAILURE);
	}
	if (fsync(fileno(fp)) < 0) {
		fprintf(stderr, "Error while executing fflush()!\n");
		exit(EXIT_FAILURE);
	}

	off = ftell(fp);
	fclose(fp);
	*wtime += MPI_Wtime() - t;

	return off;
}

static size_t fwriteGraphSOA_GCONV(LOCINT *d_u, LOCINT *d_v, int64_t ned, const char *fname, double *wtime) {

	FILE *fp;
	char *h_data=NULL;
	char *d_data=NULL;
	size_t fsize;

	if (!ned) return 0;

	fsize = BinCouple2ASCIICuda(d_u, d_v, ned, &d_data, 0);

	h_data = (char *)Malloc(fsize);
	CHECK_CUDA(cudaMemcpy(h_data, d_data, fsize, cudaMemcpyDeviceToHost));

	*wtime = MPI_Wtime();
	fp = Fopen(fname, "w");
	Fwrite(h_data, 1, fsize, fp);
	
	/* if we must wait for data to actually hit the disk */
	if (fflush(fp) != 0) {
		fprintf(stderr, "Error while executing fflush()!\n");
		exit(EXIT_FAILURE);
	}
	if (fsync(fileno(fp)) < 0) {
		fprintf(stderr, "Error while executing fflush()!\n");
		exit(EXIT_FAILURE);
	}
	fclose(fp);
	*wtime = MPI_Wtime() - *wtime;

	free(h_data);	
	CHECK_CUDA(cudaFree(d_data));

	return fsize;
}

static inline uint8_t str2bin(char *s, LOCINT *v) {

	char	*p;
	LOCINT	ex=1;
	uint8_t len=0;

	for(; *s < '0' || *s > '9'; s++, len++);
	for(p=s; *p >= '0' && *p <= '9'; p++);

	len += p-s;

	*v = 0;
	while(p > s) {
		*v += ex*(*--p - '0');
		ex *= 10;
	}
	return len;
}

static int64_t freadGraphSOA(const char *fname, int scale, int edgef, LOCINT **uout, LOCINT **vout, int64_t *ned, double *rtime) {

	int	rank;
	LOCINT	*uv[2]={NULL,NULL};
	LOCINT	N = ((LOCINT)1) << scale;
	int64_t	nalloc, rbyte, rem, i, edge_max;
	double	t;

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	t = MPI_Wtime();
	FILE *fp = Fopen(fname, "r");
	*rtime = MPI_Wtime()-t;

	i = 0;
	rem = 0;
	nalloc = 0;
	edge_max = ((int64_t)N)*edgef;
#ifndef __APPLE__
	if (avoid_read_cache) {
		int fd = fileno(fp);
		posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
	}
#endif
	while(1) {

		char *q = IOBUF + rem;

		// read at most 2^scale*edgef edges
		if (i/2 >= edge_max) break;

		t = MPI_Wtime();
		rbyte = fread(q, 1, sizeof(IOBUF)-(q-IOBUF), fp);
		*rtime += MPI_Wtime()-t;

		if (!rbyte) break;

		for(q += rbyte-1, rem = 0;
		    *q >= '0' && *q <= '9';
		    q--, rem++);

		char *p = IOBUF;
		while(p < q) {
			LOCINT val;
			p += str2bin(p, &val);
			if (val >= N) {
				fprintf(stderr,
					"[%d] edge in file %s contains a vertex=%"PRILOC" >= 2^scale=%"PRILOC"\n",
					rank, fname, val, N);
				exit(EXIT_FAILURE);
			}
			if (i/2 == nalloc) {
				uv[0] = (LOCINT *)Realloc(uv[0], (i+CHUNK_SIZE)*sizeof(*uv[0]));
				uv[1] = (LOCINT *)Realloc(uv[1], (i+CHUNK_SIZE)*sizeof(*uv[1]));
				nalloc += CHUNK_SIZE;
			}
			uv[i&1][i/2] = val;
			i++;
			
		}
		//if (rem) memcpy(IOBUF, q+1, rem); 
		for(int i = 0; i < rem; i++) IOBUF[i] = q[1+i];
	}
	if (i&1) {
		fprintf(stderr,"[%d] incomplete egde in file %s!\n", rank, fname);
		exit(EXIT_FAILURE);
	}

	int64_t pos = ftell(fp);

	t = MPI_Wtime();
	fclose(fp);
	*rtime += MPI_Wtime()-t;

	*uout = uv[0];
	*vout = uv[1];
	*ned = i/2;

	return pos;
}

static size_t freadGraphSOA_GCONV(const char *fname, LOCINT **h_u, LOCINT **h_v, int64_t *ned, double *rtime) {

	FILE	*fp=NULL;
	char	*h_data=NULL;
	size_t	fsize;

	fsize = getFsize(fname);
	h_data = (char *)Malloc(fsize);

	*rtime = MPI_Wtime();
	fp = Fopen(fname, "r");
#ifndef __APPLE__
	if (avoid_read_cache) {
		int fd = fileno(fp);
		posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
	}
#endif
	Fread(h_data, 1, fsize, fp);
	fclose(fp);
	*rtime = MPI_Wtime() - *rtime;

	// +1 because last row may not end with a newline
	int nchunks = (fsize+1 + sizeof(uint4)-1) / sizeof(uint4);

	uint4 *d_data;
	CHECK_CUDA(cudaMalloc(&d_data, nchunks*sizeof(*d_data)));
	CHECK_CUDA(cudaMemcpy(d_data, h_data, fsize, cudaMemcpyHostToDevice));

	LOCINT *d_u=NULL, *d_v=NULL;
	*ned = ASCIICouple2BinCuda_entry(d_data, fsize, &d_u, &d_v, 0);

	h_u[0] = (LOCINT *)Malloc(ned[0]*sizeof(**h_u));
	h_v[0] = (LOCINT *)Malloc(ned[0]*sizeof(**h_v));
	CHECK_CUDA(cudaMemcpy(h_u[0], d_u, ned[0]*sizeof(**h_u), cudaMemcpyDeviceToHost));
	CHECK_CUDA(cudaMemcpy(h_v[0], d_v, ned[0]*sizeof(**h_v), cudaMemcpyDeviceToHost));

	free(h_data);
	CHECK_CUDA(cudaFree(d_u));
	CHECK_CUDA(cudaFree(d_v));
	CHECK_CUDA(cudaFree(d_data));

	return fsize;
}

// quinck and dirty general type vector reader from file
static LOCINT freadArray(const char *fname, const char *typefmt, const int typesz, void *v, const LOCINT n) {

	LOCINT	i=0;
	char	buf[MAX_LINE];
	FILE	*fp  = Fopen(fname, "r");

	while(1) {
		char *ptr = fgets(buf, MAX_LINE, fp);
		if (!ptr) break;
		sscanf(ptr, typefmt, ((char *)v)+i*typesz);
		i++;
	}
	fclose(fp);
	return i;
}

void fwriteArrayDbl(const char *fprefix, double *a, int64_t n) {

	FILE	*fp;
	int	rank;
	char	fname[256];

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	snprintf(fname, 256, "%s_%d.txt", fprefix, rank);
	
	fp = Fopen(fname, "w");
	for(int64_t i = 0; i < n; i++)
		fprintf(fp, "%E\n", a[i]);

	fclose(fp);
	return;
}

void fwriteArrayInt(const char *fprefix, LOCINT *a, int64_t n) {

	FILE	*fp;
	int	rank;
	char	fname[256];

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	snprintf(fname, 256, "%s_%d.txt", fprefix, rank);
	
	fp = Fopen(fname, "w");
	for(int64_t i = 0; i < n; i++)
		fprintf(fp, "%"PRILOC"\n", a[i]);

	fclose(fp);
	return;
}

static void check_row_overlap(LOCINT first_row, LOCINT last_row, int *exchup, int *exchdown) {

	int		rank, ntask, nr;
	LOCINT 		prevrow, nextrow;
	MPI_Request	request[2];
	
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &ntask);

	nr = 0;
	prevrow = nextrow = -1;

	if (rank > 0)	    MPI_Irecv(&prevrow, 1, LOCINT_MPI, rank-1, TAG(rank-1), MPI_COMM_WORLD, &request[nr++]);
	if (rank < ntask-1) MPI_Irecv(&nextrow, 1, LOCINT_MPI, rank+1, TAG(rank+1), MPI_COMM_WORLD, &request[nr++]);

	if (rank < ntask-1) MPI_Send(&last_row,  1, LOCINT_MPI, rank+1, TAG(rank), MPI_COMM_WORLD);
	if (rank > 0)	    MPI_Send(&first_row, 1, LOCINT_MPI, rank-1, TAG(rank), MPI_COMM_WORLD);

        MPI_Waitall(nr, request, MPI_STATUS_IGNORE);

	*exchup   = (prevrow == first_row);
	*exchdown = (nextrow == last_row);

	return;
}
	
void adjust_row_range(int scale, LOCINT *first_row, LOCINT *last_row) {

	int	rank, ntask;
	
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &ntask);

	MPI_Sendrecv(first_row, 1, LOCINT_MPI, (rank+ntask-1)%ntask, rank,
		     last_row, 1, LOCINT_MPI, (rank+1)%ntask, (rank+1)%ntask,
    		     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	*last_row = ((rank < ntask-1) ? *last_row : ((LOCINT)1)<<scale) -1;
	if (rank == 0) *first_row = 0;

	return;	
}

static void postIrecvs(REAL *recvBuffer, int recvNum, int *recvNeigh, int64_t *recvOffs, int64_t *recvCnts,
		       MPI_Request *request, MPI_Comm COMM) {

	int ntask;
	MPI_Comm_size(COMM, &ntask);

	for(int i = 0; i < recvNum; i++) {
		if (recvCnts[i]) {
			MPI_Irecv(recvBuffer + recvOffs[i], recvCnts[i],
				  REAL_MPI, recvNeigh[i], TAG(recvNeigh[i]),
				  COMM, request + i);
		}
	}
	return;
}

static void exchangeDataSingle(REAL *sendBuffer, int sendNeigh, int64_t sendCnts,
			       REAL *recvBuffer, int recvNeigh, int64_t recvCnts,
			       MPI_Request *request, MPI_Comm COMM) {

	int rank, ntask;
	
	MPI_Comm_rank(COMM, &rank);
	MPI_Comm_size(COMM, &ntask);

	if (sendCnts) {
		MPI_Send(sendBuffer, sendCnts,
			 REAL_MPI, sendNeigh, TAG(rank),
			 COMM);
	}
	MPI_Wait(request, MPI_STATUS_IGNORE);
	if (recvCnts) {
		MPI_Irecv(recvBuffer, recvCnts,
			  REAL_MPI, recvNeigh, TAG(recvNeigh),
			  COMM, request);
	}
	return;
}

static inline void cancelReqs(MPI_Request *request, int n) {

        int i;
        for(i = 0; i < n; i++) {
		int flag;
		MPI_Test(request+i, &flag, MPI_STATUS_IGNORE);
		if (!flag) {
			MPI_Cancel(request+i);
			MPI_Wait(request+i, MPI_STATUSES_IGNORE);
		}
	}
        return;
}

static void kernel0(int scale, int edgef, int64_t gseed, int krcf, int krpm, const char *fin, elist_t *eout) {

	double	tg=0, tw=0, tdw=0;
	double	min_wt, max_wt;
	double	min_wr, max_wr;

	int	rank, ntask;
	LOCINT	*u=NULL, *u_m=NULL;
	LOCINT	*v=NULL, *v_m=NULL;

	int64_t	ned;
	int64_t	N=((uint64_t)1)<<scale;
	
	size_t	wbytes=0;

	char	fname[MAX_LINE];

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &ntask);

	ned = N*edgef;
	ned = ned/ntask + (rank < (ned%ntask));

	if (fin) {
		ned = parallelReadGraph(fin, scale, &u, &v);
#if defined(USE_DEV_IOCONV_WRITE)
		CHECK_CUDA(cudaMalloc(&u_m, ned*sizeof(*u_m)));
		CHECK_CUDA(cudaMalloc(&v_m, ned*sizeof(*u_m)));
		CHECK_CUDA(cudaMemcpy(u_m, u, ned*sizeof(*u_m), cudaMemcpyHostToDevice));
		CHECK_CUDA(cudaMemcpy(v_m, v, ned*sizeof(*v_m), cudaMemcpyHostToDevice));
#endif
	} else {
		u = (LOCINT *)Malloc(ned*sizeof(*u));
		v = (LOCINT *)Malloc(ned*sizeof(*v));
#if defined(USE_DEV_IOCONV_WRITE)
		CHECK_CUDA(cudaMalloc(&u_m, ned*sizeof(*u_m)));
		CHECK_CUDA(cudaMalloc(&v_m, ned*sizeof(*u_m)));
#else
		CHECK_CUDA(cudaHostRegister(u, ned*sizeof(*u), cudaHostRegisterMapped));
		CHECK_CUDA(cudaHostRegister(v, ned*sizeof(*v), cudaHostRegisterMapped));
		CHECK_CUDA(cudaHostGetDevicePointer((void **)&u_m, u, 0) );
		CHECK_CUDA(cudaHostGetDevicePointer((void **)&v_m, v, 0) );
#endif
		MPI_Barrier(MPI_COMM_WORLD);
		tg = MPI_Wtime();
		generate_kron(scale, ned, RMAT_A, RMAT_B, RMAT_C, u_m, v_m, gseed+rank, krcf, krpm);
		MPI_Barrier(MPI_COMM_WORLD);
		tg = MPI_Wtime()-tg;
	}

	if (!eout) {
		snprintf(fname, MAX_LINE, "unsorted_%d.txt", rank);
		Remove(fname);

		MPI_Barrier(MPI_COMM_WORLD);
		tw = MPI_Wtime();
#if !defined(USE_DEV_IOCONV_WRITE)
		wbytes = fwriteGraphSOA(u, v, ned, fname, &tdw);
#else
		wbytes = fwriteGraphSOA_GCONV(u_m, v_m, ned, fname, &tdw);
#endif
		MPI_Barrier(MPI_COMM_WORLD);
		tw = MPI_Wtime() - tw;
		
		MPI_Reduce(&tdw, &min_wt, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
		MPI_Reduce(&tdw, &max_wt, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

		double wr = wbytes/(1024.0*1024.0)/tdw;
		MPI_Reduce(&wr, &min_wr, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
		MPI_Reduce(&wr, &max_wr, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
		
		MPI_Reduce(rank?&wbytes:MPI_IN_PLACE, &wbytes, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
	}

	if (0 == rank && (!fin || !eout)) {
		printf("Kernel 0 (scale=%d, edgef=%d):\n", scale, edgef);
		if (!fin) printf("\tgen   time: %.4lf secs\n", tg);
		if (!eout) {
			printf("\twrite time: %.4lf secs, %.2lf Mbytes/sec\n", tw, wbytes/(1024.0*1024.0)/tw);
			printf("\t\tmin/max disk write time: %.4lf/%.4lf secs, %.2lf/%.2lf Mbytes/sec\n", min_wt, max_wt, min_wr, max_wr);
		}
		printf("\tTOTAL time: %.4lf secs (I/O: %.0lf%%), %.2lf Medges/sec\n\n",
		       tg+tw, 100.0*tw/(tg+tw), 1E-6*(((LOCINT)1) << scale)*edgef/(tg+tw));
	}

	if (!eout) {
		free(u);
		free(v);
	} else {
#if defined(USE_DEV_IOCONV_WRITE)
		CHECK_CUDA(cudaMemcpy(u, u_m, ned*sizeof(*u), cudaMemcpyDeviceToHost));
		CHECK_CUDA(cudaMemcpy(v, v_m, ned*sizeof(*v), cudaMemcpyDeviceToHost));
#endif
		eout->u = u;
		eout->v = v;
		eout->ned = ned;
	}

#if defined(USE_DEV_IOCONV_WRITE)
	CHECK_CUDA(cudaFree(u_m));
	CHECK_CUDA(cudaFree(v_m));
#else
	if (!fin) {
		CHECK_CUDA(cudaHostUnregister(u));
		CHECK_CUDA(cudaHostUnregister(v));
	}
#endif
	return;
}

static void kernel1(int scale, int edgef, double rtol, elist_t *eio, int transpose) {

	double	tr=0, ts=0, tw=0, tdr=0, tdw=0;
	
	double	min_rt, max_rt;
	double	min_rr, max_rr;
	
	double	min_wt, max_wt;
	double	min_wr, max_wr;

	int64_t ned, rbytes=0;
	size_t	wbytes=0;
	int	rank, ntask;
	LOCINT	*u=NULL, *v=NULL;
	char	fname[MAX_LINE];

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &ntask);

	if (!eio) {
		snprintf(fname, MAX_LINE, "unsorted_%d.txt", rank);

		MPI_Barrier(MPI_COMM_WORLD);
		tr = MPI_Wtime();
#if !defined(USE_DEV_IOCONV_READ)
		rbytes = freadGraphSOA(fname, scale, edgef, &u, &v, &ned, &tdr);
#else
		rbytes = freadGraphSOA_GCONV(fname, &u, &v, &ned, &tdr);
#endif
		MPI_Barrier(MPI_COMM_WORLD);
		tr = MPI_Wtime()-tr;
	} else {
		u = eio->u;
		v = eio->v;
		ned = eio->ned;
	}

	ts = MPI_Wtime();

	if (transpose) {
		LOCINT *tmp = u;
		u = v;
		v = tmp;
	}

	// simple parallel histogram-sort implementation
	// returns u[] amnd v[] sorted first across u and
	// then v
	phsort(&u, &v, &ned, 1.0E-2, 0);
	MPI_Barrier(MPI_COMM_WORLD);
	ts = MPI_Wtime()-ts;

	if (!eio) {
		snprintf(fname, MAX_LINE, "sorted_%d.txt", rank);
		Remove(fname);
#if defined(USE_DEV_IOCONV_WRITE)
		LOCINT  *u_d=NULL;
		LOCINT  *v_d=NULL;
		CHECK_CUDA(cudaMalloc(&u_d, ned*sizeof(*u_d)));
		CHECK_CUDA(cudaMalloc(&v_d, ned*sizeof(*u_d)));
#endif	
		MPI_Barrier(MPI_COMM_WORLD);
		tw = MPI_Wtime();
#if !defined(USE_DEV_IOCONV_WRITE)
		wbytes = fwriteGraphSOA(u, v, ned, fname, &tdw);
#else
		CHECK_CUDA(cudaMemcpy(u_d, u, ned*sizeof(*u_d), cudaMemcpyHostToDevice));
		CHECK_CUDA(cudaMemcpy(v_d, v, ned*sizeof(*v_d), cudaMemcpyHostToDevice));
		wbytes = fwriteGraphSOA_GCONV(u_d, v_d, ned, fname, &tdw);
#endif
		MPI_Barrier(MPI_COMM_WORLD);
		tw = MPI_Wtime()-tw;

		MPI_Reduce(&tdr, &min_rt, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
		MPI_Reduce(&tdr, &max_rt, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

		double rr = rbytes/(1024.0*1024.0)/tdr;
		MPI_Reduce(&rr, &min_rr, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
		MPI_Reduce(&rr, &max_rr, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

		MPI_Reduce(rank ? &rbytes : MPI_IN_PLACE, &rbytes, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

		MPI_Reduce(&tdw, &min_wt, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
		MPI_Reduce(&tdw, &max_wt, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

		double wr = wbytes/(1024.0*1024.0)/tdw;
		MPI_Reduce(&wr, &min_wr, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
		MPI_Reduce(&wr, &max_wr, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

		MPI_Reduce(rank ? &wbytes : MPI_IN_PLACE, &wbytes, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
#if defined(USE_DEV_IOCONV_WRITE)
		CHECK_CUDA(cudaFree(u_d));
		CHECK_CUDA(cudaFree(v_d));
#endif		
		free(u);
		free(v);
	} else {
		eio->u = u;
		eio->v = v;
		eio->ned = ned;
	}

	MPI_Reduce(rank ? &ned : MPI_IN_PLACE, &ned, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

	if (0 == rank) {
		printf("Kernel 1 (scale=%d, edgef=%d):\n", scale, edgef);
		if (!eio) {
			printf("\tread  time: %.4lf secs, %.2lf Mbytes/sec\n", tr, rbytes/(1024.0*1024.0)/tr);
			printf("\t\tmin/max disk read time: %.4lf/%.4lf secs, %.2lf/%.2lf Mbytes/sec\n", min_rt, max_rt, min_rr, max_rr);
		}
		printf("\tsort  time: %.4lf secs\n", ts);
		if (!eio) {
			printf("\twrite time: %.4lf secs, %.2lf Mbytes/sec\n", tw, wbytes/(1024.0*1024.0)/tw);
			printf("\t\tmin/max disk write time: %.4lf/%.4lf secs, %.2lf/%.2lf Mbytes/sec\n", min_wt, max_wt, min_wr, max_wr);
		}
		printf("\tTOTAL time: %.4lf secs (I/O: %.0lf%%), %.2lf Medges/sec\n\n",
		       (tr+ts+tw), 100.0*(tr+tw)/(tr+ts+tw), 1E-6*ned/(tr+ts+tw));
	}
	return;
}

static void kernel2_multi(int scale, int edgef, spmat_t *m, elist_t *ein) {

	double tr=0, tg=0, tdr=0;

	double	min_rt, max_rt;
	double	min_rr, max_rr;
	
	int64_t ned, rbytes=0;

	LOCINT	*u=NULL,   *v=NULL;

	int	EXCH_UP, EXCH_DOWN;
	int	rank, ntask;

	LOCINT	*lastrow_all=NULL;
	LOCINT	*rowsToSend=NULL;

	char	fname[MAX_LINE];

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &ntask);

	if (!ein) {
		snprintf(fname, MAX_LINE, "sorted_%d.txt", rank);

		MPI_Barrier(MPI_COMM_WORLD);
		tr = MPI_Wtime();
#if !defined(USE_DEV_IOCONV_READ)
		rbytes = freadGraphSOA(fname, scale, edgef, &u, &v, &ned, &tdr);
#else
		rbytes = freadGraphSOA_GCONV(fname, &u, &v, &ned, &tdr);
#endif
		MPI_Barrier(MPI_COMM_WORLD);
		tr = MPI_Wtime()-tr;
	} else {
		u = ein->u;
		v = ein->v;
		ned = ein->ned;
	}

	m->firstRow = u[0];
	m->lastRow  = u[ned-1];

	// quick sanity check on (sorted) edges received as input
	check_row_overlap(m->firstRow, m->lastRow, &EXCH_UP, &EXCH_DOWN);
	if (EXCH_UP || EXCH_DOWN) {
		fprintf(stderr, "Processor %d shares rows with neighboring processors!!!\n", rank);
		exit(EXIT_FAILURE);
	}

	//cudaProfilerStart();
	tg = MPI_Wtime();

	LOCINT	*u_d=NULL, *v_d=NULL; // alloc-ed in remove_rows_cuda() and
				      // dealloc-ed in get_csr_multi_cuda()
	ned = remove_rows_cuda(u, v, ned, &u_d, &v_d);
	CHECK_CUDA(cudaMemcpy(&m->firstRow, u_d, sizeof(m->firstRow), cudaMemcpyDeviceToHost));

	// quick sanity check
	check_row_overlap(m->firstRow, m->lastRow, &EXCH_UP, &EXCH_DOWN);
	if (EXCH_UP || EXCH_DOWN) {
		fprintf(stderr, "Processor %d shares rows with neighboring processors!!!\n", rank);
		exit(EXIT_FAILURE);
	}
	// expand [m->firstRow, m->lastRow] ranges in order to partition [0, N-1]
	// (empty rows outside any [m->firstRow, m->lastRow] range may appear as
	// columns in other rows
	adjust_row_range(scale, &m->firstRow, &m->lastRow);
	m->intColsNum = m->lastRow - m->firstRow + 1;

	lastrow_all = (LOCINT *)Malloc(ntask*sizeof(*lastrow_all));
	MPI_Allgather(&m->lastRow, 1, LOCINT_MPI, lastrow_all, 1, LOCINT_MPI, MPI_COMM_WORLD);

	get_csr_multi_cuda(u_d, v_d, ned, lastrow_all, ntask,
			   m->nnz, m->nrows, m->roff_d,
			   m->rows_d, m->cols_d, m->vals_d);
	normalize_cols_multi(m->nnz, m->cols_d, m->vals_d, lastrow_all, m->ncsr, MPI_COMM_WORLD);
	for(int i = 0; i < m->ncsr; i++) {
#if 1
		m->kthr[i][0] = 1024;
		m->kthr[i][1] = 16; // good values: 7, 16
		if (m->kthr[i][1] > m->kthr[i][0]) {
			fprintf(stderr, "[%d] Error with thresholds!!!\n", rank);
			exit(EXIT_FAILURE);
		}
		// sort CSR rows in descending order of length
		sort_csr(m->nnz[i], m->nrows[i], m->kthr[i],
			 m->rows_d+i, m->roff_d+i, m->cols_d+i,
			 m->vals_d+i, m->koff[i]);
#else
		m->koff[i][0] = 0;
		m->koff[i][1] = m->nrows[i];
#endif
	}
	get_extdata_cuda(m->ncsr, m->nnz, m->cols_d, lastrow_all,
			 &m->recvNeigh, &m->recvNum, &m->recvCnts, &m->recvOffs, &m->extColsNum, 
			 &m->sendNeigh, &m->sendNum, &m->sendCnts, &m->sendOffs, &m->totToSend,
			 &rowsToSend, MPI_COMM_WORLD);
	if (m->extColsNum) {
		m->recvBuffer = (REAL *)Malloc(m->extColsNum*sizeof(*m->recvBuffer));
		CHECK_CUDA(cudaHostRegister(m->recvBuffer, m->extColsNum*sizeof(*m->recvBuffer), cudaHostRegisterMapped));
	}
	if (m->sendNum) {
		m->totToSend = m->sendOffs[m->sendNum-1] + m->sendCnts[m->sendNum-1]; // redundant
		m->sendBuffer = (REAL *)Malloc(m->totToSend*sizeof(*m->sendBuffer));
		CHECK_CUDA(cudaHostRegister(m->sendBuffer, m->totToSend*sizeof(*m->sendBuffer), cudaHostRegisterMapped));
#ifdef USE_MAPPED_SENDBUF
		CHECK_CUDA(cudaHostGetDevicePointer((void **)&(m->sendBuffer_m), m->sendBuffer, 0) );
#else
		CHECK_CUDA(cudaMalloc(&m->sendBuffer_d, m->totToSend*sizeof(*m->sendBuffer_d)));
#endif
		CHECK_CUDA(cudaMalloc(&m->rowsToSend_d, m->totToSend*sizeof(*m->rowsToSend_d)));
		CHECK_CUDA(cudaMemcpy(m->rowsToSend_d, rowsToSend, m->totToSend*sizeof(*m->rowsToSend_d), cudaMemcpyHostToDevice));
	}

	relabel_cuda_multi(lastrow_all,
			   m->ncsr,
			   (LOCINT *)m->nrows, m->rows_d,
			   (LOCINT *)m->nnz, m->cols_d,
			   m->totToSend, m->rowsToSend_d, MPI_COMM_WORLD);
	tg = MPI_Wtime()-tg;

	if (!ein) {
		MPI_Reduce(&tdr, &min_rt, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
		MPI_Reduce(&tdr, &max_rt, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

		double rr = rbytes/(1024.0*1024.0)/tdr;
		MPI_Reduce(&rr, &min_rr, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
		MPI_Reduce(&rr, &max_rr, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

		MPI_Reduce(rank ? &rbytes : MPI_IN_PLACE, &rbytes, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
	}
	//cudaProfilerStop();
	if (0 == rank) {
		printf("Kernel 2 (scale=%d, edgef=%d):\n", scale, edgef); 
		if (!ein) {
			printf("\tread  time: %.4lf secs, %.2lf Mbytes/sec\n", tr, rbytes/(1024.0*1024.0)/tr);
			printf("\t\tmin/max disk read time: %.4lf/%.4lf secs, %.2lf/%.2lf Mbytes/sec\n", min_rt, max_rt, min_rr, max_rr);
		}
		printf("\tgen   time: %.4lf secs\n", tg);
		printf("\tTOTAL time: %.4lf secs (I/O: %.0lf%%), %.2lf Medges/sec\n\n",
		       (tr+tg), 100.0*tr/(tr+tg), 1E-6*(((LOCINT)1) << scale)*edgef/(tr+tg));
		fflush(stdout);
	}

	// sanity check (untimed)
	if (m->totToSend) {
		CHECK_CUDA(cudaMemcpy(rowsToSend, m->rowsToSend_d, m->totToSend*sizeof(*rowsToSend), cudaMemcpyDeviceToHost));
		for(int i = 0; i < m->sendNum; i++) {
			for(int j = 0; j < m->sendCnts[i]; j++) {
				if (rowsToSend[m->sendOffs[i]+j] < 0 || rowsToSend[m->sendOffs[i]+j] > m->intColsNum) {
					fprintf(stderr, "[%d] error: rowsToSend[%"PRId64"] (%d-th row to send to proc %d) = %"PRILOC" > %"PRId64"\n",
						rank, m->sendOffs[i]+j, j, m->sendNeigh[i], rowsToSend[m->sendOffs[i]+j], m->intColsNum);
					exit(EXIT_FAILURE);
				}
			}
		}
	}
	if (rowsToSend) free(rowsToSend);
	if (lastrow_all) free(lastrow_all);
	if (!ein) {
		free(u);
		free(v);
	}
	return;
}

static void kernel3_multi(int scale, int edgef, int numIter, REAL c, REAL a, rhsv_t rval, spmat_t *m) {

	int		i, rank, ntask;
	float		evt;
	double		tg=0, tc=0, t=0;
	double		tspmv[2]={0,0}, tmpi[2]={0,0}, td2h[2]={0,0}, th2d[2]={0,0};
	REAL		*r_h=NULL, *r_d[2]={NULL,NULL}, sum;
	LOCINT		N = ((LOCINT)1) << scale;
	MPI_Request	*reqs=NULL;

	cudaStream_t	stream[2];
	cudaEvent_t	event[4];

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &ntask);

	CHECK_CUDA(cudaStreamCreate(stream));
	CHECK_CUDA(cudaStreamCreate(stream+1));

	CHECK_CUDA(cudaEventCreate(event));
	CHECK_CUDA(cudaEventCreate(event+1));
	CHECK_CUDA(cudaEventCreate(event+2));
	CHECK_CUDA(cudaEventCreate(event+3));

	CHECK_CUDA(cudaMalloc(r_d+0, (m->intColsNum + MAX(1,m->extColsNum))*sizeof(*r_d[0])));
	CHECK_CUDA(cudaMalloc(r_d+1, (m->intColsNum + MAX(1,m->extColsNum))*sizeof(*r_d[1])));

	if (m->recvNum)
		reqs = (MPI_Request *)Malloc(m->recvNum*sizeof(MPI_Request));

	MPI_Barrier(MPI_COMM_WORLD);
	tg = MPI_Wtime();

	// generate RHS
	if (rval.code == RHS_RANDOM) {
		generate_rhs(m->intColsNum, r_d[0]);
	} else if (rval.code == RHS_CONSTANT) {
		setarray(r_d[0], m->intColsNum, rval.fp); 
		CHECK_CUDA(cudaDeviceSynchronize());
	} else {
		r_h = (REAL *)Malloc(N*sizeof(*r_h));
		LOCINT nread = freadArray(rval.str, REAL_SPEC"\n", sizeof(REAL), (void *)r_h, N);
		if (nread != N) {
			fprintf(stderr, "[%d] found %"PRILOC" elements in RHS file %s but 2^scale=%"PRILOC" were needed!\n",
				rank, nread, rval.str, N);
			exit(EXIT_FAILURE);
		}
		CHECK_CUDA(cudaMemcpy(r_d[0], r_h + m->firstRow, m->intColsNum*sizeof(*(r_d[0])), cudaMemcpyHostToDevice));
	}
	MPI_Barrier(MPI_COMM_WORLD);
	tg = MPI_Wtime()-tg;
#if 1
	postIrecvs(m->recvBuffer, m->recvNum, m->recvNeigh, m->recvOffs, m->recvCnts,
		   reqs, MPI_COMM_WORLD);
#endif
	//MPI_Pcontrol(1);
	tc = MPI_Wtime();
	START_RANGE("SPMV_ALL_ITER", 1);
	for(i = 0; i < numIter; i++) {

		int s = i&1;
		int d = s^1;

		START_RANGE("BEFORE_SPMV_LOOP", 0);
#ifdef ASYNC_RED
		reduce_cuda_async(r_d[s], m->intColsNum, &sum, stream[0]);
#else
		sum = reduce_cuda(r_d[s], m->intColsNum);
#endif		
		if (m->totToSend) {
			CHECK_CUDA(cudaEventRecord(event[2], stream[1]));
#ifdef USE_MAPPED_SENDBUF
			getSendElems(r_d[s], m->rowsToSend_d, m->totToSend, m->sendBuffer_m, stream[1]);
#else
			getSendElems(r_d[s], m->rowsToSend_d, m->totToSend, m->sendBuffer_d, stream[1]);
			CHECK_CUDA(cudaMemcpyAsync(m->sendBuffer,
						   m->sendBuffer_d,
						   m->totToSend*sizeof(*m->sendBuffer),
						   cudaMemcpyDeviceToHost, stream[1]));
#endif
			CHECK_CUDA(cudaEventRecord(event[3], stream[1]));
		}

#ifdef ASYNC_RED
		CHECK_CUDA(cudaStreamSynchronize(stream[0]));
#endif
		t = MPI_Wtime();
		MPI_Allreduce(MPI_IN_PLACE, &sum, 1, REAL_MPI, MPI_SUM, MPI_COMM_WORLD);
		tmpi[0] += MPI_Wtime()-t;

		setarray(r_d[d], m->intColsNum, a*sum, stream[0]);
		END_RANGE;

		START_RANGE("SPMV_LOOP", 1);
		for(int k = 0; k < m->ncsr; k++) {

			int curr = (rank+k) % ntask;

			START_RANGE("SPMV_str0", 2);
			CHECK_CUDA(cudaEventRecord(event[0], stream[0]));
			computeSpmvAcc(/* NO ADD TERM */
				       c, m->nrows[curr], m->rows_d[curr],
                            	       m->roff_d[curr], m->cols_d[curr], m->vals_d[curr],
                            	       r_d[s], r_d[d], m->koff[curr], stream[0]);
			CHECK_CUDA(cudaEventRecord(event[1], stream[0]));

			START_RANGE("MPI+H2D_str1", 3)
			if (k < m->ncsr-1) {

				START_RANGE("SYNC_str1", 4)
				if (k == 0) {
					// wait for the D2H copy of m->sendBuffer initiated before SPMV loop
					if (m->totToSend) {
						CHECK_CUDA(cudaStreamSynchronize(stream[1]));
						CHECK_CUDA(cudaEventElapsedTime(&evt, event[2], event[3]));
						td2h[0] += evt / 1000.0; 
					}
				}
				END_RANGE;
			
	 			START_RANGE("MPI", 5)
				//MPI_Barrier(MPI_COMM_WORLD);
				t = MPI_Wtime();
				exchangeDataSingle(m->sendBuffer + m->sendOffs[k], m->sendNeigh[k], m->sendCnts[k],
						   m->recvBuffer + m->recvOffs[k], m->recvNeigh[k], m->recvCnts[k],
						   reqs+k, MPI_COMM_WORLD);
				tmpi[0] += MPI_Wtime()-t;
				END_RANGE;

				START_RANGE("H2D_str1", 6)
				if (m->recvCnts[k]) {
					CHECK_CUDA(cudaEventRecord(event[2], stream[1]));
					CHECK_CUDA(cudaMemcpyAsync(r_d[s] + m->intColsNum + m->recvOffs[k],
								   m->recvBuffer + m->recvOffs[k],
								   m->recvCnts[k]*sizeof(*r_d[s]),
								   cudaMemcpyHostToDevice, stream[1]));
					CHECK_CUDA(cudaEventRecord(event[3], stream[1]));

					CHECK_CUDA(cudaStreamSynchronize(stream[1]));
					CHECK_CUDA(cudaEventElapsedTime(&evt, event[2], event[3]));
					th2d[0] += evt / 1000.0; 
				}
				END_RANGE;
			}
			END_RANGE;

			CHECK_CUDA(cudaStreamSynchronize(stream[0]));
			CHECK_CUDA(cudaEventElapsedTime(&evt, event[0], event[1]));
			tspmv[0] += evt / 1000.0; 
			END_RANGE;
		}
		END_RANGE;
	}
	//MPI_Pcontrol(0);
	END_RANGE;
	MPI_Barrier(MPI_COMM_WORLD);
	tc = MPI_Wtime()-tc;

	sum = reduce_cuda(r_d[numIter&1], m->intColsNum);
	MPI_Reduce(rank?&sum:MPI_IN_PLACE, &sum, 1, REAL_MPI, MPI_SUM, 0, MPI_COMM_WORLD);
	
	{// to test results for now...
		char fname[256];
		snprintf(fname, 256, "myresult_%d.txt", rank);
		REAL *r = new REAL[m->intColsNum];
		CHECK_CUDA(cudaMemcpy(r, r_d[numIter&1], m->intColsNum*sizeof(*r), cudaMemcpyDeviceToHost));
		FILE *fp = fopen(fname, "w");
		for(i = 0; i < m->intColsNum; i++)
			fprintf(fp, "%E\n", r[i]);
		fclose(fp);
		delete [] r;
	}


	MPI_Reduce(td2h, td2h+1, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	MPI_Reduce(rank?td2h:MPI_IN_PLACE, td2h, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);

	MPI_Reduce(tmpi, tmpi+1, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	MPI_Reduce(rank?tmpi:MPI_IN_PLACE, tmpi, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);

	MPI_Reduce(th2d, th2d+1, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	MPI_Reduce(rank?th2d:MPI_IN_PLACE, th2d, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);

	MPI_Reduce(tspmv, tspmv+1, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	MPI_Reduce(rank?tspmv:MPI_IN_PLACE, tspmv, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);

	if (0 == rank) {
		printf("Kernel 3 (scale=%d, edgef=%d, numiter=%d, c=%G, a=%G, rhs=%s):\n",
		       scale, edgef, numIter, c, a, (rval.code==0)?"RND":rval.str); 

		printf("\tgen   time: %.4lf secs\n", tg);
		printf("\tcomp  time: %.4lf secs\n", tc);
		printf("\t\tmin/max  d2h: %.4lf/%.4lf secs\n", td2h[0], td2h[1]);
		printf("\t\tmin/max  mpi: %.4lf/%.4lf secs\n", tmpi[0], tmpi[1]);
		printf("\t\tmin/max  h2d: %.4lf/%.4lf secs\n", th2d[0], th2d[1]);
		printf("\t\tmin/max spmv: %.4lf/%.4lf secs\n", tspmv[0], tspmv[1]);
		printf("\tTOTAL time: %.4lf secs, %.2lf Medges/sec, MFLOPS: %.4f\n\n", 
		       (tg+tc), 1E-6*N*edgef*numIter/(tg+tc), 2E-6*N*edgef*numIter/(tg+tc));
		printf("PageRank sum: %E\n", sum);
	}

	cancelReqs(reqs, m->recvNum);

	CHECK_CUDA(cudaStreamDestroy(stream[0]));
	CHECK_CUDA(cudaStreamDestroy(stream[1]));
	CHECK_CUDA(cudaEventDestroy(event[0]));
	CHECK_CUDA(cudaEventDestroy(event[1]));

	if (r_h) free(r_h);
	if (reqs) free(reqs);
	if (r_d[0]) CHECK_CUDA(cudaFree(r_d[0]));
	if (r_d[1]) CHECK_CUDA(cudaFree(r_d[1]));

	return;
}

int main(int argc, char **argv) {

	int	rank, ntask;
	int	och;
	REAL	c=REAL_MAX, a=REAL_MAX;
	
	int	rv;
	rhsv_t	rval = {RHS_RANDOM, REALV(0.0), NULL};
	
	int	scale=21, edgef=16, nit=20;
	int	startk=0, doIO=1;
	LOCINT	N;

	elist_t	*el=NULL;
	int64_t	gseed=0;
	char	*ginFile=NULL;
	int	krcf=1, krpm=1;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &ntask);

	while (1) {
		int option_index = 0;
		static struct option long_options[] = {
			{"scale",     required_argument, 0,     'S'},
			{"edgef",     required_argument, 0,     'E'},
			{"seed",      required_argument, 0,     's'},
			{"help",      required_argument, 0,     'h'},
			{"iter",      required_argument, 0,     'i'},
			{"kstart",    required_argument, 0,     'k'},
			{"rhs",       required_argument, 0,     'r'},
			{"fast",      no_argument,	      &doIO,  0 },
			{"pmesh",     required_argument, 0,     'p'},
			{"graph",     required_argument, 0,     'g'},
			{"no-krcf",   no_argument,       &krcf,  0 },
			{"no-krpm",   no_argument,       &krpm,  0 },
			{"no-rcache", no_argument,   &avoid_read_cache, 1},
			{0,        0,                 0,      0 }
		};

		och = getopt_long(argc, argv, "S:E:s:hi:k:a:c:r:fg:", long_options, &option_index);
		if (och == -1) break;

		MPI_Barrier(MPI_COMM_WORLD);
		switch (och) {
			case   0:// handles long opts with non-NULL flag field
				break;
			case 'S':
				if (0 == sscanf(optarg, "%d", &scale))
					prexit("Invalid scale (-S): %s\n", optarg);
					
				if (scale > 32 && LOCINT_SIZE != 8)  
					prexit("Scale (-S) too big for 32 bit integer!\n");
				break;
			case 'E':
				if (0 == sscanf(optarg, "%d", &edgef))
					prexit("Invalid edge factor (-S): %s\n", optarg);
				break;
			case 's':
                                if (0 == sscanf(optarg, "%"PRId64, &gseed))
                                        prexit("Invalid seed for graph generator (-s): %s\n", optarg);
                                break;
			case 'i':
                                if (0 == sscanf(optarg, "%d", &nit))
                                        prexit("Invalid number of iterations (-i): %s\n", optarg);
                                break;
			case 'k':
                                if (0 == sscanf(optarg, "%d", &startk))
                                        prexit("Invalid number for start kernel (-k): %s\n", optarg);
				if (startk < 0 || startk > 2) {
					if (rank == 0)
						fprintf(stderr, "Error, start kernel must be >= 0 and <= 2!\n");
					MPI_Finalize();
					exit(EXIT_FAILURE);
				}
                                break;
			case 'c':
                                if (0 == sscanf(optarg, REAL_SPEC, &c))
                                        prexit("Invalid value for c factor (-c): %s\n", optarg);
                                break;
			case 'a':
                                if (0 == sscanf(optarg, REAL_SPEC, &a))
                                        prexit("Invalid value for a factor (-a): %s\n", optarg);
                                break;
			case 'r':
				rval.str = strdup(optarg);
				if (!rval.str) {
					fprintf(stderr, "Cannot allocate %zu bytes!\n", strlen(optarg));
					exit(EXIT_FAILURE);
				}
				rval.code = RHS_CONSTANT;
				rv = sscanf(rval.str, REAL_SPEC, &rval.fp);
                                if (0 == rv) {
					rval.code = RHS_FILE;
				}
                                break;
			case 'f':
				doIO = 0;
				break;
			case 'g':
				ginFile = strdup(optarg);
				break;
			case 'h':
				usage(argv[0]);
			case '?':
				usage(argv[0]);
			default:
				if (rank == 0)
					fprintf(stderr, "unknown option: %c\n", och);
				usage(argv[0]);
		}
	}
#ifdef __APPLE__
	if (avoid_read_cache) {
		prexit("Option --no-rcache not supported on Apple MacOSX!\n");
	}
#endif
	if (ginFile && startk != 0) {
		prexit("Graphs from file (-g option) can only be processed starting from kernel 0 (-k 0).\n"); 
	}

	N=((LOCINT)1)<<scale;
	if (REAL_MAX == c) c = REALV(0.85);
	if (REAL_MAX == a) a = (REALV(1.0)-c)/((REAL)N);

	MPI_Barrier(MPI_COMM_WORLD);
	init_cuda();

	spmat_t *m = createSpmat(ntask);

	if (!doIO) {
		el = (elist_t *)Malloc(sizeof(*el));
		el->u = NULL;
		el->v = NULL;
		el->ned = 0;
	}

	if (rank == 0) {
		fprintf(stdout, "\nRunning Pagerank Benchmark with:\n\n");
		fprintf(stdout, "Scale: %d (%"PRILOC" vertices)\n", scale, N);
		fprintf(stdout, "Edge factor: %d (%"PRId64" edges)\n", edgef, ((int64_t)N)*edgef);
		fprintf(stdout, "Number of processes: %d\n", ntask);
		fprintf(stdout, "Rhs vector: %s\n", (rval.code==RHS_RANDOM)?"random":rval.str); 
		fprintf(stdout, "Size of real: %zu\n", sizeof(REAL)); 
		fprintf(stdout, "Size of integer: %zu\n", sizeof(LOCINT)); 
		fprintf(stdout, "Starting from kernel: %d\n", startk);
		fprintf(stdout, "Number of iterations for kernel 3: %d\n", nit); 
		fprintf(stdout, "Fast mode %s\n", doIO?"OFF (write to disk)":"ON (don't write to disk)");
		if (doIO) {
			fprintf(stdout, "\tusing standard I/O library");
			if (avoid_read_cache)
				fprintf(stdout, " (bypassing file read cache)");
			fprintf(stdout, "\n");
		}
		if (ginFile == NULL) {
			fprintf(stdout, "Generating graph with:\n");
			fprintf(stdout, "\tseed: %"PRId64"\n", gseed);
			fprintf(stdout, "\tclip'n'flip: %s\n", krcf?"yes":"no");
			fprintf(stdout, "\tvertex permutation: %s\n", krpm?"yes":"no");
		} else {
			fprintf(stdout, "Reading graph from file: %s\n", ginFile); 
		}
		fprintf(stdout, "\n");
	}
	switch(startk) {
		case 0: kernel0(scale, edgef, gseed, krcf, krpm, ginFile, el);
		case 1: kernel1(scale, edgef, 100.0, el, 1);
		case 2: kernel2_multi(scale, edgef, m, el);
	}

	if (!doIO) {
		if (el) {
			if (el->u) free(el->u);
			if (el->v) free(el->v);
			free(el);
		}
	}
	kernel3_multi(scale, edgef, nit, c, a, rval, m);

	if (rval.str) free(rval.str);

	destroySpmat(m);
	cleanup_cuda();

	if (ginFile) free(ginFile);

	CHECK_CUDA(cudaDeviceReset());
	MPI_Finalize();

	return EXIT_SUCCESS;
}
