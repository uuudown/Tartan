//
//  Tools.c
//  Burgers3d-GPU-MPI
//
//  Created by Manuel Diaz on 7/26/16.
//  Copyright © 2016 Manuel Diaz. All rights reserved.
//

#include "DiffusionMPICUDA.h"

#define checkCuda(error) __checkCuda(error, __FILE__, __LINE__)

/*********************************************/
/* A method for checking error in CUDA calls */
/*********************************************/
inline void __checkCuda(cudaError_t error, const char *file, const int line)
{
	#if defined(DEBUG) || defined(_DEBUG)
	if (error != cudaSuccess)
	{
		printf("checkCuda error at %s:%i: %s\n", file, line, cudaGetErrorString(cudaGetLastError()));
		exit(-1);
	}
	#endif

	return;
}

/*******************************/
/* Prints a flattened 3D array */
/*******************************/
void Print3D(REAL *u, const unsigned int nx, const unsigned int ny, const unsigned int nz)
{
  unsigned int i, j, k, xy;
  xy=nx*ny; 
  // print a single property on terminal
  for(k = 0; k < nz+HALO; k++) {
    for (j = 0; j < ny; j++) {
      for (i = 0; i < nx; i++) {
				printf("%8.2f", u[i+nx*j+xy*k]);
      }
      printf("\n");
    }
    printf("\n");
  }
  printf("\n");
}

/*******************************/
/* Prints a flattened 2D array */
/*******************************/
void Print2D(REAL *u, const unsigned int nx, const unsigned int ny)
{
  unsigned int i, j;
  // print a single property on terminal
  for (j = 0; j < ny; j++) {
    for (i = 0; i < nx; i++) {
      printf("%g ", u[i+nx*j]);
    }
    printf("\n");
  }
  printf("\n");
}

/*****************************/
/* Write ASCII file 3D array */
/*****************************/
void Save3D(REAL *u, const unsigned int nx, const unsigned int ny, const unsigned int nz)
{
  unsigned int i, j, k, xy;
  xy = nx*ny;
  // print result to txt file
  FILE *pFile = fopen("result.bin", "w");
  if (pFile != NULL) {
    for (k = 0; k < nz; k++) {
      for (j = 0; j < ny; j++) {
        for (i = 0; i < nx; i++) {
          fprintf(pFile, "%g\n",u[i+nx*j+xy*k]);
        }
      }
    }
    fclose(pFile);
  } else {
    printf("Unable to save to file\n");
  }
}

/******************************/
/* Write Binary file 3D array */
/******************************/
void SaveBinary3D(REAL *u, const unsigned int nx, const unsigned int ny, const unsigned int nz, const char *name)
{
  /* NOTE: We save our result as float values always!
   *
   * In Matlab, the results can be loaded by simply doing 
   *  >> fID = fopen('result.bin');
   *  >> result = fread(fID,[1,nx*ny*nz],'float')';
   *  >> myplot(result,nx,ny,nz);
   */

  float data;
  unsigned int i, j, k, xy, o;
  xy = nx*ny;
  // print result to txt file
  FILE *pFile = fopen(name, "w");
  if (pFile != NULL) {
      for (k = 0; k < nz; k++) {
          for (j = 0; j < ny; j++) {
              for (i = 0; i < nx; i++) {
                  o = i+nx*j+xy*k; // index
                  data = (float)u[o]; fwrite(&data,sizeof(float),1,pFile);
              }
          }
      }
      fclose(pFile);
  } else {
      printf("Unable to save to file\n");
  }
}

/**********************/
/* Initializes arrays */
/**********************/
void Init_domain(const int IC, REAL *u0, const REAL dx, const REAL dy, const REAL dz, unsigned int nx, unsigned int ny, unsigned int nz)
{
	unsigned int i, j, k, o, xy;
  xy = nx*ny;
	switch (IC) {
    case 1: {
      // A Square Jump problem
      for (k= 0; k < nz; k++) {
        for (j= 0; j < ny; j++) {
          for (i= 0; i < nx; i++) {
            o = i+nx*j+xy*k;
            if (i>=nx/4 && i<3*nx/4 && j>=ny/4 && j<3*ny/4 && k>=nz/4 && k<3*nz/4) {
              u0[o]=1.;
            } else {
              u0[o]=0.;
            }
          }
        }
      }
      break;
    }
    case 2: {
      // Homogeneous IC
      for (k= 0; k < nz; k++) {
        for (j= 0; j < ny; j++) {
          for (i= 0; i < nx; i++) {
            o = i+nx*j+xy*k;
            u0[o]=0.0;
          }
        }
      }
      break;
    }
		case 3: {
			// Sine Distribution in pressure field
			for(k = 0; k < nz; k++) {
				for (j = 0; j < ny; j++) {
					for (i = 0; i < nx; i++) {
						o = i+nx*j+xy*k; 
						if (i>=3 || i<nx-2 || j>=3 || j<ny-2 || k>=3 || k<nz-2) {
							u0[o] = GAUSSIAN_DISTRIBUTION((0.5*(nx-1)-i)*dx,(0.5*(ny-1)-j)*dy,(0.5*(nz-1)-k)*dz);
						} else {
							u0[o] = 0.0;
						}
					}
				}
			}
			break;
		}
		// Here to add another IC
	}
}

/******************************/
/* Initialize the sub-domains */
/******************************/
void Init_subdomain(REAL *h_q, REAL *h_s_q, unsigned int n, unsigned int Nx, unsigned int Ny, unsigned int _Nz)
{
	unsigned int idx_3d; // Global 3D index
	unsigned int idx_sd; // Subdomain index
	unsigned int i, j, k, XY, NX; 
	XY = Nx*Ny; NX = Nx;

	// Copy Domain into n-subdomains
	for(k = 0; k < _Nz+HALO; k++) {
		for (j = 0; j < Ny; j++) {
			for (i = 0; i < Nx; i++) {

				idx_3d = i+NX*j+XY*(k+n*_Nz);
				idx_sd = i+NX*j+XY*(k);

				h_s_q[idx_sd] = h_q[idx_3d];
			}
		}
	}
}

/*******************************************************/
/* Merges the smaller sub-domains into a larger domain */
/*******************************************************/
void Merge_domains(REAL *h_s_q, REAL *h_q, unsigned int n, unsigned int Nx, unsigned int Ny, unsigned int _Nz)
{
	unsigned int idx_3d; // Global 3D index
	unsigned int idx_sd; // Subdomain index
	unsigned int i, j, k, XY, NX; 
	XY = Nx*Ny; NX = Nx;

	// Copy n-subdomains into the Domain
	for(k = RADIUS; k < _Nz+RADIUS; k++) {
		for (j = 0; j < Ny; j++) {
			for (i = 0; i < Nx; i++) {

				idx_3d = i+NX*j+XY*(k+n*_Nz);
				idx_sd = i+NX*j+XY*(k);

				h_q[idx_3d] = h_s_q[idx_sd];
			}
		}
	}
}

/******************************/
/* Function to initialize MPI */
/******************************/
void InitializeMPI(int* argc, char*** argv, int* rank, int* numberOfProcesses)
{
	MPI_CHECK(MPI_Init(argc, argv));
	MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, rank));
	MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, numberOfProcesses));
	MPI_CHECK(MPI_Errhandler_set(MPI_COMM_WORLD, MPI_ERRORS_RETURN));
}

/****************************/
/* Function to finalize MPI */
/****************************/
void FinalizeMPI()
{
	MPI_CHECK(MPI_Finalize());
}

/********************/
/* Calculate Gflops */
/********************/
float CalcGflops(float computeTimeInSeconds, unsigned int iterations, unsigned int nx, unsigned int ny, unsigned int nz)
{
    return (3*iterations)*(double)((nx * ny * nz) * 1e-9 * FLOPS)/computeTimeInSeconds;
}

/****************************/
/* Print Experiment Summary */
/****************************/
void PrintSummary(const char* kernelName, const char* optimization,
    double computeTimeInSeconds, double hostToDeviceTimeInSeconds, double deviceToHostTimeInSeconds, 
    float gflops, const int computeIterations, unsigned int nx, unsigned int ny, unsigned int nz)
{
    printf("=======================%s=====================\n", kernelName);
    printf("Optimization                                 :  %s\n", optimization);
    printf("Kernel time ex. data transfers               :  %lf seconds\n", computeTimeInSeconds);
    printf("Data transfer(s) HtD                         :  %lf seconds\n", hostToDeviceTimeInSeconds);
    printf("Data transfer(s) DtH                         :  %lf seconds\n", deviceToHostTimeInSeconds);
    printf("===================================================================\n");
    printf("Total effective GFLOPs                       :  %lf\n", gflops);
    printf("===================================================================\n");
    printf("3D Grid Size                                 :  %d x %d x %d\n",nx,ny,nz);
    printf("Iterations                                   :  %d x 3 RK steps\n", computeIterations);
    printf("===================================================================\n");
}