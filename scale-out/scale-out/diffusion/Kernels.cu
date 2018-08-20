//
//  kernels.cu
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

/******************************/
/* Kernel for computing halos */
/******************************/
__global__ void copy_br_to_gc(
  const REAL * __restrict__ un, 
  REAL * __restrict__ gc_un, 
  const unsigned int Nx, 
  const unsigned int Ny, 
  const unsigned int _Nz, 
  const unsigned int pitch, 
  const unsigned int gc_pitch, 
  const unsigned int p /* p = {0,1} */) 
{
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int j = threadIdx.y + blockIdx.y * blockDim.y;

  unsigned int k0 = _Nz-6 + p*(3-_Nz+6); // {0,1}: k1 = {3,(Nz-1)-5}
  unsigned int k1 = _Nz-5 + p*(4-_Nz+5); // {0,1}: k2 = {4,(Nz-1)-4}
  unsigned int k2 = _Nz-4 + p*(5-_Nz+4); // {0,1}: k3 = {5,(Nz-1)-3}
  unsigned int xy = Ny*pitch;
  unsigned int gc_xy = Ny*gc_pitch;
  unsigned int ibr0 = i + j*  pitch  + k0*xy;
  unsigned int ibr1 = i + j*  pitch  + k1*xy;
  unsigned int ibr2 = i + j*  pitch  + k2*xy;
  unsigned int igc0 = i + j*gc_pitch;//0*gc_xy;
  unsigned int igc1 = i + j*gc_pitch + 1*gc_xy;
  unsigned int igc2 = i + j*gc_pitch + 2*gc_xy;

  if( i < Nx && j < Ny && k2 < _Nz )
  {
    gc_un[igc0] = un[ibr0];
    gc_un[igc1] = un[ibr1];
    gc_un[igc2] = un[ibr2];
  }
}

/******************************/
/* Kernel for computing halos */
/******************************/
__global__ void copy_gc_to_br(
  REAL * __restrict__ un, 
  const REAL * __restrict__ gc_un, 
  const unsigned int Nx, 
  const unsigned int Ny, 
  const unsigned int _Nz, 
  const unsigned int pitch, 
  const unsigned int gc_pitch, 
  const unsigned int p /* p = {0,1} */)
{
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int j = threadIdx.y + blockIdx.y * blockDim.y;

  unsigned int k0 = _Nz-3 + p*( -_Nz+3); // {0,1}: k1 = {0,(Nz-1)-2}
  unsigned int k1 = _Nz-2 + p*(1-_Nz+2); // {0,1}: k2 = {1,(Nz-1)-1}
  unsigned int k2 = _Nz-1 + p*(2-_Nz+1); // {0,1}: k3 = {2,(Nz-1)-0}
  unsigned int xy = Ny*pitch;
  unsigned int gc_xy = Ny*gc_pitch;
  unsigned int igc0 = i + j*gc_pitch;//0*gc_xy;
  unsigned int igc1 = i + j*gc_pitch + 1*gc_xy;
  unsigned int igc2 = i + j*gc_pitch + 2*gc_xy;
  unsigned int ibr0 = i + j*  pitch  + k0*xy;
  unsigned int ibr1 = i + j*  pitch  + k1*xy;
  unsigned int ibr2 = i + j*  pitch  + k2*xy;

  if( i < Nx && j < Ny && k2 < _Nz )
  {
    un[ibr0] = gc_un[igc0];
    un[ibr1] = gc_un[igc1];
    un[ibr2] = gc_un[igc2];
  }
}

/***********************************************************/
/* Kernel for computing the 3D Laplace Operator on the GPU */
/***********************************************************/
__global__ void Laplace(
  REAL * __restrict__ u, 
  REAL * __restrict__ Lu,  
  const REAL diff, 
  const unsigned int nx, 
  const unsigned int ny, 
  const unsigned int nz)
{
  register REAL above2;
  register REAL above;
  register REAL center;
  register REAL below;
  register REAL below2;
  unsigned int i, j, k, o, xy, nx2, xy2;
  xy = nx*ny; nx2 = nx+nx; xy2 = xy+xy; 

    i = threadIdx.x + blockIdx.x * blockDim.x;
    j = threadIdx.y + blockIdx.y * blockDim.y;

    o = i+nx*j+xy2;

    if (i>1 && i<nx-2 && j>1 && j<ny-2)
    {
      below2=u[o-xy2]; below=u[o-xy]; center=u[o]; above=u[o+xy]; above2=u[o+xy2];

      Lu[o] += diff/12 * (
        - u[o-2] +16*u[o-1] + 16*u[o+1] - u[o+2]  +
        -u[o-nx2]+16*u[o-nx]- 90*center + 16*u[o+nx] - u[o+nx2]+ 
        - below2 +16*below  + 16* above - above2 );


      for(k = 3; k < nz-2; k++)
      {
        o=o+xy; below2=below; below=center; center=above; above=above2; above2=u[o+xy2];

        Lu[o] += diff/12 * (
          - u[o-2] +16*u[o-1] + 16*u[o+1] - u[o+2]  +
          -u[o-nx2]+16*u[o-nx]- 90*center + 16*u[o+nx] - u[o+nx2]+ 
          - below2 +16* below + 16* above - above2 );
      }
    }
    // else : do nothing!
}


/*****************************************************************/
/* Kernel for computing the 3D laplace operator async on the GPU */
/*****************************************************************/
__global__ void LaplaceO2_async(
  const REAL * __restrict__ u, 
  REAL * __restrict__ Lu,
  const REAL diff_x,
  const REAL diff_y,
  const REAL diff_z,
  const unsigned int picth, // pitch
  const unsigned int Nx, 
  const unsigned int Ny, 
  const unsigned int _Nz, 
  const unsigned int kstart, 
  const unsigned int kstop, 
  const unsigned int loop_z)
{
  register REAL center;
  register REAL above;
  register REAL below;
  unsigned int i, j, k, o, z, XY;

    i = threadIdx.x + blockIdx.x * blockDim.x;
    j = threadIdx.y + blockIdx.y * blockDim.y;
    k =  blockIdx.z * loop_z;

    k = MAX(kstart,k);

    XY = picth*Ny; o = i+picth*j+XY*k;

    if (i>2 && i<Nx-3 && j>2 && j<Ny-3)
    {
    below = u[o-XY]; center= u[o]; above = u[o+XY];

      Lu[o] = diff_x * (u[o-1] - 2*center + u[o+1]) +
              diff_y * (u[o-picth] - 2*center + u[o+picth]) +
              diff_z * (below - 2*center + above);

      for(z = 1; z < loop_z; z++)
      {
        k += 1;

        if (k < MIN(kstop,_Nz+1))
        {
          o = o+XY; below = center; center = above; above = u[o+XY];

        Lu[o] = diff_x * (u[o-1] - 2*center + u[o+1]) +
                diff_y * (u[o-picth] - 2*center + u[o+picth]) +
                diff_z * (below - 2*center + above);
        }
      }
    }
}


/*****************************************************************/
/* Kernel for computing the 3D Laplace async Operator on the GPU */
/*****************************************************************/
__global__ void LaplaceO4_async(
	const REAL * __restrict__ u, 
	REAL * __restrict__ Lu, 
	const REAL diff_x,
  const REAL diff_y,
  const REAL diff_z,
	const unsigned int pitch, // allocation pitch
	const unsigned int Nx, 
	const unsigned int Ny, 
	const unsigned int _Nz, 
	const unsigned int kstart, 
	const unsigned int kstop, 
	const unsigned int loop_z)
{
	register REAL above2;
  register REAL above;
  register REAL center;
  register REAL below;
  register REAL below2;
	unsigned int i, j, k, o, z, XY, Nx2, XY2;

  i = threadIdx.x + blockIdx.x * blockDim.x;
  j = threadIdx.y + blockIdx.y * blockDim.y;
  k =  blockIdx.z * loop_z;

  k = MAX(kstart,k);

  XY=pitch*Ny; Nx2=pitch+pitch; XY2=XY+XY; o=i+pitch*j+XY*k;

  if (i>2 && i<Nx-3 && j>2 && j<Ny-3)
  {
    below2=u[o-XY2]; below=u[o-XY]; center=u[o]; above=u[o+XY]; above2=u[o+XY2];

    Lu[o] = diff_x/12 * (
      - u[o-2] + 16*u[o-1] + 16*u[o+1] - u[o+2] 
      -u[o-Nx2] + 16*u[o-pitch] - 90*center + 16*u[o+pitch] - u[o+Nx2]
      - below2 + 16*below + 16* above - above2 );

  	for(z = 1; z < loop_z; z++)
  	{
  		k += 1;

  		if (k < MIN(kstop,_Nz+1))
  		{
  			o=o+XY; below2=below; below=center; center=above; above=above2; above2=u[o+XY2];

        Lu[o] = diff_x/12 * (
          - u[o-2] + 16*u[o-1] + 16*u[o+1] - u[o+2] 
          -u[o-Nx2] + 16*u[o-pitch] - 90*center + 16*u[o+pitch] - u[o+Nx2]
          - below2 + 16* below + 16* above - above2 );
  		}
  	}
  }
  // else : do nothing!
}

/***********************/
/* Runge Kutta Methods */  // <==== this is perfectly parallel!
/***********************/
__global__ void Compute_RK( 
  REAL * __restrict__ q, 
  const REAL * __restrict__ qo, 
  const REAL * __restrict__ Lq, 
  const unsigned int step,
  const unsigned int pitch, 
  const unsigned int Nx,
  const unsigned int Ny,
  const unsigned int _Nz, 
  const REAL dt)
{
  unsigned int i, j, k, o, XY;
  // local threads indexes
  i = blockDim.x * blockIdx.x + threadIdx.x;
  j = blockDim.y * blockIdx.y + threadIdx.y;
  k = blockDim.z * blockIdx.z + threadIdx.z;

  // Single index
  XY = pitch*Ny; o=i+pitch*j+XY*k; 

  // Compute Runge-Kutta step only on internal cells
  // if (i < Nx && j < Ny && k < _Nz)
  if (i>2 && i<Nx-3 && j>2 && j<Ny-3 && k<_Nz)
  {
    switch (step) {
      case 1: // step 1
        q[o] = qo[o]+dt*Lq[o]; break;
      case 2: // step 2
        q[o] = 0.75*qo[o]+0.25*(q[o]+dt*Lq[o]); break;
      case 3: // step 3
        q[o] = (qo[o]+2*(q[o]+dt*Lq[o]))/3; break;
    }
    // q[o] = o; // <-- debuging tool
  }
}


/***********************/
/* Runge Kutta Methods */  // <==== this is perfectly parallel!
/***********************/
__global__ void Compute_RK_async( 
  REAL * __restrict__ q, 
  const REAL * __restrict__ qo, 
  const REAL * __restrict__ Lq, 
  const unsigned int step,
  const unsigned int pitch, 
  const unsigned int Nx, 
  const unsigned int Ny, 
  const unsigned int _Nz, 
  const unsigned int kstart, 
  const unsigned int kstop, 
  const unsigned int loop_z, 
  const REAL dt)
{
  unsigned int i, j, k, o, z, XY;
  // local threads indexes
  i = blockDim.x * blockIdx.x + threadIdx.x;
  j = blockDim.y * blockIdx.y + threadIdx.y;
  k = blockIdx.z * loop_z;

  k = MAX(kstart,k);

  // Single index
  XY = pitch*Ny; o=i+pitch*j+XY*k;

  // Compute Runge-Kutta step only on internal cells
  // if (i < Nx && j < Ny && k < _Nz)
  if (i>2 && i<Nx-3 && j>2 && j<Ny-3 && k<_Nz)
  {
    for(z = 0; z < loop_z; z++)
      {
        if (k < MIN(kstop,_Nz+1)) 
        {
          switch (step) {
            case 1: // step 1
              q[o] = qo[o]+dt*(Lq[o]); break;
            case 2: // step 2
              q[o] = 0.75*qo[o]+0.25*(q[o]+dt*(Lq[o])); break;
            case 3: // step 3
              q[o] = (qo[o]+2*(q[o]+dt*(Lq[o])))/3; break;
          }
          o = o+XY;
        }
        k += 1;
      }
  }
}


/********************/
/* Print GPU memory */ 
/********************/
__global__ void PrintGPUmemory( 
  REAL * __restrict__ q, 
  const unsigned int pitch, 
  const unsigned int Nx, 
  const unsigned int Ny, 
  const unsigned int _Nz, 
  const unsigned int kstart, 
  const unsigned int kstop)
{
  unsigned int i, j, k, o, XY=pitch*Ny;

  printf("kstart: %d,\t kstop: %d\n",kstart,kstop);

  // Print only on internal cells
  for (k = 0; k < _Nz; k++)
  { 
    for (j = 0; j < Ny; j++) 
    {
      for (i = 0; i < Nx; i++) 
      {
        o=i+pitch*j+XY*k; printf("%8.2f", q[o]); 
      }
      printf("\n"); 
    }
    printf("\n"); 
  }
}

/******************************************************************************/
/* Function that copies content from the host to the device's constant memory */
/******************************************************************************/
extern "C" void CopyBoundaryRegionToGhostCellAsync(dim3 thread_blocks_halo, dim3 threads_per_block, cudaStream_t aStream, 
  REAL* d_s_q, REAL* d_send_buffer, unsigned int Nx, unsigned int Ny, unsigned int _Nz, unsigned int pitch, unsigned int gc_pitch, unsigned int p)
{
  copy_br_to_gc<<<thread_blocks_halo,threads_per_block,0,aStream>>>(d_s_q, d_send_buffer, Nx, Ny, _Nz, pitch, gc_pitch, p);
}

extern "C" void CopyGhostCellToBoundaryRegionAsync(dim3 thread_blocks_halo, dim3 threads_per_block, cudaStream_t aStream, 
  REAL* d_s_q, REAL* d_recv_buffer, unsigned int Nx, unsigned int Ny, unsigned int _Nz, unsigned int pitch, unsigned int gc_pitch, unsigned int p)
{
  copy_gc_to_br<<<thread_blocks_halo,threads_per_block,0,aStream>>>(d_s_q, d_recv_buffer, Nx, Ny, _Nz, pitch, gc_pitch, p);
}

extern "C" void Call_sspRK(dim3 numBlocks, dim3 threadsPerBlock, cudaStream_t aStream, 
  unsigned int step, unsigned int pitch, unsigned int Nx, unsigned int Ny, unsigned int _Nz, const REAL dt, 
  REAL *u, REAL *uo, REAL *Lu)
{
  Compute_RK<<<numBlocks,threadsPerBlock,0,aStream>>>(u,uo,Lu,step,pitch,Nx,Ny,_Nz,dt);
}

extern "C" void Call_Diff_(dim3 numBlocks, dim3 threadsPerBlock, cudaStream_t aStream, 
  unsigned int pitch, unsigned int Nx, unsigned int Ny, unsigned int _Nz, unsigned int kstart, unsigned int kstop, 
  REAL diff_x, REAL diff_y, REAL diff_z, REAL* q, REAL* Lq)
{
  // LaplaceO2_async<<<numBlocks,threadsPerBlock,0,aStream>>>(q,Lq,diff_x,diff_y,diff_z,pitch,Nx,Ny,_Nz,kstart,kstop,LOOP);
  LaplaceO4_async<<<numBlocks,threadsPerBlock,0,aStream>>>(q,Lq,diff_x,diff_y,diff_z,pitch,Nx,Ny,_Nz,kstart,kstop,LOOP);
}

extern "C" void printGPUmem(dim3 numBlocks, dim3 threadsPerBlock, cudaStream_t aStream, 
  unsigned int pitch, unsigned int Nx, unsigned int Ny, unsigned int _Nz, unsigned int kstart, unsigned int kstop, 
  REAL *q)
{
  PrintGPUmemory<<<numBlocks,threadsPerBlock,0,aStream>>>(q,pitch,Nx,Ny,_Nz,kstart,kstop);
}
