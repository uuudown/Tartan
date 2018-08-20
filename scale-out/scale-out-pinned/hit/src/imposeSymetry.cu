#include "turH.h"

static __global__ void normalize_kernel(float2* t1,float2* t2,float2* t3,int IGLOBAL,int NXSIZE)
{

	
	int h  = blockIdx.x * blockDim.x + threadIdx.x;
	
	//if(i<NXSIZE &&  j<NY && k<NZ )
        if( h < (NXSIZE*NY*NZ) )
	{

	
	float N3=(float)N*(float)N*(float)N;	
	
	t1[h].x/=N3;
	t2[h].x/=N3;
	t3[h].x/=N3;

	t1[h].y/=N3;
	t2[h].y/=N3;
	t3[h].y/=N3;

		


	}

}


static dim3 threadsPerBlock;
static dim3 blocksPerGrid;
static cudaError_t ret;

// Functino to turn to zero all those modes dealiased

extern void imposeSymetry(vectorField t)
{
	
       int elements = NXSIZE*NY*NZ;

        // Operate over N*N*(N/2+1) matrix
        threadsPerBlock.x=THREADSPERBLOCK_NU;

        blocksPerGrid.x=(elements+threadsPerBlock.x-1)/threadsPerBlock.x;


	fftBackward(t.x);
	fftBackward(t.y);
	fftBackward(t.z);	

	normalize_kernel<<<blocksPerGrid,threadsPerBlock,0,compute_stream>>>(t.x,t.y,t.z,IGLOBAL,NXSIZE);
	kernelCheck(ret,"normalize kern",1);

	fftForward(t.x);
	fftForward(t.y);
	fftForward(t.z);	

	return;

}


