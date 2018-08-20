#include "turH.h"


static __global__ void routinekernel(float2* t1,float2* t2,float2* t3,int IGLOBAL,int NXSIZE)
{

	
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;
		
	int k=j%NZ;
	j=(j-k)/NZ;

	float k1,k2,k3;
	
	// X indices		
	k1=(i+IGLOBAL)<NX/2 ? (float)(i+IGLOBAL) : (float)(i+IGLOBAL)-(float)NX ;

	// Y indice
	k2=j<NY/2 ? (float)j : (float)j-(float)NY ;
	
	// Z indices
	k3=(float)k;

	int h=i*NY*NZ+j*NZ+k;
	
	if(i<NXSIZE &&  j<NY && k<NZ )
	{

	
	
	t1[h].x=k1;
	t2[h].x=k2;
	t3[h].x=k3;

	t1[h].y=k1;
	t2[h].y=k2;
	t3[h].y=k3;

	}

}


static dim3 threadsPerBlock;
static dim3 blocksPerGrid;


// Functino to turn to zero all those modes dealiased

extern void routineCheck(vectorField t)
{
	

	//SET BLOCK DIMENSIONS
	
	threadsPerBlock.x=THREADSPERBLOCK_IN;
	threadsPerBlock.y=THREADSPERBLOCK_IN;

        blocksPerGrid.y=(NXSIZE+THREADSPERBLOCK_IN-1)/THREADSPERBLOCK_IN;
	blocksPerGrid.x=NY*NZ/threadsPerBlock.y;


	routinekernel<<<blocksPerGrid,threadsPerBlock>>>(t.x,t.y,t.z,IGLOBAL,NXSIZE);
	kernelCheck(RET,"routine kern",1);
	
	return;

}



