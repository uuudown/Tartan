#include "turH.h"


static __global__ void calcWkernel(float2* ux,float2* uy,float2* uz,float2* wx,float2* wy,float2* wz,int IGLOBAL,int NXSIZE)
{
	

	int j  = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	
	int k=j%NZ;
	j=(j-k)/NZ;
	
	float k1;
	float k2;
	float k3;

	if (i<NXSIZE && j<NY && k<NZ)
	{

	
	// X indices		
	k1=(i+IGLOBAL)<NX/2 ? (float)(i+IGLOBAL) : (float)(i+IGLOBAL)-(float)NX;

	// Y indice
	
	k2=j<NY/2 ? (float)j : (float)j-(float)NY;
	
	// Z indices
	k3=(float)k;

	float kk=k1*k1+k2*k2+k3*k3;	
	
	int kl=floor(sqrt(kk)+0.5);

	int h=i*NY*NZ+j*NZ+k;

	// Read {u1,u2,u3}	
	
	float2 u1=ux[h];
	float2 u2=uy[h];
	float2 u3=uz[h];
	
	float2 w1;
	float2 w2;
	float2 w3;
	
	w1.x=-(k2*u3.y-k3*u2.y);
	w1.y=  k2*u3.x-k3*u2.x ;

	w2.x=-(k3*u1.y-k1*u3.y);
	w2.y=  k3*u1.x-k1*u3.x ;

	w3.x=-(k1*u2.y-k2*u1.y);
	w3.y=  k1*u2.x-k2*u1.x ;


	wx[h]=w1;
	wy[h]=w2;
	wz[h]=w3;

	}
	
	
}


static dim3 threadsPerBlock;
static dim3 blocksPerGrid;


extern void calc_U_W( vectorField U,vectorField W)
{
	
	// Operate over N*N*(N/2+1) matrix

	threadsPerBlock.x=THREADSPERBLOCK_IN;
	threadsPerBlock.y=THREADSPERBLOCK_IN;

	blocksPerGrid.y=(NXSIZE+THREADSPERBLOCK_IN-1)/THREADSPERBLOCK_IN;
	blocksPerGrid.x=NY*NZ/threadsPerBlock.y;
	
	calcWkernel<<<blocksPerGrid,threadsPerBlock>>>(U.x,U.y,U.z,W.x,W.y,W.z,IGLOBAL,NXSIZE);
	kernelCheck(RET,"Wkernel",1);
	


}








