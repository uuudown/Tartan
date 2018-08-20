
#include "turH.h"

static __global__ void convolution_rotor(float2* wx,float2* wy,float2* wz,float2* ux,float2* uy,float2* uz,int NXSIZE)
{


	int j  = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;
		
	int k=j%NZ;
	j=(j-k)/NZ;

	float N3=(float) N*N*N;

	float2 m1;
	float2 m2;
	float2 m3;

	if (i<NXSIZE &&  j<NY && k<NZ )
	{
	
	int h=i*NY*NZ+j*NZ+k;	

	// Read velocity and vorticity	
	
	float2 u1=ux[h];
	float2 u2=uy[h];
	float2 u3=uz[h];
	
	float2 w1=wx[h];
	float2 w2=wy[h];
	float2 w3=wz[h];
	
	// Normalize velocity and vorticity
	
	u1.x=u1.x/N3;
	u2.x=u2.x/N3;
	u3.x=u3.x/N3;

	u1.y=u1.y/N3;
	u2.y=u2.y/N3;
	u3.y=u3.y/N3;

	w1.x=w1.x/N3;
	w2.x=w2.x/N3;
	w3.x=w3.x/N3;
		
	w1.y=w1.y/N3;
	w2.y=w2.y/N3;
	w3.y=w3.y/N3;
	
	// Calculate the convolution rotor

	m1.x=u2.x*w3.x-u3.x*w2.x;
	m2.x=u3.x*w1.x-u1.x*w3.x;
	m3.x=u1.x*w2.x-u2.x*w1.x;

	m1.y=u2.y*w3.y-u3.y*w2.y;
	m2.y=u3.y*w1.y-u1.y*w3.y;
	m3.y=u1.y*w2.y-u2.y*w1.y;

	// Output must be normalized with N^3	
	
	wx[h].x=m1.x;
	wx[h].y=m1.y;

	wy[h].x=m2.x;
	wy[h].y=m2.y;

	wz[h].x=m3.x;
	wz[h].y=m3.y;	


	}
	
	
}


static dim3 threadsPerBlock;
static dim3 blocksPerGrid;


extern  void calc_conv_rotor(vectorField r, vectorField s)
{	

	// Operate over N*N*(N/2+1) matrix	
	
	threadsPerBlock.x=THREADSPERBLOCK_IN;
	threadsPerBlock.y=THREADSPERBLOCK_IN;

	blocksPerGrid.y=NXSIZE/threadsPerBlock.x;
	blocksPerGrid.x=NY*NZ/threadsPerBlock.y;

		
	convolution_rotor<<<blocksPerGrid,threadsPerBlock>>>(r.x,r.y,r.z,s.x,s.y,s.z,NXSIZE);
	kernelCheck(RET,"convolution_rotor",1);
	
	return;

}
