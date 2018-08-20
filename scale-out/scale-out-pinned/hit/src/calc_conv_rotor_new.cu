#include "turH.h"

static __global__ void convolution_rotor_3(float2* wx,float2* wy,float2* wz,float2* ux,float2* uy,float2* uz, int elements)
{


        int h  = blockIdx.x * blockDim.x + threadIdx.x;

        float N3=(float) N* (float) N* (float)N;

        float2 m3;

        if (h<elements)
        {

        // Read velocity and vorticity  

        float2 u1=ux[h];
        float2 u2=uy[h];

        float2 w1=wx[h];
        float2 w2=wy[h];

        // Normalize velocity and vorticity

        u1.x=u1.x/N3;
        u2.x=u2.x/N3;

        u1.y=u1.y/N3;
        u2.y=u2.y/N3;

        w1.x=w1.x/N3;
        w2.x=w2.x/N3;

        w1.y=w1.y/N3;
        w2.y=w2.y/N3;

        // Calculate the 3rd component  of convolution rotor

        m3.x=u1.x*w2.x-u2.x*w1.x;
        m3.y=u1.y*w2.y-u2.y*w1.y;

        // Output must be normalized with N^3   

        wz[h]=m3;

        }


}


static __global__ void convolution_rotor_12(float2* wx,float2* wy,float2* wz,float2* ux,float2* uy,float2* uz, int elements)
{


	int h  = blockIdx.x * blockDim.x + threadIdx.x;
        float N3=(float) N* (float) N* (float)N;

	float2 m1;
	float2 m2;

	if (h<elements)
	{
	
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
/*
	u1.x=u1.x*oN3;
	u2.x=u2.x*oN3;
	u3.x=u3.x*oN3;

	u1.y=u1.y*oN3;
	u2.y=u2.y*oN3;
	u3.y=u3.y*oN3;

	w1.x=w1.x*oN3;
	w2.x=w2.x*oN3;
	w3.x=w3.x*oN3;
		
	w1.y=w1.y*oN3;
	w2.y=w2.y*oN3;
	w3.y=w3.y*oN3;
*/	
	// Calculate the first and second component of the convolution rotor

	m1.x=u2.x*w3.x-u3.x*w2.x;
	m2.x=u3.x*w1.x-u1.x*w3.x;

	m1.y=u2.y*w3.y-u3.y*w2.y;
	m2.y=u3.y*w1.y-u1.y*w3.y;

	// Output must be normalized with N^3	
	
	wx[h].x=m1.x;
	wx[h].y=m1.y;

	wy[h].x=m2.x;
	wy[h].y=m2.y;


	}
	
	
}


static dim3 threadsPerBlock;
static dim3 blocksPerGrid;


extern  void calc_conv_rotor_3(vectorField r, vectorField s )
{	
        int elements = NXSIZE*NY*NZ;

	// Operate over N*N*(N/2+1) matrix	
	threadsPerBlock.x=THREADSPERBLOCK_NU;

	blocksPerGrid.x=(elements+threadsPerBlock.x-1)/threadsPerBlock.x;
		
	convolution_rotor_3<<<blocksPerGrid,threadsPerBlock,0,compute_stream>>>(r.x,r.y,r.z,s.x,s.y,s.z,elements);
	kernelCheck(RET,"convolution_rotor_3",1);
	
	return;

}

extern  void calc_conv_rotor_12(vectorField r, vectorField s, float2* temprz)
{	
        int elements = NXSIZE*NY*NZ;

	// Operate over N*N*(N/2+1) matrix	
	
        threadsPerBlock.x=THREADSPERBLOCK_NU;

        blocksPerGrid.x=(elements+threadsPerBlock.x-1)/threadsPerBlock.x;
		
	convolution_rotor_12<<<blocksPerGrid,threadsPerBlock,0,compute_stream>>>(r.x,r.y,temprz,s.x,s.y,s.z,elements);
	kernelCheck(RET,"convolution_rotor_12",1);
	
	return;

}
