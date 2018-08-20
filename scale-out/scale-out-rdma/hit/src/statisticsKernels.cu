#include "turH.h"

static __global__ void calcEkernel(float2* ux,float2* uy,float2* uz,float2* t,int IGLOBAL,int NXSIZE)
{
	

        int ind  = blockIdx.x * blockDim.x + threadIdx.x;


        int  k = ind%NZ;
        int  i = ind/(NZ*NY);
        int  j = ind/NZ-i*NY;

	
	float N3=(float)N*(float)N*(float)N;

	int h=i*NY*NZ+j*NZ+k;

//	if(i<NXSIZE &&  j<NY && k<NZ )
        if( ind < (NXSIZE*NY*NZ) )
	{

	// Read {u1,u2,u3}	
	
	float2 u1=ux[h];
	float2 u2=uy[h];
	float2 u3=uz[h];


	// Normalize velocities

	u1.x/=N3;
	u1.y/=N3;

	
	u2.x/=N3;
	u2.y/=N3;
	
	
	u3.x/=N3;
	u3.y/=N3;
	
	float2 energy;

	if(k==0){
	energy.x=u1.x*u1.x+u2.x*u2.x+u3.x*u3.x;
	energy.y=u1.y*u1.y+u2.y*u2.y+u3.y*u3.y;
	}else{
	energy.x=2.0f*(u1.x*u1.x+u2.x*u2.x+u3.x*u3.x);
	energy.y=2.0f*(u1.y*u1.y+u2.y*u2.y+u3.y*u3.y);	
	};	

	energy.x*=0.5f;
	energy.y*=0.5f;	


	// write
	
	t[h]=energy;

	}
	
	
	
}

static __global__ void calcDkernel(float2* ux,float2* uy,float2* uz,float2* t,float Reynolds,int IGLOBAL,int NXSIZE)
{
	
        int ind  = blockIdx.x * blockDim.x + threadIdx.x;

        int  k = ind%NZ;
        int  i = ind/(NZ*NY);
        int  j = ind/NZ-i*NY;
	
	float N3=(float)N*(float)N*(float)N;

	int h=i*NY*NZ+j*NZ+k;

//	if(i<NXSIZE &&  j<NY && k<NZ )
        if( ind < (NXSIZE*NY*NZ) )
	{

	// Read {u1,u2,u3}	
	
	float2 u1=ux[h];
	float2 u2=uy[h];
	float2 u3=uz[h];
	
	float k1;
	float k2;
	float k3;

	// X indices		
	k1=(i+IGLOBAL)<NX/2 ? (float)(i+IGLOBAL) : (float)(i+IGLOBAL)-(float)NX ;

	// Y indice
	
	k2=j<NY/2 ? (float)j : (float)j-(float)NY;
	
	// Z indices
	k3=(float)k;

	float kk=k1*k1+k2*k2+k3*k3;	


	// Normalize velocities

	u1.x/=N3;
	u1.y/=N3;

	
	u2.x/=N3;
	u2.y/=N3;
	
	
	u3.x/=N3;
	u3.y/=N3;
	
	float2 energy;

	if(k==0){
	energy.x=u1.x*u1.x+u2.x*u2.x+u3.x*u3.x;
	energy.y=u1.y*u1.y+u2.y*u2.y+u3.y*u3.y;
	}else{
	energy.x=2.0f*(u1.x*u1.x+u2.x*u2.x+u3.x*u3.x);
	energy.y=2.0f*(u1.y*u1.y+u2.y*u2.y+u3.y*u3.y);	
	}	

	energy.x=(kk/Reynolds)*energy.x;
	energy.y=(kk/Reynolds)*energy.y;

	// write
	
	t[h]=energy;

	}
	
	
	
}


static dim3 threadsPerBlock;
static dim3 blocksPerGrid;


extern void calc_E_kernel( vectorField u, float2* t)
{
         int elements = NXSIZE*NY*NZ;

        // Operate over N*N*(N/2+1) matrix
        threadsPerBlock.x=THREADSPERBLOCK_NU;

        blocksPerGrid.x=(elements+threadsPerBlock.x-1)/threadsPerBlock.x;

	
	calcEkernel<<<blocksPerGrid,threadsPerBlock,0,compute_stream>>>(u.x,u.y,u.z,t,IGLOBAL,NXSIZE);
	kernelCheck(RET,"MemInfo1_caca_2",1);	

	
	return;
}


extern void calc_D_kernel( vectorField u, float2* t)
{
         int elements = NXSIZE*NY*NZ;

        // Operate over N*N*(N/2+1) matrix
        threadsPerBlock.x=THREADSPERBLOCK_NU;

        blocksPerGrid.x=(elements+threadsPerBlock.x-1)/threadsPerBlock.x;

		
	calcDkernel<<<blocksPerGrid,threadsPerBlock,0,compute_stream>>>(u.x,u.y,u.z,t,REYNOLDS,IGLOBAL,NXSIZE);
	kernelCheck(RET,"MemInfo1_caca_2",1);	
	
	
	return;
}



