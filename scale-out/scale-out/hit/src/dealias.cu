#include "turH.h"


static __global__ void dealias_kernel(float2* t1,float2* t2,float2* t3,int IGLOBAL,int NXSIZE)
{

	
        int ind  = blockIdx.x * blockDim.x + threadIdx.x;

        int  k = ind%NZ;
        int  i = ind/(NZ*NY);
        int  j = ind/NZ-i*NY;

	float k1,k2,k3;
	
	// X indices		
	k1=(i+IGLOBAL)<NX/2 ? (float)(i+IGLOBAL) : (float)(i+IGLOBAL)-(float)NX ;

	// Y indice
	k2=j<NY/2 ? (float)j : (float)j-(float)NY ;
	
	// Z indices
	k3=(float)k;
	
	int kk=k1*k1+k2*k2+k3*k3;

	int h=i*NY*NZ+j*NZ+k;
	
	//if(i<NXSIZE &&  j<NY && k<NZ )
        if( ind < (NXSIZE*NY*NZ) )
	{

	// Dealiasing also in the corners of the cube

	int kl=floor(sqrt((float)kk)+0.5f);

	if(kl>(sqrt(2.0f)/3.0f)*N){
	

		
	
	t1[h].x=0.0f;
	t2[h].x=0.0f;
	t3[h].x=0.0f;

	t1[h].y=0.0f;
	t2[h].y=0.0f;
	t3[h].y=0.0f;

		

	}

	if(i+IGLOBAL+j+k==0){

	
	t1[h].x=0.0f;
	t2[h].x=0.0f;
	t3[h].x=0.0f;

	t1[h].y=0.0f;
	t2[h].y=0.0f;
	t3[h].y=0.0f;

		

	}

	}

}

static void __global__ projectionKernel(float2* ux,float2* uy,float2* uz,int IGLOBAL,int NXSIZE)
{
        int ind  = blockIdx.x * blockDim.x + threadIdx.x;

        int  k = ind%NZ;
        int  i = ind/(NZ*NY);
        int  j = ind/NZ-i*NY;


	float k1,k2,k3;
	
	// X indices		
	k1=(i+IGLOBAL)<NX/2 ? (float)(i+IGLOBAL) : (float)(i+IGLOBAL)-(float)NX ;

	// Y indice
	k2=j<NY/2 ? (float)j : (float)j-(float)NY ;
	
	// Z indices
	k3=(float)k;
	
	float kk=k1*k1+k2*k2+k3*k3;

	int h=i*NY*NZ+j*NZ+k;
	
	//if(i<NXSIZE &&  j<NY && k<NZ )
        if( ind < (NXSIZE*NY*NZ) )
	{	


	float2 r1=ux[h];
	float2 r2=uy[h];
	float2 r3=uz[h];
	
	float2 s_prod;

	// Avoid undefined operations	

	if(kk<0.5f) kk=1.0f;
	
	// Proyection in fourier plane to fulfill continuity
	// Ri=Ri+(ki*kj)*Rj/k^2	
	
	s_prod.x=r1.x*k1+r2.x*k2+r3.x*k3;
	s_prod.y=r1.y*k1+r2.y*k2+r3.y*k3;
	
	r1.x=r1.x-s_prod.x*k1/kk;
	r1.y=r1.y-s_prod.y*k1/kk;
	
	r2.x=r2.x-s_prod.x*k2/kk;
	r2.y=r2.y-s_prod.y*k2/kk;
	
	r3.x=r3.x-s_prod.x*k3/kk;
	r3.y=r3.y-s_prod.y*k3/kk;

	if(i+IGLOBAL+j+k==0){

	r1.x=0.0f;
	r1.y=0.0f;
	
	r2.x=0.0f;
	r2.y=0.0f;

	r3.x=0.0f;
	r3.y=0.0f;	
	
	}

	ux[h]=r1;
	uy[h]=r2;
	uz[h]=r3;

	
	}


}

static void __global__ zeroKernel(float2* ux,int IGLOBAL,int NXSIZE)
{
	
        int ind  = blockIdx.x * blockDim.x + threadIdx.x;

        int  k = ind%NZ;
        int  i = ind/(NZ*NY);
        int  j = ind/NZ-i*NY;

	int h=i*NY*NZ+j*NZ+k;	
	
	//if(i<NXSIZE &&  j<NY && k<NZ )
        if( ind < (NXSIZE*NY*NZ) )
	{	


	float2 zero;
	zero.x=0.0f;
	zero.y=0.0f;

	ux[h]=zero;


	
	}


}



static dim3 threadsPerBlock;
static dim3 blocksPerGrid;


// Functino to turn to zero all those modes dealiased

extern void dealias(vectorField t)
{
	

        int elements = NXSIZE*NY*NZ;

        threadsPerBlock.x=THREADSPERBLOCK_NU;

        blocksPerGrid.x=(elements+threadsPerBlock.x-1)/threadsPerBlock.x;


	
	dealias_kernel<<<blocksPerGrid,threadsPerBlock,0,compute_stream>>>(t.x,t.y,t.z,IGLOBAL,NXSIZE);
	kernelCheck(RET,"dealias kern",1);
	
	return;
}

extern void projectFourier(vectorField u)
{


        int elements = NXSIZE*NY*NZ;

        // Operate over N*N*(N/2+1) matrix
        threadsPerBlock.x=THREADSPERBLOCK_NU;

        blocksPerGrid.x=(elements+threadsPerBlock.x-1)/threadsPerBlock.x;
	

	
	projectionKernel<<<blocksPerGrid,threadsPerBlock,0,compute_stream>>>(u.x,u.y,u.z,IGLOBAL,NXSIZE);	
	kernelCheck(RET,"projection Kern",1);
}

extern void set2zero(float2* u)
{

        int elements = NXSIZE*NY*NZ;

        // Operate over N*N*(N/2+1) matrix
        threadsPerBlock.x=THREADSPERBLOCK_NU;

        blocksPerGrid.x=(elements+threadsPerBlock.x-1)/threadsPerBlock.x;

	
	zeroKernel<<<blocksPerGrid,threadsPerBlock,0,compute_stream>>>(u,IGLOBAL,NXSIZE);	
	kernelCheck(RET,"zero kern",1);
}

