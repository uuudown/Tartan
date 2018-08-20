#include "turH.h"


static __global__ void shift_kernel(float2* tx,float2* ty,float2* tz,float Delta_1,float Delta_2,float Delta_3,int IGLOBAL,int NXSIZE)
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
	

	int h=i*NY*NZ+j*NZ+k;
	
	if(i<NXSIZE &&  j<NY && k<NZ )
	{

	float2 t1=tx[h];
	float2 t2=ty[h];
	float2 t3=tz[h];
	
	//float aux_x;
	//float aux_y;

	// Phase shifting by Delta;

//        float sine=sin(k1*Delta_1+k2*Delta_2+k3*Delta_3);
//        float cosine=cos(k1*Delta_1+k2*Delta_2+k3*Delta_3);


	float sine, cosine;
        sincosf(k1*Delta_1+k2*Delta_2+k3*Delta_3,&sine,&cosine);
	
	//t1;

        float2 temp1;
	temp1.x=cosine*t1.x-sine*t1.y;
	temp1.y=sine*t1.x+cosine*t1.y;
        t1 = temp1;
	
	//t2;
        float2 temp2;
	temp2.x=cosine*t2.x-sine*t2.y;
        temp2.y=sine*t2.x+cosine*t2.y;	
        t2 = temp2;

	//t3	
        float2 temp3;	
	temp3.x=cosine*t3.x-sine*t3.y;
	temp3.y=sine*t3.x+cosine*t3.y;
        t3 = temp3;
	
	
	tx[h]=t1;
	ty[h]=t2;
	tz[h]=t3;



	}

}


static dim3 threadsPerBlock;
static dim3 blocksPerGrid;


extern void shift(vectorField t,float* Delta)
{


         int elements = NXSIZE*NY*NZ;

        // Operate over N*N*(N/2+1) matrix
        threadsPerBlock.x=THREADSPERBLOCK_NU;

        blocksPerGrid.x=(elements+threadsPerBlock.x-1)/threadsPerBlock.x;


	shift_kernel<<<blocksPerGrid,threadsPerBlock,0,compute_stream>>>(t.x,t.y,t.z,Delta[0],Delta[1],Delta[2],IGLOBAL,NXSIZE);
	kernelCheck(RET,"dealias",1);
	
	return;

}


