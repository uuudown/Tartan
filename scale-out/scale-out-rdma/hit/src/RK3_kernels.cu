#include "turH.h"


static void __global__ rk_step_1(float2* ux,float2* uy,float2* uz,float2* u_wx,float2* u_wy,float2* u_wz,
									float2* rx, float2* ry, float2* rz,float Re,float dt,float Cf,int kf,int IGLOBAL,int NXSIZE,int nc)
{

	const float alpha[]={ 29.0f/96.0f, -3.0f/40.0f, 1.0f/6.0f};
	const float dseda[]={ 0.0f, -17.0f/60.0f, -5.0f/12.0f};
	

		
        int ind  = blockIdx.x * blockDim.x + threadIdx.x;

        int  k = ind%NZ;
        int  i = ind/(NZ*NY);
        int  j = ind/NZ-i*NY;

	float k1;
	float k2;
	float k3;

	float kk;
	float lap;

	float2 s_prod;
	
	//if (i<NXSIZE && j<NY && k<NZ)
        if( ind < (NXSIZE*NY*NZ) )	
	{
	
	// X indices		
	k1=(i+IGLOBAL)<NX/2 ? (float)(i+IGLOBAL) : (float)(i+IGLOBAL)-(float)NX ;

	// Y indice
	k2=j<NY/2 ? (float)j : (float)j-(float)NY ;
	
	// Z indices
	k3=(float)k;

	// Wave numbers

	kk=k1*k1+k2*k2+k3*k3;
	lap=-kk/Re;
	
	if(kk<kf*kf){lap=lap+Cf;}
	
	int h=i*NY*NZ+j*NZ+k;
	
	float2 u1;
	float2 u2;
	float2 u3;
	
	float2 r1;
	float2 r2;
	float2 r3;

	float2 u_w1;
	float2 u_w2;
	float2 u_w3;

	// Read u and R from global memory

	u1=ux[h];
	u2=uy[h];
	u3=uz[h];
	
	r1=rx[h];
	r2=ry[h];
	r3=rz[h];

	//Project r

	if(kk<0.5f)kk=1.0f;
		
	s_prod.x=r1.x*k1+r2.x*k2+r3.x*k3;
	s_prod.y=r1.y*k1+r2.y*k2+r3.y*k3;
	
	r1.x=r1.x-s_prod.x*k1/kk;
	r1.y=r1.y-s_prod.y*k1/kk;
	
	r2.x=r2.x-s_prod.x*k2/kk;
	r2.y=r2.y-s_prod.y*k2/kk;
	
	r3.x=r3.x-s_prod.x*k3/kk;
	r3.y=r3.y-s_prod.y*k3/kk;	
		

	// u_1=u_0+betha(0))*Lu+dseda*R
	
	u_w1.x=u1.x+dt*(alpha[nc]*lap*u1.x+dseda[nc]*r1.x);
	u_w1.y=u1.y+dt*(alpha[nc]*lap*u1.y+dseda[nc]*r1.y);
	
	u_w2.x=u2.x+dt*(alpha[nc]*lap*u2.x+dseda[nc]*r2.x);
	u_w2.y=u2.y+dt*(alpha[nc]*lap*u2.y+dseda[nc]*r2.y);
	
	u_w3.x=u3.x+dt*(alpha[nc]*lap*u3.x+dseda[nc]*r3.x);
	u_w3.y=u3.y+dt*(alpha[nc]*lap*u3.y+dseda[nc]*r3.y);
	
	//Write to global memory
	
	u_wx[h]=u_w1;
	u_wy[h]=u_w2;
	u_wz[h]=u_w3;
	
	}

	
	
}

static void __global__ rk_step_2(float2* ux,float2* uy,float2* uz,float2* u_wx,float2* u_wy,float2* u_wz,
									float2* rx, float2* ry, float2* rz,float Re,float dt,float Cf,int kf,int IGLOBAL,int NXSIZE,int nc)
{
	const float betha[]={ 37.0f/160.0f, 5.0f/24.0f, 1.0f/6.0f};
	const float gammad[]={ 8.0f/15.0f, 5.0f/12.0f, 3.0f/4.0f};
	
		
         int ind  = blockIdx.x * blockDim.x + threadIdx.x;


        int  k = ind%NZ;
        int  i = ind/(NZ*NY);
        int  j = ind/NZ-i*NY;


	float k1;
	float k2;
	float k3;

	float kk;
	float lap;
	
	float2 s_prod;
	
//	if (i<NXSIZE && j<NY && k<NZ)
         if( ind < (NXSIZE*NY*NZ) )
	{
	
	// X indices		
	k1=(i+IGLOBAL)<NX/2 ? (float)(i+IGLOBAL) : (float)(i+IGLOBAL)-(float)NX ;

	// Y indice
	k2=j<NY/2 ? (float)j : (float)j-(float)NY ;
	
	// Z indices
	k3=(float)k;

	// Wave numbers

	kk=k1*k1+k2*k2+k3*k3;

	lap=-kk/Re;
	if(kk<kf*kf){lap=lap+Cf;}	


	int h=i*NY*NZ+j*NZ+k;
	
	float2 r1;
	float2 r2;
	float2 r3;

	float2 u_w1;
	float2 u_w2;
	float2 u_w3;


	// Read u and R from global memory

	u_w1=u_wx[h];
	u_w2=u_wy[h];
	u_w3=u_wz[h];

	r1=rx[h];
	r2=ry[h];
	r3=rz[h];

	//Project r

	if(kk<0.5f)kk=1.0f;
		
	s_prod.x=r1.x*k1+r2.x*k2+r3.x*k3;
	s_prod.y=r1.y*k1+r2.y*k2+r3.y*k3;
	
	r1.x=r1.x-s_prod.x*k1/kk;
	r1.y=r1.y-s_prod.y*k1/kk;
	
	r2.x=r2.x-s_prod.x*k2/kk;
	r2.y=r2.y-s_prod.y*k2/kk;
	
	r3.x=r3.x-s_prod.x*k3/kk;
	r3.y=r3.y-s_prod.y*k3/kk;		

	// u=(alpha(0)+betha(0))*Lu+dseda*R
	
	u_w1.x=u_w1.x+dt*(gammad[nc]*r1.x);
	u_w1.y=u_w1.y+dt*(gammad[nc]*r1.y);
	
	u_w2.x=u_w2.x+dt*(gammad[nc]*r2.x);
	u_w2.y=u_w2.y+dt*(gammad[nc]*r2.y);
	
	u_w3.x=u_w3.x+dt*(gammad[nc]*r3.x);
	u_w3.y=u_w3.y+dt*(gammad[nc]*r3.y);

	//Implicit step

	float Imp=1.0f/(1.0f-betha[nc]*dt*lap);

	u_w1.x=u_w1.x*Imp;
	u_w1.y=u_w1.y*Imp;

	u_w2.x=u_w2.x*Imp;
	u_w2.y=u_w2.y*Imp;

	u_w3.x=u_w3.x*Imp;
	u_w3.y=u_w3.y*Imp;
	

	//Write to global memory
	
	ux[h]=u_w1;
	uy[h]=u_w2;
	uz[h]=u_w3;
	
	}

	
}


static dim3 threadsPerBlock;
static dim3 blocksPerGrid;


extern void RK3_step_1(vectorField u,vectorField uw,vectorField r,float Re,float dt,float Cf,int kf,int nc)
{

        int elements = NXSIZE*NY*NZ;

        // Operate over N*N*(N/2+1) matrix
        threadsPerBlock.x=THREADSPERBLOCK_NU;

        blocksPerGrid.x=(elements+threadsPerBlock.x-1)/threadsPerBlock.x;

	rk_step_1<<<blocksPerGrid,threadsPerBlock,0,compute_stream>>>(u.x,u.y,u.z,uw.x,uw.y,uw.z,r.x,r.y,r.z,Re,dt,Cf,kf,IGLOBAL,NXSIZE,nc);
	kernelCheck(RET,"rk_initstep",1);
	
	return;
}



extern void RK3_step_2(vectorField u,vectorField uw,vectorField r,float Re,float dt,float Cf,int kf,int nc)
{
        int elements = NXSIZE*NY*NZ;

        // Operate over N*N*(N/2+1) matrix
        threadsPerBlock.x=THREADSPERBLOCK_NU;

        blocksPerGrid.x=(elements+threadsPerBlock.x-1)/threadsPerBlock.x;

	rk_step_2<<<blocksPerGrid,threadsPerBlock,0,compute_stream>>>(u.x,u.y,u.z,uw.x,uw.y,uw.z,r.x,r.y,r.z,Re,dt,Cf,kf,IGLOBAL,NXSIZE,nc);
	kernelCheck(RET,"rk_substep",1);

	return;
}




