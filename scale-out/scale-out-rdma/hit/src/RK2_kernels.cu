#include "turH.h"


static void __global__ rk_substep_1(float2* ux,float2* uy,float2* uz,float2* u_wx,float2* u_wy,float2* u_wz,
					float2* rx, float2* ry, float2* rz,float Re,float dt,float Cf,int kf,int IGLOBAL,int NXSIZE)
{

	int j  = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;
		
	int k=j%NZ;
	j=(j-k)/NZ;
		
	float k1;
	float k2;
	float k3;

	float kk;
	float lap;
	
	float2 s_prod;
	
	float2 u1;
	float2 u2;
	float2 u3;
	
	float2 r1;
	float2 r2;
	float2 r3;

	float2 u_w1;
	float2 u_w2;
	float2 u_w3;



	if (i<NXSIZE && j<NY && k<NZ)
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
	

	int h=i*NY*NZ+j*NZ+k;
	
	u1=ux[h];
	u2=uy[h];
	u3=uz[h];

	r1=rx[h];
	r2=ry[h];
	r3=rz[h];

	if(kk<0.5f)kk=1.0f;

	//Forcing

	
	kk=k1*k1+k2*k2+k3*k3;
	lap=-kk/Re;
	
	if(kk<kf*kf){lap=lap+Cf;}

	s_prod.x=r1.x*k1+r2.x*k2+r3.x*k3;
	s_prod.y=r1.y*k1+r2.y*k2+r3.y*k3;
	
	r1.x=r1.x-s_prod.x*k1/kk;
	r1.y=r1.y-s_prod.y*k1/kk;
	
	r2.x=r2.x-s_prod.x*k2/kk;
	r2.y=r2.y-s_prod.y*k2/kk;
	
	r3.x=r3.x-s_prod.x*k3/kk;
	r3.y=r3.y-s_prod.y*k3/kk;
	
	// u_w=dt*(u+gamma(i)*R
	
	u_w1.x=u1.x+0.5f*dt*(r1.x+lap*u1.x);
	u_w1.y=u1.y+0.5f*dt*(r1.y+lap*u1.y);
	
	u_w2.x=u2.x+0.5f*dt*(r2.x+lap*u2.x);
	u_w2.y=u2.y+0.5f*dt*(r2.y+lap*u2.y);
	
	u_w3.x=u3.x+0.5f*dt*(r3.x+lap*u3.x);
	u_w3.y=u3.y+0.5f*dt*(r3.y+lap*u3.y);

	u_wx[h]=u_w1;
	u_wy[h]=u_w2;
	u_wz[h]=u_w3;
	
	
	}
	

}


static void __global__ rk_substep_05(float2* ux,float2* uy,float2* uz,float2* u_wx,float2* u_wy,float2* u_wz,
					float Re,float dt,float Cf,int kf,int IGLOBAL,int NXSIZE)
{


	int j  = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;
		

	int k=j%NZ;
	j=(j-k)/NZ;
		
	float k1;
	float k2;
	float k3;

	float kk;
	float lap;
	
	float2 u1;
	float2 u2;
	float2 u3;


	float2 u_w1;
	float2 u_w2;
	float2 u_w3;




	if (i<NXSIZE && j<NY && k<NZ)
	{

	
	// X indices in global		
	k1=(i+IGLOBAL)<NX/2 ? (float)(i+IGLOBAL) : (float)(i+IGLOBAL)-(float)NX ;

	// Y indice
	k2=j<NY/2 ? (float)j : (float)j-(float)NY ;
	
	// Z indices
	k3=(float)k;

	// Wave numbers

	kk=k1*k1+k2*k2+k3*k3;
	lap=-kk/Re;
	
	int h=i*NY*NZ+j*NZ+k;
	
	u1=ux[h];
	u2=uy[h];
	u3=uz[h];

	u_w1=u_wx[h];
	u_w2=u_wy[h];
	u_w3=u_wz[h];

	//Forcing

	kk=k1*k1+k2*k2+k3*k3;
	lap=-kk/Re;
	
	if(kk<kf*kf){lap=lap+Cf;}

	// u_w=dt*(u+gamma(i)*R
		
	u_w1.x=u1.x+dt*(lap*u_w1.x);
	u_w1.y=u1.y+dt*(lap*u_w1.y);
	
	u_w2.x=u2.x+dt*(lap*u_w2.x);
	u_w2.y=u2.y+dt*(lap*u_w2.y);
	
	u_w3.x=u3.x+dt*(lap*u_w3.x);
	u_w3.y=u3.y+dt*(lap*u_w3.y);

	ux[h]=u_w1;
	uy[h]=u_w2;
	uz[h]=u_w3;
	
	
	}
	

}

static void __global__ rk_substep_2(float2* ux,float2* uy,float2* uz,float2* rx, float2* ry, float2* rz,
					float Re,float dt,float Cf,int kf,int IGLOBAL,int NXSIZE)
{

	int j  = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;
		

	int k=j%NZ;
	j=(j-k)/NZ;
		
	float k1;
	float k2;
	float k3;

	float kk;

	float2 s_prod;
	
	float2 u1;
	float2 u2;
	float2 u3;
	
	float2 r1;
	float2 r2;
	float2 r3;


	if (i<NXSIZE && j<NY && k<NZ)
	{
	
	// X indices		
	k1=(i+IGLOBAL)<NX/2 ? (float)(i+IGLOBAL) : (float)(i+IGLOBAL)-(float)NX ;

	// Y indice
	k2=j<NY/2 ? (float)j : (float)j-(float)NY ;
	
	// Z indices
	k3=(float)k;

	// Wave numbers

	kk=k1*k1+k2*k2+k3*k3;
	
	int h=i*NY*NZ+j*NZ+k;
	
	u1=ux[h];
	u2=uy[h];
	u3=uz[h];

	r1=rx[h];
	r2=ry[h];
	r3=rz[h];
	
	

	if(kk<0.5f)kk=1.0f;
		
	s_prod.x=r1.x*k1+r2.x*k2+r3.x*k3;
	s_prod.y=r1.y*k1+r2.y*k2+r3.y*k3;
	
	r1.x=r1.x-s_prod.x*k1/kk;
	r1.y=r1.y-s_prod.y*k1/kk;
	
	r2.x=r2.x-s_prod.x*k2/kk;
	r2.y=r2.y-s_prod.y*k2/kk;
	
	r3.x=r3.x-s_prod.x*k3/kk;
	r3.y=r3.y-s_prod.y*k3/kk;	
	
	// u_w=dt*(u+gamma(i)*R
	
	u1.x=u1.x+dt*(r1.x);
	u1.y=u1.y+dt*(r1.y);
	
	u2.x=u2.x+dt*(r2.x);
	u2.y=u2.y+dt*(r2.y);
	
	u3.x=u3.x+dt*(r3.x);
	u3.y=u3.y+dt*(r3.y);

	ux[h]=u1;
	uy[h]=u2;
	uz[h]=u3;
	
	
	}
	

}

static dim3 threadsPerBlock;
static dim3 blocksPerGrid;




extern void RK2_step_1(vectorField uw,vectorField u,vectorField r,float Re,float dt,float Cf,int kf)
{
	
		
	threadsPerBlock.x=THREADSPERBLOCK_IN;
	threadsPerBlock.y=THREADSPERBLOCK_IN;

	blocksPerGrid.y=NXSIZE/threadsPerBlock.x;
	blocksPerGrid.x=NY*NZ/threadsPerBlock.y;

	rk_substep_1<<<blocksPerGrid,threadsPerBlock>>>(u.x,u.y,u.z,uw.x,uw.y,uw.z,r.x,r.y,r.z,Re,dt,Cf,kf,IGLOBAL,NXSIZE);
	kernelCheck(RET,"rk_initstep",1);
	
	return;
}




extern void RK2_step_05(vectorField u,vectorField uw,float Re,float dt,float Cf,int kf)
{
		
	threadsPerBlock.x=THREADSPERBLOCK_IN;
	threadsPerBlock.y=THREADSPERBLOCK_IN;

	blocksPerGrid.y=NXSIZE/threadsPerBlock.x;
	blocksPerGrid.x=NY*NZ/threadsPerBlock.y;

	rk_substep_05<<<blocksPerGrid,threadsPerBlock>>>(u.x,u.y,u.z,uw.x,uw.y,uw.z,Re,dt,Cf,kf,IGLOBAL,NXSIZE);
	kernelCheck(RET,"rk_initstep",1);
	
	return;
}


extern void RK2_step_2(vectorField uw,vectorField r,float Re,float dt,float Cf,int kf)
{

		
	threadsPerBlock.x=THREADSPERBLOCK_IN;
	threadsPerBlock.y=THREADSPERBLOCK_IN;

	blocksPerGrid.y=NXSIZE/threadsPerBlock.x;
	blocksPerGrid.x=NY*NZ/threadsPerBlock.y;

	rk_substep_2<<<blocksPerGrid,threadsPerBlock>>>(uw.x,uw.y,uw.z,r.x,r.y,r.z,Re,dt,Cf,kf,IGLOBAL,NXSIZE);
	kernelCheck(RET,"rk_substep",1);

	return;
}




