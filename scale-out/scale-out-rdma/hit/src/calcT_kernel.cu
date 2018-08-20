#include "turH.h"

// Normalizatino kernel

static __global__ void normalize_kernel(float2* ux,float2* uy,float2* uz,int IGLOBAL,int NXSIZE)
{
	

	int i  = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	
	int k=j%NZ;
	j=(j-k)/NZ;

	if (i<NXSIZE && j<NY && k<NZ)
	{

	int h=i*NY*NZ+j*NZ+k;

	float N3=(float)N*N*N;

	// Read {u1,u2,u3}	
	
	float2 u1=ux[h];
	float2 u2=uy[h];
	float2 u3=uz[h];
	
	u1.x/=N3;
	u1.y/=N3;

	u2.x/=N3;
	u2.y/=N3;

	u3.x/=N3;
	u3.y/=N3;


	ux[h]=u1;
	uy[h]=u2;
	uz[h]=u3;
		
	}
		
}


// Calculate uu normalizing

static __global__ void calcUU_kernel(float2* uxx,float2* uyy,float2* uzz,int IGLOBAL,int NXSIZE)
{
	


	int i  = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	
	int k=j%NZ;
	j=(j-k)/NZ;

	if (i<NXSIZE && j<NY && k<NZ)
	{

	int h=i*NY*NZ+j*NZ+k;

	float N3=(float)N*N*N;

	// Read {u1,u2,u3}	
	
	float2 u1=uxx[h];
	float2 u2=uyy[h];
	float2 u3=uzz[h];
	
	float2 u11;
	float2 u22;
	float2 u33;



	u11.x= u1.x*u1.x/(N3*N3);
	u11.y= u1.y*u1.y/(N3*N3);

	u22.x= u2.x*u2.x/(N3*N3);
	u22.y= u2.y*u2.y/(N3*N3);

	u33.x= u3.x*u3.x/(N3*N3);
	u33.y= u3.y*u3.y/(N3*N3);


	uxx[h]=u11;
	uyy[h]=u22;
	uzz[h]=u33;
		
	}
	
	
	
}

static __global__ void calcUV_kernel(float2* uxx,float2* uyy,float2* uzz,int IGLOBAL,int NXSIZE)
{
	

	int i  = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	
	int k=j%NZ;
	j=(j-k)/NZ;

	if (i<NXSIZE && j<NY && k<NZ)
	{

	int h=i*NY*NZ+j*NZ+k;

	float N3=(float)N*N*N;

	// Read {u1,u2,u3}	
	
	float2 u1=uxx[h];
	float2 u2=uyy[h];
	float2 u3=uzz[h];
	
	float2 u11;
	float2 u22;
	float2 u33;


	u11.x= u1.x*u2.x/(N3*N3);
	u11.y= u1.y*u2.y/(N3*N3);

	u22.x= u2.x*u3.x/(N3*N3);
	u22.y= u2.y*u3.y/(N3*N3);

	u33.x= u3.x*u1.x/(N3*N3);
	u33.y= u3.y*u1.y/(N3*N3);


	uxx[h]=u11;
	uyy[h]=u22;
	uzz[h]=u33;
		
	}
	
	
	
}
// Calculate Sii and normalize

static __global__ void calcSii_kernel(float2* ux,float2* uy,float2* uz,int IGLOBAL,int NXSIZE)
{
	
	int i  = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	
	int k=j%NZ;
	j=(j-k)/NZ;

	if (i<NXSIZE && j<NY && k<NZ)
	{

	int h=i*NY*NZ+j*NZ+k;

	float N3=(float)N*N*N;
	

	// Read {u1,u2,u3}	
	float k1;
	float k2;
	float k3;
	
	// X indices		
	k1=(i+IGLOBAL)<NX/2 ? (float)(i+IGLOBAL) : (float)(i+IGLOBAL)-(float)NX ;

	// Y indice
	k2=j<NY/2 ? (float)j : (float)j-(float)NY ;
	
	// Z indices
	k3=(float)k;


	float2 u1=ux[h];
	float2 u2=uy[h];
	float2 u3=uz[h];
	
	
	float2 u11;
	float2 u22;
	float2 u33;
	
	u11.x=-k1*u1.y/N3;
	u11.y= k1*u1.x/N3;
		
	u22.x=-k2*u2.y/N3;
	u22.y= k2*u2.x/N3;

	u33.x=-k3*u3.y/N3;
	u33.y= k3*u3.x/N3;

	ux[h]=u11;
	uy[h]=u22;
	uz[h]=u33;
		
	
	}
	
	
	
}


// Calculate Sij normalizing

static __global__ void calcSij_kernel(float2* ux,float2* uy,float2* uz,int IGLOBAL,int NXSIZE)
{
	
	int i  = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	
	int k=j%NZ;
	j=(j-k)/NZ;

	if (i<NXSIZE && j<NY && k<NZ)
	{

	int h=i*NY*NZ+j*NZ+k;

	float N3=(float)N*N*N;
	

	// Read {u1,u2,u3}	
	float k1;
	float k2;
	float k3;
	
	// X indices		
	k1=(i+IGLOBAL)<NX/2 ? (float)(i+IGLOBAL) : (float)(i+IGLOBAL)-(float)NX ;

	// Y indice
	k2=j<NY/2 ? (float)j : (float)j-(float)NY ;
	
	// Z indices
	k3=(float)k;


	float2 u1=ux[h];
	float2 u2=uy[h];
	float2 u3=uz[h];

	
	float2 u11;
	float2 u22;
	float2 u33;
	
	u11.x=0.5f*(-k2*u1.y-k1*u2.y)/N3;
	u11.y=0.5f*( k2*u1.x+k1*u2.x)/N3;
		
	u22.x=0.5f*(-k3*u2.y-k2*u3.y)/N3;
	u22.y=0.5f*( k3*u2.x+k2*u3.x)/N3;

	u33.x=0.5f*(-k1*u3.y-k3*u1.y)/N3;
	u33.y=0.5f*( k1*u3.x+k3*u1.x)/N3;

	ux[h]=u11;
	uy[h]=u22;
	uz[h]=u33;
		

	}
	
	
	
}


static __global__ void calc_dTau_kernel(float2* ux,float2* uy,float2* uz,int mode,int IGLOBAL,int NXSIZE)
{
	
	int i  = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	
	int k=j%NZ;
	j=(j-k)/NZ;

	if (i<NXSIZE && j<NY && k<NZ)
	{

	int h=i*NY*NZ+j*NZ+k;

	float N3=(float)N*N*N;
	

	// Read {u1,u2,u3}	
	float k1;
	float k2;
	float k3;
	
	// X indices		
	k1=(i+IGLOBAL)<NX/2 ? (float)(i+IGLOBAL) : (float)(i+IGLOBAL)-(float)NX ;

	// Y indice
	k2=j<NY/2 ? (float)j : (float)j-(float)NY ;
	
	// Z indices
	k3=(float)k;


	float2 u1=ux[h];
	float2 u2=uy[h];
	float2 u3=uz[h];

	
	float2 u11;
	float2 u22;
	float2 u33;

	if(mode==0){	

	u11.x=(-k1*u1.y)/N3;
	u11.y=( k1*u1.x)/N3;
		
	u22.x=(-k2*u2.y)/N3;
	u22.y=( k2*u2.x)/N3;

	u33.x=(-k3*u3.y)/N3;
	u33.y=( k3*u3.x)/N3;

	}else if(mode==1){
	
	u11.x=(-k1*u1.y)/N3;
	u11.y=( k1*u1.x)/N3;
		
	u22.x=(-k2*u2.y)/N3;
	u22.y=( k2*u2.x)/N3;

	u33.x=(-k3*u3.y)/N3;
	u33.y=( k3*u3.x)/N3;

	}else if(mode==2){
	
	u11.x=(-k2*u1.y)/N3;
	u11.y=( k2*u1.x)/N3;
		
	u22.x=(-k3*u2.y)/N3;
	u22.y=( k3*u2.x)/N3;

	u33.x=(-k1*u3.y)/N3;
	u33.y=( k1*u3.x)/N3;


	}

	ux[h]=u11;
	uy[h]=u22;
	uz[h]=u33;
		
	
	}
	
	
	
}

static __global__ void gaussFilter_kernel(float2* uxx,float2* uyy,float2* uzz,float alpha,int IGLOBAL,int NXSIZE)
{
	
	int i  = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	
	int k=j%NZ;
	j=(j-k)/NZ;

	if (i<NXSIZE && j<NY && k<NZ)
	{

	int h=i*NY*NZ+j*NZ+k;

	float N3=(float)N*N*N;
	

	// Read {u1,u2,u3}	
	float k1;
	float k2;
	float k3;
	
	// X indices		
	k1=(i+IGLOBAL)<NX/2 ? (float)(i+IGLOBAL) : (float)(i+IGLOBAL)-(float)NX ;

	// Y indice
	k2=j<NY/2 ? (float)j : (float)j-(float)NY ;
	
	// Z indices
	k3=(float)k;

	// Read {u1,u2,u3}	
	
	float2 u11=uxx[h];
	float2 u22=uyy[h];
	float2 u33=uzz[h];
	
	float kk=k1*k1+k2*k2+k3*k3;
	float D=alpha/(2.0f*acos(-1.0f));
	float g=exp(-kk*D*D);

	u11.x*=g; 
	u11.y*=g; 

	u22.x*=g;
	u22.y*=g;

	u33.x*=g;
	u33.y*=g;


	uxx[h]=u11;
	uyy[h]=u22;
	uzz[h]=u33;
	
	
	
	}
	
	
}

static __global__ void gaussFilter_High_kernel(float2* uxx,float2* uyy,float2* uzz,float alpha,int IGLOBAL,int NXSIZE)
{
	
	int i  = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	
	int k=j%NZ;
	j=(j-k)/NZ;

	if (i<NXSIZE && j<NY && k<NZ)
	{

	int h=i*NY*NZ+j*NZ+k;

	float N3=(float)N*N*N;
	

	// Read {u1,u2,u3}	
	float k1;
	float k2;
	float k3;
	
	// X indices		
	k1=(i+IGLOBAL)<NX/2 ? (float)(i+IGLOBAL) : (float)(i+IGLOBAL)-(float)NX ;

	// Y indice
	k2=j<NY/2 ? (float)j : (float)j-(float)NY ;
	
	// Z indices
	k3=(float)k;

	// Read {u1,u2,u3}	
	
	float2 u11=uxx[h];
	float2 u22=uyy[h];
	float2 u33=uzz[h];
	
	float kk=k1*k1+k2*k2+k3*k3;
	float D=alpha/(2.0f*acos(-1.0f));
	float g=exp(-kk*D*D);

	u11.x*=(1.0f-g); 
	u11.y*=(1.0f-g); 

	u22.x*=(1.0f-g);
	u22.y*=(1.0f-g);

	u33.x*=(1.0f-g);
	u33.y*=(1.0f-g);


	uxx[h]=u11;
	uyy[h]=u22;
	uzz[h]=u33;
	
	
	
	}
	
	
}

static __global__ void calcL_kernel(float2* A1,float2* A2,float2* A3,float2* B1,float2* B2,float2* B3,int IGLOBAL,int NXSIZE)
{
	

	int i  = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	
	int k=j%NZ;
	j=(j-k)/NZ;

	if (i<NXSIZE && j<NY && k<NZ)
	{

	int h=i*NY*NZ+j*NZ+k;

	float N3=(float)N*N*N;
	

	// Read {u1,u2,u3}	
	float k1;
	float k2;
	float k3;
	
	// X indices		
	k1=(i+IGLOBAL)<NX/2 ? (float)(i+IGLOBAL) : (float)(i+IGLOBAL)-(float)NX ;

	// Y indice
	k2=j<NY/2 ? (float)j : (float)j-(float)NY ;
	
	// Z indices
	k3=(float)k;

	
	float2 L11=B1[h];
	float2 L12=B2[h];
	float2 L13=B3[h];
	
	float2 L21=A1[h];
	float2 L22=A2[h];
	float2 L23=A3[h];

	
	L11.x= L11.x-L21.x;
	L11.y= L11.y-L21.y;
	
	L12.x= L12.x-L22.x;
	L12.y= L12.y-L22.y;
	
	L13.x= L13.x-L23.x;
	L13.y= L13.y-L23.y;



	A1[h]=L11;
	A2[h]=L12;
	A3[h]=L13;

	}
	
	
	
}



static __global__ void calc_tauS_kernel(float2* OUT,float2* Ax,float2* Ay,float2* Az,float2* Bx,float2* By,float2* Bz,int IGLOBAL,int NXSIZE)
{
	
	int i  = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	
	int k=j%NZ;
	j=(j-k)/NZ;

	if (i<NXSIZE && j<NY && k<NZ)
	{

	int h=i*NY*NZ+j*NZ+k;

	float N3=(float)N*N*N;
	

	// Read {u1,u2,u3}	
	float k1;
	float k2;
	float k3;
	
	// X indices		
	k1=(i+IGLOBAL)<NX/2 ? (float)(i+IGLOBAL) : (float)(i+IGLOBAL)-(float)NX ;

	// Y indice
	k2=j<NY/2 ? (float)j : (float)j-(float)NY ;
	
	// Z indices
	k3=(float)k;

	float2 L1=Ax[h];
	float2 L2=Ay[h];
	float2 L3=Az[h];

	float2 M1=Bx[h];
	float2 M2=By[h];
	float2 M3=Bz[h];
	
	float2 lm;

	lm.x= L1.x*M1.x+L2.x*M2.x+L3.x*M3.x;
	lm.y= L1.y*M1.y+L2.y*M2.y+L3.y*M3.y;
		
	OUT[h]=lm;


	}
		
}



// Calc Strain tensor and module of Strain tensor kernel

static dim3 threadsPerBlock;
static dim3 blocksPerGrid;

extern void normalize(vectorField A){

	threadsPerBlock.x=THREADSPERBLOCK_IN;
	threadsPerBlock.y=THREADSPERBLOCK_IN;

	blocksPerGrid.x=NXSIZE/threadsPerBlock.x;
	blocksPerGrid.y=NY*NZ/threadsPerBlock.y;


	//High pass filter
	normalize_kernel<<<blocksPerGrid,threadsPerBlock>>>(A.x,A.y,A.z,IGLOBAL,NXSIZE);
	kernelCheck(RET,"rk_initstep",1);

	return;

}

extern void gaussFilter_High(vectorField A,float alpha){

	threadsPerBlock.x=THREADSPERBLOCK_IN;
	threadsPerBlock.y=THREADSPERBLOCK_IN;

	blocksPerGrid.x=NXSIZE/threadsPerBlock.x;
	blocksPerGrid.y=NY*NZ/threadsPerBlock.y;


	//High pass filter
	gaussFilter_High_kernel<<<blocksPerGrid,threadsPerBlock>>>(A.x,A.y,A.z,alpha,IGLOBAL,NXSIZE);
	kernelCheck(RET,"rk_initstep",1);

	return;

}

extern void gaussFilter(vectorField B,float alpha){

	threadsPerBlock.x=THREADSPERBLOCK_IN;
	threadsPerBlock.y=THREADSPERBLOCK_IN;

	blocksPerGrid.x=NXSIZE/threadsPerBlock.x;
	blocksPerGrid.y=NY*NZ/threadsPerBlock.y;

	//Low pass filter
	gaussFilter_kernel<<<blocksPerGrid,threadsPerBlock>>>(B.x,B.y,B.z,alpha,IGLOBAL,NXSIZE);
	kernelCheck(RET,"rk_initstep",1);

	return;
}
	

extern void calcUU(vectorField A,int d){

	threadsPerBlock.x=THREADSPERBLOCK_IN;
	threadsPerBlock.y=THREADSPERBLOCK_IN;

	blocksPerGrid.x=NXSIZE/threadsPerBlock.x;
	blocksPerGrid.y=NY*NZ/threadsPerBlock.y;

	if(d==0){
	calcUU_kernel<<<blocksPerGrid,threadsPerBlock>>>(A.x,A.y,A.z,IGLOBAL,NXSIZE);	
	kernelCheck(RET,"rk_initstep",1);
	}else{
	calcUV_kernel<<<blocksPerGrid,threadsPerBlock>>>(A.x,A.y,A.z,IGLOBAL,NXSIZE);
	kernelCheck(RET,"rk_initstep",1);
	}	

	return;

}

extern void calcS(vectorField A,int d){

	threadsPerBlock.x=THREADSPERBLOCK_IN;
	threadsPerBlock.y=THREADSPERBLOCK_IN;

	blocksPerGrid.x=NXSIZE/threadsPerBlock.x;
	blocksPerGrid.y=NY*NZ/threadsPerBlock.y;

	if(d==0){
	calcSii_kernel<<<blocksPerGrid,threadsPerBlock>>>(A.x,A.y,A.z,IGLOBAL,NXSIZE);	
	kernelCheck(RET,"rk_initstep",1);
	}else{
	calcSij_kernel<<<blocksPerGrid,threadsPerBlock>>>(A.x,A.y,A.z,IGLOBAL,NXSIZE);
	kernelCheck(RET,"rk_initstep",1);
	}

	return;

}

extern void calc_tauS_cuda(float2* OUT,vectorField A,vectorField B,int d){

	threadsPerBlock.x=THREADSPERBLOCK_IN;
	threadsPerBlock.y=THREADSPERBLOCK_IN;

	blocksPerGrid.x=NXSIZE/threadsPerBlock.x;
	blocksPerGrid.y=NY*NZ/threadsPerBlock.y;

	if(d==0){
	calc_tauS_kernel<<<blocksPerGrid,threadsPerBlock>>>(OUT,A.x,A.y,A.z,B.x,B.y,B.z,IGLOBAL,NXSIZE);
	kernelCheck(RET,"rk_initstep",1);
	}else if(d==1){
	calc_tauS_kernel<<<blocksPerGrid,threadsPerBlock>>>(OUT,A.x,A.y,A.z,B.y,B.z,B.x,IGLOBAL,NXSIZE);
	kernelCheck(RET,"rk_initstep",1);	
	}else if(d==2){
	calc_tauS_kernel<<<blocksPerGrid,threadsPerBlock>>>(OUT,A.x,A.y,A.z,B.x,B.y,B.z,IGLOBAL,NXSIZE);
	kernelCheck(RET,"rk_initstep",1);	
	}

	return;

}

extern void calcL(vectorField A,vectorField B){

	threadsPerBlock.x=THREADSPERBLOCK_IN;
	threadsPerBlock.y=THREADSPERBLOCK_IN;

	blocksPerGrid.x=NXSIZE/threadsPerBlock.x;
	blocksPerGrid.y=NY*NZ/threadsPerBlock.y;


	calcL_kernel<<<blocksPerGrid,threadsPerBlock>>>(A.x,A.y,A.z,B.x,B.y,B.z,IGLOBAL,NXSIZE);

	return;

}

extern void calc_dTau(vectorField A,int mode){

	threadsPerBlock.x=THREADSPERBLOCK_IN;
	threadsPerBlock.y=THREADSPERBLOCK_IN;

	blocksPerGrid.x=NXSIZE/threadsPerBlock.x;
	blocksPerGrid.y=NY*NZ/threadsPerBlock.y;


	calc_dTau_kernel<<<blocksPerGrid,threadsPerBlock>>>(A.x,A.y,A.z,mode,IGLOBAL,NXSIZE);
	kernelCheck(RET,"rk_initstep",1);

	return;

}


