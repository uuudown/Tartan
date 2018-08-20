#include "turH.h"

static void __global__ trans_zyx_yblock_to_yzx_kernel(float2* input, float2* output)
{
   __shared__ float2 scratch[THREADSPERBLOCK_IN][THREADSPERBLOCK_IN+1];

  int k_in = blockIdx.x*blockDim.x + threadIdx.y;//yblock_jof;//j_in;
  int j_in = blockIdx.y*blockDim.y + threadIdx.x;
  int i    = blockIdx.z;

  int yblock_dim = gridDim.z;
  int yblock_idx = k_in/yblock_dim;
  int yblock_jof = k_in%yblock_dim;

  int index_in = yblock_idx*yblock_dim*yblock_dim*NZ + i*yblock_dim*NZ + yblock_jof*NZ + j_in;

  if(k_in<NY && j_in<NZ){
    scratch[threadIdx.x][threadIdx.y] = input[index_in];
  }

  int j_ot = blockIdx.x*blockDim.x + threadIdx.x;
  int k_ot = blockIdx.y*blockDim.y + threadIdx.y;
  int index_ot = i*NY*NZ + k_ot*NY + j_ot;

  __syncthreads();

  if(j_ot<NY && k_ot<NZ){
    output[index_ot] = scratch[threadIdx.y][threadIdx.x];
  }
}


static void __global__ trans_zyx_to_zxy_kernel(float2* input, float2* output)
{
  int k_ot = blockIdx.x*blockDim.x + threadIdx.x;
  int j_ot = blockIdx.y*blockDim.y + threadIdx.y;
  int i    = blockIdx.z;
  int index_ot = j_ot*gridDim.z*NZ + i*NZ + k_ot;
  int index_in = i*NY*NZ + j_ot*NZ + k_ot;
  if(j_ot<NY && k_ot<NZ){
    output[index_ot] = input[index_in];
  }
}

static void __global__ trans_zxy_to_zyx_kernel(float2* input, float2* output)
{
  int k_in = blockIdx.x*blockDim.x + threadIdx.x;
  int j_in = blockIdx.y*blockDim.y + threadIdx.y;
  int i    = blockIdx.z;
  int index_in = j_in*gridDim.z*NZ + i*NZ + k_in;
  int index_ot = i*NY*NZ + j_in*NZ + k_in;
  if(j_in<NY && k_in<NZ){
    output[index_ot] = input[index_in];
  }
}


static void __global__ trans_zxy_to_yzx_kernel(float2* input, float2* output)
{
   __shared__ float2 scratch[THREADSPERBLOCK_IN][THREADSPERBLOCK_IN+1];
  int k_in = blockIdx.x*blockDim.x + threadIdx.x;
  int j_in = blockIdx.y*blockDim.y + threadIdx.y;
  int i    = blockIdx.z;
  int index_in = j_in*gridDim.z*NZ + i*NZ + k_in;
  if(j_in<NY && k_in<NZ){
    scratch[threadIdx.y][threadIdx.x] = input[index_in];
  }
  __syncthreads();
  int k_ot = blockIdx.y*blockDim.y + threadIdx.x;
  int j_ot = blockIdx.x*blockDim.x + threadIdx.y;
  int index_ot = i*NY*NZ + j_ot*NY + k_ot;
  if(k_ot<NY && j_ot<NZ){
    output[index_ot] = scratch[threadIdx.x][threadIdx.y];
  }
}

static void __global__ trans_zyx_to_yzx_kernel(float2* input, float2* output)
{
//   __shared__ float scratch_x[THREADSPERBLOCK_IN][THREADSPERBLOCK_IN+1];
//   __shared__ float scratch_y[THREADSPERBLOCK_IN][THREADSPERBLOCK_IN+1];

   __shared__ float2 scratch[THREADSPERBLOCK_IN][THREADSPERBLOCK_IN+1];

  int k_in = blockIdx.x*blockDim.x + threadIdx.x;
  int j_in = blockIdx.y*blockDim.y + threadIdx.y;
  int i    = blockIdx.z;
  int index_in = i*NY*NZ + j_in*NZ + k_in;
/*
  int k_ot = j_in;
  int j_ot = k_in;
  int index_ot = i*NY*NZ + j_ot*NY + k_ot;
  if(j_in<NY && k_in<NZ){
    output[index_ot] = input[index_in];
  }
*/
  if(j_in<NY && k_in<NZ){
    scratch[threadIdx.y][threadIdx.x] = input[index_in];
  }
  __syncthreads();
  int k_ot = blockIdx.y*blockDim.y + threadIdx.x;
  int j_ot = blockIdx.x*blockDim.x + threadIdx.y;
  int index_ot = i*NY*NZ + j_ot*NY + k_ot;
  if(k_ot<NY && j_ot<NZ){
    output[index_ot] = scratch[threadIdx.x][threadIdx.y];
  }

/*
  if(j_in<NY && k_in<NZ){
    float2 temp = input[index_in];
    scratch_x[threadIdx.y][threadIdx.x] = temp.x;
    scratch_y[threadIdx.y][threadIdx.x] = temp.y;
  }
  __syncthreads();
  int k_ot = blockIdx.y*blockDim.y + threadIdx.x;
  int j_ot = blockIdx.x*blockDim.x + threadIdx.y;
  int index_ot = i*NY*NZ + j_ot*NY + k_ot;
  if(k_ot<NY && j_ot<NZ){
    float2 temp;
    temp.x = scratch_x[threadIdx.x][threadIdx.y];
    temp.y = scratch_y[threadIdx.x][threadIdx.y];
    output[index_ot] = temp;
  }
*/
}

static void __global__ trans_yzx_to_zyx_kernel(float2* input, float2* output)
{
   __shared__ float2 scratch[THREADSPERBLOCK_IN][THREADSPERBLOCK_IN+1];
/*
  int j_in = blockIdx.x*blockDim.x + threadIdx.x;
  int k_in = blockIdx.y*blockDim.y + threadIdx.y;
  int i    = blockIdx.z;
  int index_in = i*NY*NZ + k_in*NY + j_in;

  int k_ot = j_in;
  int j_ot = k_in;
  int index_ot = i*NY*NZ + k_ot*NZ + j_ot;
  if(j_in<NY && k_in<NZ){
    output[index_ot] = input[index_in];
  }
*/

  int j_in = blockIdx.x*blockDim.x + threadIdx.x;
  int k_in = blockIdx.y*blockDim.y + threadIdx.y;
  int i    = blockIdx.z;
  int index_in = i*NY*NZ + k_in*NY + j_in;
  if(j_in<NY && k_in<NZ){
    scratch[threadIdx.y][threadIdx.x] = input[index_in];
  }
//  __syncthreads();

  int k_ot = blockIdx.x*blockDim.x + threadIdx.y;
  int j_ot = blockIdx.y*blockDim.y + threadIdx.x;
  int index_ot = i*NY*NZ + k_ot*NZ + j_ot;

  __syncthreads();

  if(k_ot<NY && j_ot<NZ){
    output[index_ot] = scratch[threadIdx.x][threadIdx.y];;
  }


/*
  int k_in = blockIdx.x*blockDim.x + threadIdx.x;
  int j_in = blockIdx.y*blockDim.y + threadIdx.y;
  int i    = blockIdx.z;
  int index_in = i*NY*NZ + j_in*NZ + k_in;
  if(j_in<NY && k_in<NZ){
    scratch[threadIdx.y][threadIdx.x] = input[index_in];
  }
  __syncthreads();
  int k_ot = blockIdx.y*blockDim.y + threadIdx.x;
  int j_ot = blockIdx.x*blockDim.x + threadIdx.y;
  int index_ot = i*NY*NZ + j_ot*NY + k_ot;
  if(k_ot<NY && j_ot<NZ){
    output[index_ot] = scratch[threadIdx.x][threadIdx.y];
  }
*/
}

static void __global__ trans_yzx_to_zyx_yblock_kernel(float2* input, float2* output)
{
   __shared__ float2 scratch[THREADSPERBLOCK_IN][THREADSPERBLOCK_IN+1];
/*
  int j_in = blockIdx.x*blockDim.x + threadIdx.x;
  int k_in = blockIdx.y*blockDim.y + threadIdx.y;
  int i    = blockIdx.z;
  int index_in = i*NY*NZ + k_in*NY + j_in;

  int yblock_dim = gridDim.z;
  int yblock_idx = j_in/yblock_dim;
  int yblock_jof = j_in%yblock_dim;

  int k_ot = yblock_jof;//j_in;
  int j_ot = k_in;
  int index_ot = yblock_idx*yblock_dim*yblock_dim*NZ + i*yblock_dim*NZ + k_ot*NZ + j_ot;
  if(j_in<NY && k_in<NZ){
    output[index_ot] = input[index_in];
  }
*/
  int j_in = blockIdx.x*blockDim.x + threadIdx.x;
  int k_in = blockIdx.y*blockDim.y + threadIdx.y;
  int i    = blockIdx.z;
  int index_in = i*NY*NZ + k_in*NY + j_in;
  if(j_in<NY && k_in<NZ){
    scratch[threadIdx.y][threadIdx.x] = input[index_in];
  }

  int k_ot = blockIdx.x*blockDim.x + threadIdx.y;//yblock_jof;//j_in;
  int j_ot = blockIdx.y*blockDim.y + threadIdx.x;

  int yblock_dim = gridDim.z;
  int yblock_idx = k_ot/yblock_dim;
  int yblock_jof = k_ot%yblock_dim;

  int index_ot = yblock_idx*yblock_dim*yblock_dim*NZ + i*yblock_dim*NZ + yblock_jof*NZ + j_ot;

  __syncthreads();

  if(k_ot<NY && j_ot<NZ){
    output[index_ot] = scratch[threadIdx.x][threadIdx.y];;
  }


}


static dim3 threadsPerBlock;
static dim3 blocksPerGrid;

extern void trans_zyx_yblock_to_yzx(float2* input, float2* output,cudaStream_t stream)
{


        threadsPerBlock.x=THREADSPERBLOCK_IN;
        threadsPerBlock.y=THREADSPERBLOCK_IN;

        blocksPerGrid.y=(NZ+threadsPerBlock.x-1)/threadsPerBlock.x;
        blocksPerGrid.x=NY/threadsPerBlock.y;
        blocksPerGrid.z=NXSIZE;


        trans_zyx_yblock_to_yzx_kernel<<<blocksPerGrid,threadsPerBlock,0,stream>>>(input,output);
        //kernelCheck(RET,"rk_substep",1);

        return;
}


extern void trans_zyx_to_zxy(float2* input, float2* output,cudaStream_t stream)
{

        threadsPerBlock.x=THREADSPERBLOCK_IN;
        threadsPerBlock.y=THREADSPERBLOCK_IN;

        blocksPerGrid.x=(NZ+threadsPerBlock.x-1)/threadsPerBlock.x;
        blocksPerGrid.y=NY/threadsPerBlock.y;
        blocksPerGrid.z=NXSIZE;


        trans_zyx_to_zxy_kernel<<<blocksPerGrid,threadsPerBlock,0,stream>>>(input,output);
        //kernelCheck(RET,"rk_substep",1);

        return;
}


extern void trans_zxy_to_zyx(float2* input, float2* output,cudaStream_t stream)
{

        threadsPerBlock.x=THREADSPERBLOCK_IN;
        threadsPerBlock.y=THREADSPERBLOCK_IN;

        blocksPerGrid.x=(NZ+threadsPerBlock.x-1)/threadsPerBlock.x;
        blocksPerGrid.y=NY/threadsPerBlock.y;
        blocksPerGrid.z=NXSIZE;


        trans_zxy_to_zyx_kernel<<<blocksPerGrid,threadsPerBlock,0,stream>>>(input,output);
        //kernelCheck(RET,"rk_substep",1);

        return;
}


extern void trans_zxy_to_yzx(float2* input, float2* output,cudaStream_t stream)
{

        threadsPerBlock.x=THREADSPERBLOCK_IN;
        threadsPerBlock.y=THREADSPERBLOCK_IN;

        blocksPerGrid.x=(NZ+threadsPerBlock.x-1)/threadsPerBlock.x;
        blocksPerGrid.y=NY/threadsPerBlock.y;
        blocksPerGrid.z=NXSIZE;


        trans_zxy_to_yzx_kernel<<<blocksPerGrid,threadsPerBlock,0,stream>>>(input,output);
        //kernelCheck(RET,"rk_substep",1);

        return;
}


extern void trans_zyx_to_yzx(float2* input, float2* output,cudaStream_t stream)
{

                
        threadsPerBlock.x=THREADSPERBLOCK_IN;
        threadsPerBlock.y=THREADSPERBLOCK_IN;

        blocksPerGrid.x=(NZ+threadsPerBlock.x-1)/threadsPerBlock.x;
        blocksPerGrid.y=NY/threadsPerBlock.y;
        blocksPerGrid.z=NXSIZE;


        trans_zyx_to_yzx_kernel<<<blocksPerGrid,threadsPerBlock,0,stream>>>(input,output);
        //kernelCheck(RET,"rk_substep",1);

        return;
}

extern void trans_yzx_to_zyx(float2* input, float2* output,cudaStream_t stream)
{


        threadsPerBlock.x=THREADSPERBLOCK_IN;
        threadsPerBlock.y=THREADSPERBLOCK_IN;

        blocksPerGrid.y=(NZ+threadsPerBlock.x-1)/threadsPerBlock.x;
        blocksPerGrid.x=NY/threadsPerBlock.y;
        blocksPerGrid.z=NXSIZE;

        //cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);

        trans_yzx_to_zyx_kernel<<<blocksPerGrid,threadsPerBlock,0,stream>>>(input,output);
        //kernelCheck(RET,"rk_substep",1);

        return;
}

extern void trans_yzx_to_zyx_yblock(float2* input, float2* output,cudaStream_t stream)
{


        threadsPerBlock.x=THREADSPERBLOCK_IN;
        threadsPerBlock.y=THREADSPERBLOCK_IN;

        blocksPerGrid.y=(NZ+threadsPerBlock.x-1)/threadsPerBlock.x;
        blocksPerGrid.x=NY/threadsPerBlock.y;
        blocksPerGrid.z=NXSIZE;


        trans_yzx_to_zyx_yblock_kernel<<<blocksPerGrid,threadsPerBlock,0,stream>>>(input,output);
        //kernelCheck(RET,"rk_substep",1);

        return;
}


/*
static void __global__ rk_step_1(float2* ux,float2* uy,float2* uz,float2* u_wx,float2* u_wy,float2* u_wz,
									float2* rx, float2* ry, float2* rz,float Re,float dt,float Cf,int kf,int IGLOBAL,int NXSIZE,int nc)
{

	const float alpha[]={ 29.0f/96.0f, -3.0f/40.0f, 1.0f/6.0f};
	const float dseda[]={ 0.0f, -17.0f/60.0f, -5.0f/12.0f};
	

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
	
		
	threadsPerBlock.x=THREADSPERBLOCK_IN;
	threadsPerBlock.y=THREADSPERBLOCK_IN;

	blocksPerGrid.y=NXSIZE/threadsPerBlock.x;
	blocksPerGrid.x=NY*NZ/threadsPerBlock.y;


	rk_step_1<<<blocksPerGrid,threadsPerBlock>>>(u.x,u.y,u.z,uw.x,uw.y,uw.z,r.x,r.y,r.z,Re,dt,Cf,kf,IGLOBAL,NXSIZE,nc);
	kernelCheck(RET,"rk_initstep",1);
	
	return;
}



extern void RK3_step_2(vectorField u,vectorField uw,vectorField r,float Re,float dt,float Cf,int kf,int nc)
{

		
	threadsPerBlock.x=THREADSPERBLOCK_IN;
	threadsPerBlock.y=THREADSPERBLOCK_IN;

	blocksPerGrid.y=NXSIZE/threadsPerBlock.x;
	blocksPerGrid.x=NY*NZ/threadsPerBlock.y;


	rk_step_2<<<blocksPerGrid,threadsPerBlock>>>(u.x,u.y,u.z,uw.x,uw.y,uw.z,r.x,r.y,r.z,Re,dt,Cf,kf,IGLOBAL,NXSIZE,nc);
	kernelCheck(RET,"rk_substep",1);

	return;
}

*/


