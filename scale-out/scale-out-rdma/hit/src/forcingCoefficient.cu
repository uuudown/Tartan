#include "turH.h"

extern int SMCOUNT;

//__device__ unsigned int retirementCount = 0;

__device__ static float atomicMax(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
            __float_as_int(::fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

#define UMAX_BLK (128)
#define UMAX_GRD (70)

static __global__ void calcUmaxKernel(const float2 * __restrict data, float * __restrict temp, const int elements)
{
  __shared__ float scratch[UMAX_BLK];
  int index;
  float2 max={0.f,0.f};
  //const int stride = UMAX_BLK*gridDim.x;
  data += blockIdx.y*elements;
  for(index=threadIdx.x+blockIdx.x*UMAX_BLK; index<elements; index+=UMAX_BLK*UMAX_GRD)
  {
    float2 val  = data[index];
    if(val.x<0.f) val.x = -val.x;
    if(val.y<0.f) val.y = -val.y;
    if(val.x>max.x) max.x = val.x;
    if(val.y>max.y) max.y = val.y;
  }
  if(max.y > max.x) max.x = max.y;
  scratch[threadIdx.x] = max.x;
  __syncthreads();
  for(int offset=UMAX_BLK/2; offset>0; offset/=2)
  {
    if(threadIdx.x<offset){
      float val=scratch[threadIdx.x+offset];
      if(val>max.x){
        max.x = val;
        scratch[threadIdx.x] = val;
      }
    }
    __syncthreads();
  }
  if(threadIdx.x==0) atomicMax(&temp[blockIdx.y],scratch[0]);
}

#if 0
static __global__ void calcUmaxKernel(const float * __restrict x, const float * __restrict y, const float * __restrict z, float * __restrict temp, const int elements)
{
  //__shared__ int me;
  __shared__ float scratch[3][UMAX_BLK];
  int index;
  float max_x=0.f, max_y=0.f, max_z=0.f;
  //me = 0; __syncthreads();
  for(index=threadIdx.x+blockIdx.x*UMAX_BLK; index<elements; index+=UMAX_BLK*UMAX_GRD)
  {
    float val_x = x[index];
    float val_y = y[index];
    float val_z = z[index];

    val_x = val_x > 0 ? val_x : -val_x;
    val_y = val_y > 0 ? val_y : -val_y;
    val_z = val_z > 0 ? val_z : -val_z;

    if(val_x>max_x) max_x = val_x;
    if(val_y>max_y) max_y = val_y;
    if(val_z>max_z) max_z = val_z;
    //if(index == 3381025 && (max_y/((float)N*(float)N*(float)N))>4.63f) { me = 1; printf("max_x = %g val_x = %g \n",max_y/((float)N*(float)N*(float)N),val_y/((float)N*(float)N*(float)N)); }
  }
  scratch[0][threadIdx.x] = max_x;
  scratch[1][threadIdx.x] = max_y;
  scratch[2][threadIdx.x] = max_z;
  __syncthreads();
#if 1
  for(int offset=UMAX_BLK/2; offset>0; offset/=2)
  {
    if(threadIdx.x<offset){
      scratch[0][threadIdx.x] = scratch[0][threadIdx.x+offset] > scratch[0][threadIdx.x] ? scratch[0][threadIdx.x+offset] : scratch[0][threadIdx.x];
      scratch[1][threadIdx.x] = scratch[1][threadIdx.x+offset] > scratch[1][threadIdx.x] ? scratch[1][threadIdx.x+offset] : scratch[1][threadIdx.x];
      scratch[2][threadIdx.x] = scratch[2][threadIdx.x+offset] > scratch[2][threadIdx.x] ? scratch[2][threadIdx.x+offset] : scratch[2][threadIdx.x];
    }
    __syncthreads();
  }
  if(threadIdx.x==0){
    atomicMax(&temp[0],scratch[0][0]); 
    atomicMax(&temp[1],scratch[1][0]); 
    atomicMax(&temp[2],scratch[2][0]);
  }
}
#endif
#if 0
  if(threadIdx.x<64){
    float val_x = scratch[0][threadIdx.x+64];
    float val_y = scratch[1][threadIdx.x+64];
    float val_z = scratch[2][threadIdx.x+64];    
    if(val_x>max_x) max_x = val_x;
    if(val_y>max_y) max_y = val_y;
    if(val_z>max_z) max_z = val_z;
    scratch[0][threadIdx.x] = max_x;
    scratch[1][threadIdx.x] = max_y;
    scratch[2][threadIdx.x] = max_z;
  }
  __syncthreads();
  if(threadIdx.x<32){
    float val_x = scratch[0][threadIdx.x+32];
    float val_y = scratch[1][threadIdx.x+32];
    float val_z = scratch[2][threadIdx.x+32];
    if(val_x>max_x) max_x = val_x;
    if(val_y>max_y) max_y = val_y;
    if(val_z>max_z) max_z = val_z;
    scratch[0][threadIdx.x] = max_x;
    scratch[1][threadIdx.x] = max_y;
    scratch[2][threadIdx.x] = max_z;
  }
  __syncthreads();
  if(threadIdx.x<16){
    float val_x = scratch[0][threadIdx.x+16];
    float val_y = scratch[1][threadIdx.x+16];
    float val_z = scratch[2][threadIdx.x+16];
    if(val_x>max_x) max_x = val_x;
    if(val_y>max_y) max_y = val_y;
    if(val_z>max_z) max_z = val_z;
    scratch[0][threadIdx.x] = max_x;
    scratch[1][threadIdx.x] = max_y;
    scratch[2][threadIdx.x] = max_z;
  }
  __syncthreads();
  if(threadIdx.x<8){
    float val_x = scratch[0][threadIdx.x+8];
    float val_y = scratch[1][threadIdx.x+8];
    float val_z = scratch[2][threadIdx.x+8];
    if(val_x>max_x) max_x = val_x;
    if(val_y>max_y) max_y = val_y;
    if(val_z>max_z) max_z = val_z;
    scratch[0][threadIdx.x] = max_x;
    scratch[1][threadIdx.x] = max_y;
    scratch[2][threadIdx.x] = max_z;
  }
  __syncthreads();
  if(threadIdx.x<4){
    float val_x = scratch[0][threadIdx.x+4];
    float val_y = scratch[1][threadIdx.x+4];
    float val_z = scratch[2][threadIdx.x+4];
    if(val_x>max_x) max_x = val_x;
    if(val_y>max_y) max_y = val_y;
    if(val_z>max_z) max_z = val_z;
    scratch[0][threadIdx.x] = max_x;
    scratch[1][threadIdx.x] = max_y;
    scratch[2][threadIdx.x] = max_z;
  }
  __syncthreads();
  if(threadIdx.x<2){
    float val_x = scratch[0][threadIdx.x+2];
    float val_y = scratch[1][threadIdx.x+2];
    float val_z = scratch[2][threadIdx.x+2];
    if(val_x>max_x) max_x = val_x;
    if(val_y>max_y) max_y = val_y;
    if(val_z>max_z) max_z = val_z;
    scratch[0][threadIdx.x] = max_x;
    scratch[1][threadIdx.x] = max_y;
    scratch[2][threadIdx.x] = max_z;
  }
  __syncthreads();
  if(threadIdx.x<1){
    float val_x = scratch[0][threadIdx.x+1];
    float val_y = scratch[1][threadIdx.x+1];
    float val_z = scratch[2][threadIdx.x+1];
    if(val_x>max_x) max_x = val_x;
    if(val_y>max_y) max_y = val_y;
    if(val_z>max_z) max_z = val_z;
    //int out_idx = blockIdx.x;
    atomicMax(&temp[0],max_x); //out_idx += UMAX_BLK;
    atomicMax(&temp[1],max_y); //out_idx += UMAX_BLK;
    atomicMax(&temp[2],max_z);
  }
}
#endif
#if 0
  __syncthreads();
  __shared__ bool amLast;
  __threadfence();
  if(threadIdx.x==0){
    unsigned int ticket = atomicInc(&retirementCount, UMAX_BLK);
    amLast = (ticket == UMAX_BLK-1);
    //if(me==1) printf("block: %d, ticket: %d, amLast: %d, max_y: %g \n",blockIdx.x,ticket,amLast,max_y/((float)N*(float)N*(float)N));
  }
  __syncthreads();
  if(amLast){
    int idx = threadIdx.x;
    scratch[0][threadIdx.x] = temp[idx]; idx += UMAX_BLK;
    scratch[1][threadIdx.x] = temp[idx]; idx += UMAX_BLK;
    scratch[2][threadIdx.x] = temp[idx]; 

  __syncthreads();
  if(threadIdx.x<64){
    float val_x = scratch[0][threadIdx.x+64];
    float val_y = scratch[1][threadIdx.x+64];
    float val_z = scratch[2][threadIdx.x+64];
    if(val_x>max_x) max_x = val_x;
    if(val_y>max_y) max_y = val_y;
    if(val_z>max_z) max_z = val_z;
    scratch[0][threadIdx.x] = max_x;
    scratch[1][threadIdx.x] = max_y;
    scratch[2][threadIdx.x] = max_z;
  }
  __syncthreads();
  if(threadIdx.x<32){
    float val_x = scratch[0][threadIdx.x+32];
    float val_y = scratch[1][threadIdx.x+32];
    float val_z = scratch[2][threadIdx.x+32];
    if(val_x>max_x) max_x = val_x;
    if(val_y>max_y) max_y = val_y;
    if(val_z>max_z) max_z = val_z;
    scratch[0][threadIdx.x] = max_x;
    scratch[1][threadIdx.x] = max_y;
    scratch[2][threadIdx.x] = max_z;
  }
  __syncthreads();
  if(threadIdx.x<16){
    float val_x = scratch[0][threadIdx.x+16];
    float val_y = scratch[1][threadIdx.x+16];
    float val_z = scratch[2][threadIdx.x+16];
    if(val_x>max_x) max_x = val_x;
    if(val_y>max_y) max_y = val_y;
    if(val_z>max_z) max_z = val_z;
    scratch[0][threadIdx.x] = max_x;
    scratch[1][threadIdx.x] = max_y;
    scratch[2][threadIdx.x] = max_z;
  }
  __syncthreads();
  if(threadIdx.x<8){
    float val_x = scratch[0][threadIdx.x+8];
    float val_y = scratch[1][threadIdx.x+8];
    float val_z = scratch[2][threadIdx.x+8];
    if(val_x>max_x) max_x = val_x;
    if(val_y>max_y) max_y = val_y;
    if(val_z>max_z) max_z = val_z;
    scratch[0][threadIdx.x] = max_x;
    scratch[1][threadIdx.x] = max_y;
    scratch[2][threadIdx.x] = max_z;
  }
  __syncthreads();
  if(threadIdx.x<4){
    float val_x = scratch[0][threadIdx.x+4];
    float val_y = scratch[1][threadIdx.x+4];
    float val_z = scratch[2][threadIdx.x+4];
    if(val_x>max_x) max_x = val_x;
    if(val_y>max_y) max_y = val_y;
    if(val_z>max_z) max_z = val_z;
    scratch[0][threadIdx.x] = max_x;
    scratch[1][threadIdx.x] = max_y;
    scratch[2][threadIdx.x] = max_z;
  }
  __syncthreads();
  if(threadIdx.x<2){
    float val_x = scratch[0][threadIdx.x+2];
    float val_y = scratch[1][threadIdx.x+2];
    float val_z = scratch[2][threadIdx.x+2];
    if(val_x>max_x) max_x = val_x;
    if(val_y>max_y) max_y = val_y;
    if(val_z>max_z) max_z = val_z;
    scratch[0][threadIdx.x] = max_x;
    scratch[1][threadIdx.x] = max_y;
    scratch[2][threadIdx.x] = max_z;
  }
  __syncthreads();
  if(threadIdx.x<1){
    float val_x = scratch[0][threadIdx.x+1];
    float val_y = scratch[1][threadIdx.x+1];
    float val_z = scratch[2][threadIdx.x+1];
    if(val_x>max_x) max_x = val_x;
    if(val_y>max_y) max_y = val_y;
    if(val_z>max_z) max_z = val_z;
    temp[0] = max_x;
    temp[1] = max_y;
    temp[2] = max_z;
    retirementCount = 0;

//        float N3=(float)N*N*N;
//    printf("%g %g %g \n",max_x/N3, max_y/N3, max_z/N3);
  }
  }//if amLast
}
#endif
#endif

static __global__ void calcEnergyShellKernel2(float2* ux,float2* uy,float2* uz,float2* t,int ks,int IGLOBAL,int NXSIZE)
{
    __shared__ float scratch[2][3][3];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    scratch[tz][ty][tx] = 0.f;
    int i = tx-1;
    int j = ty-1;
    int k = tz;
    if(tx==0) i = NXSIZE - 1;
    if(ty==0) j = NY - 1;
    int ig = i + IGLOBAL;
    float k1=(ig)<NX/2 ? (float)(ig) : (float)(ig)-(float)NX ;
    float k2=j<NY/2 ? (float)j : (float)j-(float)NY ;
    float k3=(float)k;
    float kk=k1*k1+k2*k2+k3*k3;
    float N3=(float)N*(float)N*(float)N;
    int h = i*NY*NZ+j*NZ+k;

    float2 u1,u2,u3;
    float E1=0.0f,E2=0.0f,E3=0.0f;

    if(kk<ks*ks){
      u1=ux[h];
      u2=uy[h];
      u3=uz[h];
      u1.x=u1.x/N3;
      u2.x=u2.x/N3;
      u3.x=u3.x/N3;
      u1.y=u1.y/N3;
      u2.y=u2.y/N3;
      u3.y=u3.y/N3;
      E1=(u1.x*u1.x+u1.y*u1.y);
      E2=(u2.x*u2.x+u2.y*u2.y);
      E3=(u3.x*u3.x+u3.y*u3.y);
    }
    float factor = 2.0f;
    if(k==0) factor = 1.0f;
    if(ig+j+k == 0) factor = 0.0f;
    scratch[tz][ty][tx] = factor*(E1+E2+E3);
    __syncthreads();
    if(tz==0) scratch[0][ty][tx] += scratch[1][ty][tx];
    __syncthreads();
    if(ty==0) scratch[tz][0][tx] += scratch[tz][1][tx] + scratch[tz][2][tx];
    __syncthreads();
    if(tx==0) scratch[tz][ty][0] += scratch[tz][ty][1] + scratch[tz][ty][2];
    __syncthreads();
    if(tx==0&&ty==0&&tz==0) t[0].x = scratch[0][0][0];
}

static __global__ void calcEnergyShellKernel(float2* ux,float2* uy,float2* uz,float2* t,int ks,int IGLOBAL,int NXSIZE)
{
	

        int ind  = blockIdx.x * blockDim.x + threadIdx.x;


        int  k = ind%NZ;
        int  i = ind/(NZ*NY);
        int  j = ind/NZ-i*NY;


	float k1,k2,k3;
	float kk;

	float N3=(float)N*(float)N*(float)N;

	
	int h=i*NY*NZ+j*NZ+k;

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

	float e1,e2;

	if(kk<ks*ks){		

	// Read {u1,u2,u3}	
	
	float2 u1=ux[h];
	float2 u2=uy[h];
	float2 u3=uz[h];


	u1.x=u1.x/N3;
	u2.x=u2.x/N3;
	u3.x=u3.x/N3;

	u1.y=u1.y/N3;
	u2.y=u2.y/N3;
	u3.y=u3.y/N3;

	float E1=(u1.x*u1.x+u1.y*u1.y);
	float E2=(u2.x*u2.x+u2.y*u2.y);	
	float E3=(u3.x*u3.x+u3.y*u3.y);

	e1=2.0f*(E1+E2+E3);	
	e2=0.0f;

	}else{
	
	e1=0.0f;
	e2=0.0f;
	
	}


	if(k==0){
	e1*=1.0f/2.0f;	
	e2*=1.0f/2.0f;
	}
	
	if(h==0){
	e1=0.0f;
	e2=0.0f;
	}

        //if(e1>0.0f) printf("e[%d][%d][%d] (%d) = %g\n",i,j,k,h,e1);

	t[h].x=e1;
	t[h].y=e2;

	}
}

static dim3 threadsPerBlock;
static dim3 blocksPerGrid;

void calc_Umax2(vectorField u, float* temp)
{
  int elements=NXSIZE*NY*NZ;
  blocksPerGrid.x = UMAX_GRD;//SMCOUNT*8;
  blocksPerGrid.y = 3;

  //printf("elements = %d \n",elements);
  calcUmaxKernel<<<blocksPerGrid,UMAX_BLK,0,compute_stream>>>(u.x,temp,elements);
  kernelCheck(RET,"calcUmaxKernel",1);
}

void calc_energy_shell(vectorField u,float2* t,int ks)
{

        int elements = NXSIZE*NY*NZ;

        // Operate over N*N*(N/2+1) matrix
        threadsPerBlock.x=THREADSPERBLOCK_NU;

        blocksPerGrid.x=(elements+threadsPerBlock.x-1)/threadsPerBlock.x;


	calcEnergyShellKernel<<<blocksPerGrid,threadsPerBlock,0,compute_stream>>>(u.x,u.y,u.z,t,ks,IGLOBAL,NXSIZE);
	kernelCheck(RET,"rk_initstep",1);
		
	return;

}

static void calc_energy_shell2(vectorField u,float2* t,int ks)
{

        threadsPerBlock.x=3;
        threadsPerBlock.y=3;
        threadsPerBlock.z=2;

        blocksPerGrid.y=1;
        blocksPerGrid.x=1;

        calcEnergyShellKernel2<<<blocksPerGrid,threadsPerBlock>>>(u.x,u.y,u.z,t,ks,IGLOBAL,NXSIZE);
        kernelCheck(RET,"calcEng2",1);

        return;

}


float caclCf(vectorField u,float2* t,int kf, case_config_t *config)
{

	//conserving keta=2

	int kmax=sqrt(2.0f)*N/3.0f;

	//float res=config->resolution; //kmax*eta=res

	float energy,energy_loc;
	
	calc_energy_shell2(u,t,kf);	
        cudaMemcpy(&energy_loc,t,sizeof(float),cudaMemcpyDeviceToHost);
        reduceSUM(&energy_loc,&energy);

//printf("energy = %g %g \n",energy, energy_loc);
//getchar();
	//energy=sumElements(t);
	
	//if(RANK==0){
	//printf("\nenergy_shell=%f\n",energy/2.0f);
	//};

	float Cf=ENERGY_IN/(energy);
	
	return Cf;

}

