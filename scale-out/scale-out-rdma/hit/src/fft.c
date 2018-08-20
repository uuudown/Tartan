#include "turH.h"


static cufftHandle fft2_r2c; 
static cufftHandle fft2_c2r; 
static cufftHandle fft1_c2c; 



static float2* aux_host_1;
static float2* aux_host_2;

static float2* sum;

static size_t size;


static int n_steps=16;

//Check

static void cufftCheck( cufftResult error, const char* function )
{
	if(error != CUFFT_SUCCESS)
	{
		printf("\n error  %s : %d \n", function, error);
		exit(1);
	}
		
	return;
}  


void fftSetup(void)
{

	int n2[2]={NY,2*NZ-2};
	int n1[1]={NX};
	
	//2D fourier transforms
        while(NXSIZE/n_steps<1) n_steps/=2;
	cufftCheck(cufftPlanMany( &fft2_r2c,2,n2,NULL,1,0,NULL,1,0,CUFFT_R2C,NXSIZE/n_steps),"ALLOCATE_FFT2_R2C");
	cufftCheck(cufftPlanMany( &fft2_c2r,2,n2,NULL,1,0,NULL,1,0,CUFFT_C2R,NXSIZE/n_steps),"ALLOCATE_FFT2_C2R");

	//1D fourier transforms

	cufftCheck(cufftPlanMany(&fft1_c2c,1,n1,NULL,1,0,NULL,1,0,CUFFT_C2C,NYSIZE*NZ/n_steps),"ALLOCATE_FFT1_R2C");



	//MALLOC aux buffer host	

	size=NXSIZE*NY*NZ*sizeof(float2);

	aux_host_1=(float2*)malloc(size);
	aux_host_2=(float2*)malloc(size);
			
	//Set up for sum
	sum=(float2*)malloc(NXSIZE*sizeof(float2));			

        cudaHostRegister(aux_host_1,size,0);
        cudaHostRegister(aux_host_2,size,0);
        cudaHostRegister(sum,NXSIZE*sizeof(float2),0);


	return;
}

void fftDestroy(void)
{
 	cufftDestroy(fft2_r2c);
	cufftDestroy(fft2_c2r);

	return;
}

static void transposeForward(float2* buffer_1){

	//Copy from gpu to cpu
	
	cudaCheck(cudaMemcpy(aux_host_1,buffer_1,size,cudaMemcpyDeviceToHost),"MemInfo_TranFOR");
	
	//Transpuesta [i,j,k][NX,NY,NZ] a -----> [j,k,i][NY,NZ,NX]

	mpiCheck(chxyz2yzx((double *)aux_host_1,(double*)aux_host_2,NX,NY,NZ,RANK,SIZE),"T");

	//Copy from gpu to cpu
	
	cudaCheck(cudaMemcpy(buffer_1,aux_host_2,size,cudaMemcpyHostToDevice),"MemInfo_TranFOR");

	return;
}

static void transposeBackward(float2* buffer_1){

	//Copy from gpu to cpu
	
	cudaCheck(cudaMemcpy(aux_host_1,buffer_1,size,cudaMemcpyDeviceToHost),"MemInfo_TranBack");
	
	//Transpuesta [i,j,k][NX,NY,NZ] a -----> [j,k,i][NY,NZ,NX]

	mpiCheck(chyzx2xyz((double *)aux_host_1,(double*)aux_host_2,NX,NY,NZ,RANK,SIZE),"T");

	//Copy from gpu to cpu
	
	cudaCheck(cudaMemcpy(buffer_1,aux_host_2,size,cudaMemcpyHostToDevice),"MemInfo_TranBack");

	return;

}


void fftForward(float2* buffer_1)
{


	//NX transformadas en 2D 	

	for(int i=0;i<n_steps;i++){	

	cufftCheck(cufftExecR2C(fft2_r2c,(float*)(buffer_1)+i*2*NY*NZ*NXSIZE/n_steps,buffer_1+i*NY*NZ*NXSIZE/n_steps),"forward transform");

	}	

	
	//Transpose

	transposeForward(buffer_1);
	

	//Transformada 1D NY*NZ

	
	for(int i=0;i<n_steps;i++){	

	cufftCheck(cufftExecC2C(fft1_c2c,buffer_1+i*NY*NZ*NXSIZE/n_steps,buffer_1+i*NY*NZ*NXSIZE/n_steps,CUFFT_FORWARD),"forward transform");	

	}
	

	//Transpose

	transposeBackward(buffer_1);
		

	return;
}


void fftBackward(float2* buffer_1)
{

	//Transpose

	transposeForward(buffer_1);

	//NY*NZ transformadas 1D en X
		
	for(int i=0;i<n_steps;i++){	

	cufftCheck(cufftExecC2C(fft1_c2c,buffer_1+i*NY*NZ*NXSIZE/n_steps,buffer_1+i*NY*NZ*NXSIZE/n_steps,CUFFT_INVERSE),"forward transform");	

	}

	//Transpose

	transposeBackward(buffer_1);

	//NX transformadas en 2D 

	for(int i=0;i<n_steps;i++){	

	cufftCheck(cufftExecC2R(fft2_c2r,buffer_1+i*NY*NZ*NXSIZE/n_steps,(float*)(buffer_1)+i*2*NY*NZ*NXSIZE/n_steps),"forward transform");
	
	}


	return;	

}

void calcUmax(vectorField t,float* ux,float* uy,float* uz)
{


	int size_l=2*NXSIZE*NY*NZ;
	int index;

	index=cublasIsamax(size_l, (const float *)t.x, 1);
	cudaCheck(cudaMemcpy(ux,(float*)t.x+index-1, sizeof(float), cudaMemcpyDeviceToHost),"caca");

	index=cublasIsamax (size_l, (const float *)t.y, 1);
	cudaCheck(cudaMemcpy(uy,(float*)t.y+index-1, sizeof(float), cudaMemcpyDeviceToHost),"caca");
	
	index=cublasIsamax (size_l, (const float *)t.z, 1);
	cudaCheck(cudaMemcpy(uz,(float*)t.z+index-1, sizeof(float), cudaMemcpyDeviceToHost),"caca");

	
	*ux=fabs(*ux);
	*uy=fabs(*uy);
	*uz=fabs(*uz);
	

	//MPI reduce
	reduceMAX(ux,uy,uz);

	return;

}

float sumElements2(float2* buffer_1){
        float sum_all=0;
        
        for(int i=0;i<n_steps;i++){     

        cufftCheck(cufftExecR2C(fft2_r2c,(float*)(buffer_1)+i*2*NY*NZ*NXSIZE/n_steps,buffer_1+i*NY*NZ*NXSIZE/n_steps),"forward transform");
        
        }
        
        for(int i=0;i<NXSIZE;i++){
        
        cudaCheck(cudaMemcpy((float2*)sum+i,(float2*)buffer_1+i*NY*NZ,sizeof(float2),cudaMemcpyDeviceToHost),"MemInfo1");
        
        };
        
        for(int i=1;i<NXSIZE;i++){
        
        sum[0].x+=sum[i].x;
        }
        reduceSUM((float*)sum,&sum_all);

        return sum_all;

};


float sumElements(float2* buffer_1){

	//destroza lo que haya en el buffer
START_RANGE("sumElements",5)
	float sum_all=0;
cudaMemcpy(aux_host_1,buffer_1,size,cudaMemcpyDeviceToHost);
        int pcount=0;
        for(int i=0; i<NXSIZE*NY*NZ; i++) { 
          sum_all += aux_host_1[i].x;
          if((aux_host_1[i].x != 0.f || aux_host_1[i].y != 0.f)) {
            //printf("%d aux_host[ %d ] = %g + i %g \n",pcount,i,aux_host_1[i].x,aux_host_1[i].y);
            pcount++;
          }
        }
        //printf("host_sum = %g \n",sum_all);
        sum_all = 0.f;	

	for(int i=0;i<n_steps;i++){	

	cufftCheck(cufftExecR2C(fft2_r2c,(float*)(buffer_1)+i*2*NY*NZ*NXSIZE/n_steps,buffer_1+i*NY*NZ*NXSIZE/n_steps),"forward transform");

	}

	for(int i=0;i<NXSIZE;i++){

	cudaCheck(cudaMemcpy((float2*)sum+i,(float2*)buffer_1+i*NY*NZ,sizeof(float2),cudaMemcpyDeviceToHost),"MemInfo1");

	};
	
	for(int i=1;i<NXSIZE;i++){

	sum[0].x+=sum[i].x;
	}

	//MPI SUM

	reduceSUM((float*)sum,&sum_all);
	//printf("sum_all = %g \n",sum_all);
END_RANGE
	return sum_all;

};


