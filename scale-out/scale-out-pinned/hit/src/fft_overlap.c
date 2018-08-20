
#include "turH.h"
#include <cublas_v2.h>

static cufftHandle fft1_c2c; 
static cufftHandle fft2_c2r; 
static cufftHandle fft2_r2c; 

static float2* aux_host_1[6];
static float2* aux_host_2[6];
float2* aux_dev[6];

static float2* aux_host1;
static float2* aux_host2;
static float2* aux_host3;
static float2* aux_host4;
static float2* aux_host5;
static float2* aux_host6;

static float2* aux_host11;
static float2* aux_host22;
static float2* aux_host33;
static float2* aux_host44;
static float2* aux_host55;
static float2* aux_host66;

static float2* buffer[6];

static cublasHandle_t cublasHandle;
static float2 alpha[1];
static float2 betha[1];

static size_t size;
static float2* sum;

static cudaStream_t STREAMS[6];

int stream_idx=0;
cudaStream_t compute_stream;
cudaStream_t h2d_stream;
cudaStream_t d2h_stream;
cudaEvent_t events[1000];

static int MPIErr;


//Check

static void cublasCheck(cublasStatus_t error, const char* function )
{
	if(error !=  CUBLAS_STATUS_SUCCESS)
	{
		printf("\n error  %s : %d \n", function, error);
		exit(1);
	}
		
	return;
}  

static void cufftCheck( cufftResult error, const char* function )
{
	if(error != CUFFT_SUCCESS)
	{
		printf("\n error  %s : %d \n", function, error);
		exit(1);
	}
		
	return;
}  


void setFftAsync(void){

	int nRows = NX;
        int nCols = 2*NZ-2;

	int n2[2]={nRows, nCols};
	int n1[1]={NY};

        int idist = nRows*2*(nCols/2+1);//nRows*nCols;
        int odist = nRows*(nCols/2+1);

        int inembed[2] = {nRows, 2*(nCols/2+1) };//{nRows, nCols    };
        int onembed[2] = {nRows,    nCols/2+1  };

        int istride = 1;
        int ostride = 1;
	
	//2D fourier transforms
	cufftCheck(cufftPlanMany( &fft2_r2c,2,n2,inembed,istride,idist,onembed,ostride,odist,CUFFT_R2C,NYSIZE),"ALLOCATE_FFT2_R2C");
        cufftCheck(cufftPlanMany( &fft2_c2r,2,n2,onembed,ostride,odist,inembed,istride,idist,CUFFT_C2R,NYSIZE),"ALLOCATE_FFT2_C2R");
        //cufftCheck(cufftPlanMany( &fft2_r2c,2,n2,NULL,1,0,NULL,1,0,CUFFT_R2C,NYSIZE),"ALLOCATE_FFT2_R2C");
	//cufftCheck(cufftPlanMany( &fft2_c2r,2,n2,NULL,1,0,NULL,1,0,CUFFT_C2R,NYSIZE),"ALLOCATE_FFT2_C2R");

	//1D fourier transforms

	cufftCheck(cufftPlanMany(&fft1_c2c,1,n1,NULL,1,0,NULL,1,0,CUFFT_C2C,NXSIZE*NZ),"ALLOCATE_FFT1_R2C");

	//Set streams

	for(int i=0;i<6;i++){
	cudaCheck(cudaStreamCreate(&STREAMS[i]),"create_streams"); 
	}

        cudaCheck(cudaStreamCreate(&compute_stream),"create_streams");
        cudaCheck(cudaStreamCreate(&h2d_stream),"create_streams");
        cudaCheck(cudaStreamCreate(&d2h_stream),"create_streams");

        for(int i=0; i<1000; i++) cudaEventCreateWithFlags( &events[i], cudaEventDisableTiming ) ;

	//MALLOC aux buffer host	

	size=NXSIZE*NY*NZ*sizeof(float2);
	
	//MALLOC PINNED MEMORY TO ALLOW OVERLAPPING

	//for(int i=0;i<6;i++){
	//cudaCheck(cudaHostAlloc((void**)&aux_host_1[i],size,cudaHostAllocWriteCombined  ),"malloc_1");
	//cudaCheck(cudaHostAlloc((void**)&aux_host_2[i],size,cudaHostAllocWriteCombined  ),"malloc_2");
	//}


	/*
	cudaCheck(cudaHostAlloc((void**)&aux_host1,size,cudaHostAllocWriteCombined ),"malloc_1");
	cudaCheck(cudaHostAlloc((void**)&aux_host2,size,cudaHostAllocWriteCombined ),"malloc_2");
	*/

	aux_host1=(float2*)malloc(size);
	aux_host2=(float2*)malloc(size);	
        cudaHostRegister(aux_host1,size,0);
        cudaHostRegister(aux_host2,size,0);

        for(int i=0;i<6;i++){
//          if(i==0){
//            aux_host_1[i] = aux_host1;
//            aux_host_2[i] = aux_host2;
//            aux_dev[i] = AUX;
//          }else{
            aux_host_1[i]=(float2*)malloc(size);
            aux_host_2[i]=(float2*)malloc(size);
            cudaHostRegister(aux_host_1[i],size,0);
            cudaHostRegister(aux_host_2[i],size,0);
            cudaMalloc((void**)&aux_dev[i],size);
//          }
        } 


	/*
	cudaCheck(cudaHostAlloc((void**)&aux_host_1[1],size,cudaHostAllocWriteCombined ),"malloc_1");
	cudaCheck(cudaHostAlloc((void**)&aux_host_2[1],size,cudaHostAllocWriteCombined ),"malloc_2");

	cudaCheck(cudaHostAlloc((void**)&aux_host_1[2],size,cudaHostAllocWriteCombined ),"malloc_1");
	cudaCheck(cudaHostAlloc((void**)&aux_host_2[2],size,cudaHostAllocWriteCombined ),"malloc_2");

	cudaCheck(cudaHostAlloc((void**)&aux_host_1[3],size,cudaHostAllocWriteCombined ),"malloc_1");
	cudaCheck(cudaHostAlloc((void**)&aux_host_2[3],size,cudaHostAllocWriteCombined ),"malloc_2");

	cudaCheck(cudaHostAlloc((void**)&aux_host_1[4],size,cudaHostAllocWriteCombined ),"malloc_1");
	cudaCheck(cudaHostAlloc((void**)&aux_host_2[4],size,cudaHostAllocWriteCombined ),"malloc_2");

	cudaCheck(cudaHostAlloc((void**)&aux_host_1[5],size,cudaHostAllocWriteCombined  ),"malloc_1");
	cudaCheck(cudaHostAlloc((void**)&aux_host_2[5],size,cudaHostAllocWriteCombined  ),"malloc_2");

	cudaCheck(cudaHostAlloc((void**)&aux_host1,size,cudaHostAllocWriteCombined ),"malloc_1");
	cudaCheck(cudaHostAlloc((void**)&aux_host2,size,cudaHostAllocWriteCombined  ),"malloc_2");
	
	cudaCheck(cudaHostAlloc((void**)&aux_host3,size,cudaHostAllocWriteCombined  ),"malloc_1");
	cudaCheck(cudaHostAlloc((void**)&aux_host4,size,cudaHostAllocWriteCombined  ),"malloc_2");

	cudaCheck(cudaHostAlloc((void**)&aux_host5,size,cudaHostAllocWriteCombined  ),"malloc_1");
	cudaCheck(cudaHostAlloc((void**)&aux_host6,size,cudaHostAllocWriteCombined  ),"malloc_2");

	cudaCheck(cudaHostAlloc((void**)&aux_host11,size,cudaHostAllocWriteCombined  ),"malloc_1");
	cudaCheck(cudaHostAlloc((void**)&aux_host22,size,cudaHostAllocWriteCombined  ),"malloc_2");
	
	cudaCheck(cudaHostAlloc((void**)&aux_host33,size,cudaHostAllocWriteCombined  ),"malloc_1");
	cudaCheck(cudaHostAlloc((void**)&aux_host44,size,cudaHostAllocWriteCombined  ),"malloc_2");

	cudaCheck(cudaHostAlloc((void**)&aux_host55,size,cudaHostAllocWriteCombined  ),"malloc_1");
	cudaCheck(cudaHostAlloc((void**)&aux_host66,size,cudaHostAllocWriteCombined  ),"malloc_2");
*/

	
	//SET TRANSPOSE
		
	cublasCheck(cublasCreate(&cublasHandle),"Cre");

	alpha[0].x=1.0f;
	alpha[0].y=0.0f;

}

void transpose_A(float2* u_2,float2* u_1){

	//[NY,NZ]--->[NZ,NY]
/*printf("NX=%d NY=%d NZ=%d \n",NX,NY,NZ);
getchar();
        trans_zyx_to_yzx(u_1, u_2);

*/
///*
	for(int i=0;i<NXSIZE;i++){
	cublasCheck(cublasCgeam(cublasHandle,CUBLAS_OP_T,CUBLAS_OP_T,NY,NZ,alpha,(const float2*)u_1+i*NY*NZ,NZ,0,0,NZ,(float2*)u_2+i*NY*NZ,NY),"Tr");
	}
//*/
	return;


}

void transpose_B(float2* u_2,float2* u_1){

	//[NZ,NY]--->[NY,NZ]

	for(int i=0;i<NXSIZE;i++){
	cublasCheck(cublasCgeam(cublasHandle,CUBLAS_OP_T,CUBLAS_OP_T,NZ,NY,alpha,(const float2*)u_1+i*NY*NZ,NY,0,0,NY,(float2*)u_2+i*NY*NZ,NZ),"Tr");
	}

	return;

}

static void transpose(float2* u_2,const float2* u_1,int Nx,int Ny){

        //Transpuesta de [i,k,j][NX,NZ,NY] a -----> [j,i,k][NY,NX,NZ]
        cublasCheck(cublasCgeam(cublasHandle,CUBLAS_OP_T,CUBLAS_OP_T,Ny,Nx,alpha,(const float2*)u_1,Nx,0,0,Nx,(float2*)u_2,Ny),"Tr_1");
        return;
}

static void transposeBatched(float2* u_2,const float2* u_1,int Nx,int Ny,int batch){
        //Transpuesta de [i,k,j][NX,NZ,NY] a -----> [j,i,k][NY,NX,NZ]
        for(int nstep=0;nstep<batch;nstep++){
        int stride=nstep*Nx*Ny;
        cublasCheck(cublasCgeam(cublasHandle,CUBLAS_OP_T,CUBLAS_OP_T,Ny,Nx,alpha,(const float2*)u_1+stride,Nx,0,0,Nx,(float2*)u_2+stride,Ny),"Tr_2");
        }
        return;
}

void fftBack1T_A(float2* u1, int stid){
        int myNx = NY/SIZE;
        int myNy = NX/SIZE;
        stream_idx = stid;
        cublasCheck(cublasSetStream(cublasHandle,compute_stream),"stream");
        cufftCheck(cufftSetStream(fft1_c2c,compute_stream),"SetStream");
/*CHECK_CUDART( cudaDeviceSynchronize() );
        for(int ii=0; ii<NXSIZE*NY*NZ; ii++){
          int zid = ii%NZ;
          int yid = (ii%(NY*NZ))/NZ;
          int xid = ii/(NY*NZ);
          aux_host_1[2][ii].x = (float)zid;
          aux_host_1[2][ii].y = (float)yid + (float)xid/1000.0;
        }

        CHECK_CUDART( cudaMemcpy(u1, aux_host_1[2], size, cudaMemcpyHostToDevice) );


        transpose_A(aux_dev[0],u1);

        CHECK_CUDART( cudaMemcpy(aux_host_1[0], aux_dev[0], size, cudaMemcpyDeviceToHost) );
*/
//        transpose_A(aux_dev[stid],u1);

        trans_zyx_to_yzx(u1, aux_dev[stid],compute_stream);
/*
        CHECK_CUDART( cudaMemcpy(aux_host_1[1], aux_dev[1], size, cudaMemcpyDeviceToHost) );

        for(int ii=0; ii<NXSIZE*NY*NZ; ii++){
          float2 gold=aux_host_1[0][ii];
          float2 mine=aux_host_1[1][ii];
          if((gold.x != mine.x)||(gold.y != mine.y)) { printf("element %d error, gold=(%4.4f,%4.4f), mine=(%4.4f,%4.4f) ...",ii,gold.x,gold.y,mine.x,mine.y); getchar(); printf("\n"); }
        }

CHECK_CUDART( cudaDeviceSynchronize() );
printf("last error = %s \n",cudaGetErrorString(cudaGetLastError()));
getchar();
*/
        cufftCheck(cufftExecC2C(fft1_c2c,aux_dev[stid],u1,CUFFT_INVERSE),"forward transform");
//        cudaDeviceSynchronize();
//printf("\n calling transpose %d to %d ",NY,NZ*NX/SIZE);
        
////        transpose(aux_dev[stid],(const float2*)u1,NY,NZ*NX/SIZE);
//        cudaDeviceSynchronize();
//printf("... finished ... \n",NY,NZ*NX/SIZE);

////        transposeBatched(u1,(const float2*)aux_dev[stid],NZ*NX/SIZE,NY/SIZE,SIZE);
////        transposeBatched(aux_dev[stid],(const float2*)u1,NY/SIZE,NZ,NX);

        trans_yzx_to_zyx_yblock(u1, aux_dev[stid], compute_stream);

        cudaEventRecord(events[stid],compute_stream);
        cudaStreamWaitEvent(d2h_stream,events[stid],0);
//#define USE_GPU_MPI
//#define PIPE_XFER
//#define USE_TOGETHER
//#define USE_NB_COMM

#ifdef PIPE_XFER
//if(pipe_xfer==1){
#ifndef USE_GPU_MPI
       //int together=1;
       //while( (NZ*myNx*myNy*sizeof(float2)*together) < min_kb_xfer*1024 ) together *= 2;
       //printf("together = %d\n",together);  
//#ifdef USE_TOGETHER
#if 0
        int ii,ii_chunk;
        for(ii_chunk=0; ii_chunk<(SIZE+together-1)/together; ii_chunk++){
          if(ii_chunk==0){
            for(ii=ii_chunk*together; ii<(ii_chunk+1)*together && ii<SIZE; ii++){
              int dest=ii^RANK;
              if(dest!=RANK){
                CHECK_CUDART(cudaMemcpyAsync((float2*)aux_host_1[stid]+dest*NZ*myNx*myNy,(float2*)aux_dev[stid]+dest*NZ*myNx*myNy,NZ*myNx*myNy*sizeof(float2),cudaMemcpyDeviceToHost,d2h_stream));
                cudaEventRecord(events[30+128*stid+ii],d2h_stream);              
              }
            }
          }else{
            int min_dest=999999;
            for(ii=ii_chunk*together; ii<(ii_chunk+1)*together && ii<SIZE; ii++){
              int dest=ii^RANK;
              if(dest<min_dest) min_dest=dest;
            }
            CHECK_CUDART(cudaMemcpyAsync((float2*)aux_host_1[stid]+min_dest*NZ*myNx*myNy,(float2*)aux_dev[stid]+min_dest*NZ*myNx*myNy,together*NZ*myNx*myNy*sizeof(float2),cudaMemcpyDeviceToHost,d2h_stream));
            for(ii=ii_chunk*together; ii<(ii_chunk+1)*together && ii<SIZE; ii++){
              cudaEventRecord(events[30+128*stid+ii],d2h_stream);
            }
          }
        }
#else
       int iter;
       for(iter=1; iter<SIZE; iter++){
         int dest = RANK ^ iter;
         cudaCheck(cudaMemcpyAsync((float2*)aux_host_1[stid]+dest*NZ*myNx*myNy,(float2*)aux_dev[stid]+dest*NZ*myNx*myNy,NZ*myNx*myNy*sizeof(float2),cudaMemcpyDeviceToHost,d2h_stream),"copy");
         cudaEventRecord(events[30+128*stid+iter],d2h_stream);
       }
#endif
#endif
#else
//}else{
        cudaCheck(cudaMemcpyAsync((float2*)aux_host_1[stid],(float2*)aux_dev[stid],size,cudaMemcpyDeviceToHost,d2h_stream),"copy");
        cudaEventRecord(events[10+stid],d2h_stream);
#endif
//}//pipe_xfer
}

void fftBack1T_B(float2* u1, int stid){
         int myNx = NY/SIZE;
         int myNy = NX/SIZE;
         int stream_idx = stid;

        cublasCheck(cublasSetStream(cublasHandle,compute_stream),"stream");
        cufftCheck(cufftSetStream(fft2_c2r,compute_stream),"SetStream");
#ifdef PIPE_XFER
//if(pipe_xfer==1){
#ifdef USE_GPU_MPI
       int iter;
       for(iter=1; iter<SIZE; iter++){
         int dest = RANK ^ iter;
         cudaEventSynchronize(events[stid]);
         MPI_Sendrecv(aux_dev[stid]+dest*NZ*myNx*myNy, NZ*myNx*myNy, MPI_DOUBLE, dest, 0, u1+dest*NZ*myNx*myNy, NZ*myNx*myNy, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
         //TODO: remove need to do this copy
         cudaCheck(cudaMemcpyAsync((float2*)aux_dev[stid]+dest*NZ*myNx*myNy,(float2*)u1+dest*NZ*myNx*myNy,NZ*myNx*myNy*sizeof(float2),cudaMemcpyDeviceToDevice,h2d_stream),"copy");
       }
       cudaEventRecord(events[20+stid],h2d_stream);
#else
#ifdef USE_TOGETHER
        int ii,ii_chunk;
        for(ii_chunk=0; ii_chunk<(SIZE+together-1)/together; ii_chunk++){
          if(ii_chunk==0){
#ifdef USE_NB_COMM
            for(ii=ii_chunk*together; ii<(ii_chunk+1)*together && ii<SIZE; ii++){
              int dest=ii^RANK;
              if(dest!=RANK){
                MPI_Irecv(aux_host_2[stid]+dest*NZ*myNx*myNy, NZ*myNx*myNy, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD, &recv_requests[ii]);
              }
            }
            for(ii=ii_chunk*together; ii<(ii_chunk+1)*together && ii<SIZE; ii++){
              int dest=ii^RANK;
              if(dest!=RANK){
                CHECK_CUDART( cudaEventSynchronize(events[30+128*stid+ii]) );
                MPI_Isend(aux_host_1[stid]+dest*NZ*myNx*myNy, NZ*myNx*myNy, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD, &send_requests[ii]);
              }
            }
            for(ii=ii_chunk*together; ii<(ii_chunk+1)*together && ii<SIZE; ii++){
              int dest=ii^RANK;
              if(dest!=RANK){
                MPI_Wait(&recv_requests[ii],&recv_status[ii]);
                CHECK_CUDART( cudaMemcpyAsync((float2*)aux_dev[stid]+dest*NZ*myNx*myNy,(float2*)aux_host_2[stid]+dest*NZ*myNx*myNy,NZ*myNx*myNy*sizeof(float2),cudaMemcpyHostToDevice,h2d_stream) );
              }
            }
#else
            for(ii=ii_chunk*together; ii<(ii_chunk+1)*together && ii<SIZE; ii++){
              int dest=ii^RANK;
              if(dest!=RANK){
                CHECK_CUDART( cudaEventSynchronize(events[30+128*stid+ii]) );
START_RANGE_ASYNC("MPI",3)
                MPI_Sendrecv(aux_host_1[stid]+dest*NZ*myNx*myNy, NZ*myNx*myNy, MPI_DOUBLE, dest, 0, aux_host_2[stid]+dest*NZ*myNx*myNy, NZ*myNx*myNy, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
END_RANGE_ASYNC
                CHECK_CUDART( cudaMemcpyAsync((float2*)aux_dev[stid]+dest*NZ*myNx*myNy,(float2*)aux_host_2[stid]+dest*NZ*myNx*myNy,NZ*myNx*myNy*sizeof(float2),cudaMemcpyHostToDevice,h2d_stream) );
              }
            }
#endif
          }else{
#ifdef USE_NB_COMM
            for(ii=ii_chunk*together; ii<(ii_chunk+1)*together && ii<SIZE; ii++){
              int dest=ii^RANK;
              MPI_Irecv(aux_host_2[stid]+dest*NZ*myNx*myNy, NZ*myNx*myNy, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD, &recv_requests[ii]);
            }
            int min_dest=999999;
            for(ii=ii_chunk*together; ii<(ii_chunk+1)*together && ii<SIZE; ii++){
              int dest=ii^RANK;
              if(dest<min_dest) min_dest=dest;
              CHECK_CUDART( cudaEventSynchronize(events[30+128*stid+ii]) );
              MPI_Isend(aux_host_1[stid]+dest*NZ*myNx*myNy, NZ*myNx*myNy, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD, &send_requests[ii]);
            }
            for(ii=ii_chunk*together; ii<(ii_chunk+1)*together && ii<SIZE; ii++){
                int dest=ii^RANK;
                MPI_Wait(&recv_requests[ii],&recv_status[ii]);
                CHECK_CUDART( cudaMemcpyAsync((float2*)aux_dev[stid]+dest*NZ*myNx*myNy,(float2*)aux_host_2[stid]+dest*NZ*myNx*myNy,NZ*myNx*myNy*sizeof(float2),cudaMemcpyHostToDevice,h2d_stream) );
            }
            //CHECK_CUDART( cudaMemcpyAsync((float2*)aux_dev[stid]+min_dest*NZ*myNx*myNy,(float2*)aux_host_2[stid]+min_dest*NZ*myNx*myNy,together*NZ*myNx*myNy*sizeof(float2),cudaMemcpyHostToDevice,h2d_stream) );
#else
            int min_dest=999999;
            for(ii=ii_chunk*together; ii<(ii_chunk+1)*together && ii<SIZE; ii++){
              int dest=ii^RANK;
              if(dest<min_dest) min_dest=dest;
              CHECK_CUDART( cudaEventSynchronize(events[30+128*stid+ii]) );
START_RANGE_ASYNC("MPI",3)
              MPI_Sendrecv(aux_host_1[stid]+dest*NZ*myNx*myNy, NZ*myNx*myNy, MPI_DOUBLE, dest, 0, aux_host_2[stid]+dest*NZ*myNx*myNy, NZ*myNx*myNy, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
END_RANGE_ASYNC
            }
            CHECK_CUDART( cudaMemcpyAsync((float2*)aux_dev[stid]+min_dest*NZ*myNx*myNy,(float2*)aux_host_2[stid]+min_dest*NZ*myNx*myNy,together*NZ*myNx*myNy*sizeof(float2),cudaMemcpyHostToDevice,h2d_stream) );
#endif
          } 
        }
#else
#ifdef USE_NB_COMM
       int iter;
       for(iter=1; iter<SIZE; iter++){
         int dest = RANK ^ iter;
         MPI_Irecv(aux_host_2[stid]+dest*NZ*myNx*myNy, NZ*myNx*myNy, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD, &recv_requests[iter]);
       }
       for(iter=1; iter<SIZE; iter++){
         int dest = RANK ^ iter;
         cudaEventSynchronize(events[30+128*stid+iter]);
         MPI_Send(aux_host_1[stid]+dest*NZ*myNx*myNy, NZ*myNx*myNy, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);//, &send_requests[iter]);
/*       }
       for(iter=1; iter<SIZE; iter++){
         int dest = RANK ^ iter;
*/
         MPI_Wait(&recv_requests[iter],&recv_status[iter]);
         CHECK_CUDART( cudaMemcpyAsync((float2*)aux_dev[stid]+dest*NZ*myNx*myNy,(float2*)aux_host_2[stid]+dest*NZ*myNx*myNy,NZ*myNx*myNy*sizeof(float2),cudaMemcpyHostToDevice,h2d_stream) );
       }
#else
       //cudaCheck(cudaMemcpyAsync((float2*)u1+RANK*NZ*myNx*myNy,(float2*)aux_dev[stid]+RANK*NZ*myNx*myNy,NZ*myNx*myNy*sizeof(float2),cudaMemcpyDeviceToDevice,compute_stream),"copy");
       int iter;
       for(iter=1; iter<SIZE; iter++){
         int dest = RANK ^ iter;
         cudaEventSynchronize(events[30+128*stid+iter]);
START_RANGE_ASYNC("MPI",3)
         MPI_Sendrecv(aux_host_1[stid]+dest*NZ*myNx*myNy, NZ*myNx*myNy, MPI_DOUBLE, dest, 0, aux_host_2[stid]+dest*NZ*myNx*myNy, NZ*myNx*myNy, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
END_RANGE_ASYNC
         cudaCheck(cudaMemcpyAsync((float2*)aux_dev[stid]+dest*NZ*myNx*myNy,(float2*)aux_host_2[stid]+dest*NZ*myNx*myNy,NZ*myNx*myNy*sizeof(float2),cudaMemcpyHostToDevice,h2d_stream),"copy");
       }
#endif
#endif
        CHECK_CUDART( cudaEventRecord(events[20+stid],h2d_stream) );
#endif
#else
//}else{
        CHECK_CUDART( cudaEventSynchronize(events[10+stid]) );
START_RANGE_ASYNC("MPI",3)
        MPIErr = MPI_Alltoall(aux_host_1[stid],NZ*myNx*myNy,MPI_DOUBLE,
                              aux_host_2[stid],NZ*myNx*myNy,MPI_DOUBLE,
                              MPI_COMM_WORLD);
END_RANGE_ASYNC
        mpiCheck(MPIErr,"transpoze");
        CHECK_CUDART( cudaMemcpyAsync((float2*)aux_dev[stid],(float2*)aux_host_2[stid],size,cudaMemcpyHostToDevice,h2d_stream) );
        cudaEventRecord(events[20+stid],h2d_stream);
#endif
//}//pipe_xfer
        CHECK_CUDART( cudaStreamWaitEvent(compute_stream,events[20+stid],0) );

        //transpose(aux_dev[stid],(const float2*)u1,NZ*NY/SIZE,NY);//EP: last NX was NY
        //transposeBatched(u1,(const float2*)aux_dev[stid],NY,NZ,NY/SIZE); //EP: NX was NY
        //trans_zxy_to_yzx(u1, aux_dev[stid], compute_stream);
        //trans_yzx_to_zyx(aux_dev[stid], u1, compute_stream);
        trans_zxy_to_zyx(aux_dev[stid], u1, compute_stream);
        //cudaMemcpyAsync(u1,aux_dev[stid],size,cudaMemcpyDeviceToDevice, compute_stream);
        cufftCheck(cufftExecC2R(fft2_c2r,u1,(float*)u1),"forward transform");
}

void fftBack1T(float2* u1){

	//Transpose from [x,y,z] to [x,z,y]
START_RANGE("transA",5)
	transpose_A(AUX,u1);
END_RANGE	
	//FFT 1D on Y
START_RANGE("CUFFT",3)		
	cufftCheck(cufftExecC2C(fft1_c2c,AUX,u1,CUFFT_INVERSE),"forward transform");
END_RANGE
	//cudaCheck(cudaMemcpy((float2*)aux_host1,(float2*)u1,size,cudaMemcpyDeviceToHost),"copy");

	//Transpose from [x,z,y] to [y,x,z]
	
	//mpiCheck(chyzx2xyz((double *)aux_host1,(double*)aux_host2,NY,NX,NZ,RANK,SIZE),"T");

	//cudaCheck(cudaMemcpy((float2*)u1,(float2*)aux_host2,size,cudaMemcpyHostToDevice),"copy");

START_RANGE("transYZX2XYZ",4)
	transposeYZX2XYZ(u1,NY,NX,NZ,RANK,SIZE);
END_RANGE

	//FFT 2D on X	
START_RANGE("CUFFT",3)
	cufftCheck(cufftExecC2R(fft2_c2r,u1,(float*)u1),"forward transform");
END_RANGE

}

void fftForw1T_A(float2* u1, int stid){
        int myNx = NY/SIZE;
        int myNy = NX/SIZE;
        stream_idx = stid;

        cublasCheck(cublasSetStream(cublasHandle,compute_stream),"stream");
        cufftCheck(cufftSetStream(fft2_r2c,compute_stream),"SetStream");

  cufftCheck(cufftExecR2C(fft2_r2c,(float*)u1,(float2*)u1),"forward transform");

//  transposeBatched(aux_dev[stid],(const float2*)u1,NZ,NY,myNx);
//  transpose(u1,(const float2*)aux_dev[stid],NY,myNx*NZ);
  trans_zyx_to_zxy(u1, aux_dev[stid], compute_stream);  

  CHECK_CUDART( cudaEventRecord(events[stid],compute_stream) );
  CHECK_CUDART( cudaStreamWaitEvent(d2h_stream,events[stid],0) );
//  cudaCheck(cudaMemcpyAsync((float2*)aux_host_1[stid],(float2*)aux_dev[stid],size,cudaMemcpyDeviceToHost,d2h_stream),"copy");
//  cudaEventRecord(events[10+stid],d2h_stream);

#ifdef PIPE_XFER
//if(pipe_xfer==1){
#ifndef USE_GPU_MPI
#if 0
//#ifdef USE_TOGETHER
        int ii,ii_chunk;
        for(ii_chunk=0; ii_chunk<(SIZE+together-1)/together; ii_chunk++){
          if(ii_chunk==0){
            for(ii=ii_chunk*together; ii<(ii_chunk+1)*together && ii<SIZE; ii++){
              int dest=ii^RANK;
              if(dest!=RANK){
                CHECK_CUDART( cudaMemcpyAsync((float2*)aux_host_1[stid]+dest*NZ*myNx*myNy,(float2*)aux_dev[stid]+dest*NZ*myNx*myNy,NZ*myNx*myNy*sizeof(float2),cudaMemcpyDeviceToHost,d2h_stream) );
                CHECK_CUDART( cudaEventRecord(events[30+128*stid+ii],d2h_stream) );
              }
            }
          }else{
            int min_dest=999999;
            for(ii=ii_chunk*together; ii<(ii_chunk+1)*together && ii<SIZE; ii++){
              int dest=ii^RANK;
              if(dest<min_dest) min_dest=dest;
            }
            CHECK_CUDART(cudaMemcpyAsync((float2*)aux_host_1[stid]+min_dest*NZ*myNx*myNy,(float2*)aux_dev[stid]+min_dest*NZ*myNx*myNy,together*NZ*myNx*myNy*sizeof(float2),cudaMemcpyDeviceToHost,d2h_stream));
            for(ii=ii_chunk*together; ii<(ii_chunk+1)*together && ii<SIZE; ii++){
              CHECK_CUDART( cudaEventRecord(events[30+128*stid+ii],d2h_stream) );
            }
          }
        }
#else
       int iter;
       for(iter=1; iter<SIZE; iter++){
         int dest = RANK ^ iter;
         cudaCheck(cudaMemcpyAsync((float2*)aux_host_1[stid]+dest*NZ*myNx*myNy,(float2*)aux_dev[stid]+dest*NZ*myNx*myNy,NZ*myNx*myNy*sizeof(float2),cudaMemcpyDeviceToHost,d2h_stream),"copy");
         cudaEventRecord(events[30+128*stid+iter],d2h_stream);
       }
#endif
#endif
#else
//}else{
        cudaCheck(cudaMemcpyAsync((float2*)aux_host_1[stid],(float2*)aux_dev[stid],size,cudaMemcpyDeviceToHost,d2h_stream),"copy");
        cudaEventRecord(events[10+stid],d2h_stream);
#endif
//}//pipe_xfer


}

void fftForw1T_B(float2* u1, int stid){
        int myNx = NY/SIZE;
        int myNy = NX/SIZE;
        stream_idx = stid;
        cublasCheck(cublasSetStream(cublasHandle,compute_stream),"stream");
        cufftCheck(cufftSetStream(fft1_c2c,compute_stream),"SetStream");

#ifdef PIPE_XFER
//if(pipe_xfer==1){
#ifdef USE_GPU_MPI
       int iter;
       for(iter=1; iter<SIZE; iter++){
         int dest = RANK ^ iter;
         cudaEventSynchronize(events[stid]);
         MPI_Sendrecv(aux_dev[stid]+dest*NZ*myNx*myNy, NZ*myNx*myNy, MPI_DOUBLE, dest, 0, u1+dest*NZ*myNx*myNy, NZ*myNx*myNy, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
         cudaCheck(cudaMemcpyAsync((float2*)aux_dev[stid]+dest*NZ*myNx*myNy,(float2*)u1+dest*NZ*myNx*myNy,NZ*myNx*myNy*sizeof(float2),cudaMemcpyDeviceToDevice,h2d_stream),"copy");
       }
        cudaEventRecord(events[20+stid],h2d_stream);
#else
#ifdef USE_TOGETHER
        int ii,ii_chunk;
        for(ii_chunk=0; ii_chunk<(SIZE+together-1)/together; ii_chunk++){
          if(ii_chunk==0){
#ifdef USE_NB_COMM
            for(ii=ii_chunk*together; ii<(ii_chunk+1)*together && ii<SIZE; ii++){
              int dest=ii^RANK;
              if(dest!=RANK){
                MPI_Irecv(aux_host_2[stid]+dest*NZ*myNx*myNy, NZ*myNx*myNy, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD, &recv_requests[ii]);
              }
            }
            for(ii=ii_chunk*together; ii<(ii_chunk+1)*together && ii<SIZE; ii++){
              int dest=ii^RANK;
              if(dest!=RANK){
                CHECK_CUDART( cudaEventSynchronize(events[30+128*stid+ii]) );
                MPI_Isend(aux_host_1[stid]+dest*NZ*myNx*myNy, NZ*myNx*myNy, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD, &send_requests[ii]);
              }
            }
            for(ii=ii_chunk*together; ii<(ii_chunk+1)*together && ii<SIZE; ii++){
              int dest=ii^RANK;
              if(dest!=RANK){
                MPI_Wait(&recv_requests[ii],&recv_status[ii]);
                CHECK_CUDART( cudaMemcpyAsync((float2*)aux_dev[stid]+dest*NZ*myNx*myNy,(float2*)aux_host_2[stid]+dest*NZ*myNx*myNy,NZ*myNx*myNy*sizeof(float2),cudaMemcpyHostToDevice,h2d_stream) );
              }
            }
#else
            for(ii=ii_chunk*together; ii<(ii_chunk+1)*together && ii<SIZE; ii++){
              int dest=ii^RANK;
              if(dest!=RANK){
                CHECK_CUDART( cudaEventSynchronize(events[30+128*stid+ii]) );
START_RANGE_ASYNC("MPI",3)
                MPI_Sendrecv(aux_host_1[stid]+dest*NZ*myNx*myNy, NZ*myNx*myNy, MPI_DOUBLE, dest, 0, aux_host_2[stid]+dest*NZ*myNx*myNy, NZ*myNx*myNy, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
END_RANGE_ASYNC
                CHECK_CUDART(cudaMemcpyAsync((float2*)aux_dev[stid]+dest*NZ*myNx*myNy,(float2*)aux_host_2[stid]+dest*NZ*myNx*myNy,NZ*myNx*myNy*sizeof(float2),cudaMemcpyHostToDevice,h2d_stream));
              }
            }
#endif
          }else{
#ifdef USE_NB_COMM
            for(ii=ii_chunk*together; ii<(ii_chunk+1)*together && ii<SIZE; ii++){
              int dest=ii^RANK;
              MPI_Irecv(aux_host_2[stid]+dest*NZ*myNx*myNy, NZ*myNx*myNy, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD, &recv_requests[ii]);
            }
            int min_dest=999999;
            for(ii=ii_chunk*together; ii<(ii_chunk+1)*together && ii<SIZE; ii++){
              int dest=ii^RANK;
              if(dest<min_dest) min_dest=dest;
              CHECK_CUDART( cudaEventSynchronize(events[30+128*stid+ii]) );
              MPI_Isend(aux_host_1[stid]+dest*NZ*myNx*myNy, NZ*myNx*myNy, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD, &send_requests[ii]);
            }
            for(ii=ii_chunk*together; ii<(ii_chunk+1)*together && ii<SIZE; ii++){
                int dest=ii^RANK;
                MPI_Wait(&recv_requests[ii],&recv_status[ii]);
                CHECK_CUDART( cudaMemcpyAsync((float2*)aux_dev[stid]+dest*NZ*myNx*myNy,(float2*)aux_host_2[stid]+dest*NZ*myNx*myNy,NZ*myNx*myNy*sizeof(float2),cudaMemcpyHostToDevice,h2d_stream) );
            }
            //CHECK_CUDART( cudaMemcpyAsync((float2*)aux_dev[stid]+min_dest*NZ*myNx*myNy,(float2*)aux_host_2[stid]+min_dest*NZ*myNx*myNy,together*NZ*myNx*myNy*sizeof(float2),cudaMemcpyHostToDevice,h2d_stream) );
#else
            int min_dest=999999;
            for(ii=ii_chunk*together; ii<(ii_chunk+1)*together && ii<SIZE; ii++){
              int dest=ii^RANK;
              if(dest<min_dest) min_dest=dest;
              CHECK_CUDART( cudaEventSynchronize(events[30+128*stid+ii]) );
START_RANGE_ASYNC("MPI",3)
              MPI_Sendrecv(aux_host_1[stid]+dest*NZ*myNx*myNy, NZ*myNx*myNy, MPI_DOUBLE, dest, 0, aux_host_2[stid]+dest*NZ*myNx*myNy, NZ*myNx*myNy, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
END_RANGE_ASYNC
            }
            CHECK_CUDART(cudaMemcpyAsync((float2*)aux_dev[stid]+min_dest*NZ*myNx*myNy,(float2*)aux_host_2[stid]+min_dest*NZ*myNx*myNy,together*NZ*myNx*myNy*sizeof(float2),cudaMemcpyHostToDevice,h2d_stream));
#endif
          }
        }
#else
#ifdef USE_NB_COMM
       int iter;
       for(iter=1; iter<SIZE; iter++){
         int dest = RANK ^ iter;
         MPI_Irecv(aux_host_2[stid]+dest*NZ*myNx*myNy, NZ*myNx*myNy, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD, &recv_requests[iter]);
       }
       for(iter=1; iter<SIZE; iter++){
         int dest = RANK ^ iter;
         cudaEventSynchronize(events[30+128*stid+iter]);
         MPI_Send(aux_host_1[stid]+dest*NZ*myNx*myNy, NZ*myNx*myNy, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);//, &send_requests[iter]);
/*       }
       
       for(iter=1; iter<SIZE; iter++){
         int dest = RANK ^ iter;
*/
         MPI_Wait(&recv_requests[iter],&recv_status[iter]);
         CHECK_CUDART( cudaMemcpyAsync((float2*)aux_dev[stid]+dest*NZ*myNx*myNy,(float2*)aux_host_2[stid]+dest*NZ*myNx*myNy,NZ*myNx*myNy*sizeof(float2),cudaMemcpyHostToDevice,h2d_stream) );
       }
#else
       int iter;
       for(iter=1; iter<SIZE; iter++){
         int dest = RANK ^ iter;
         cudaEventSynchronize(events[30+128*stid+iter]);
START_RANGE_ASYNC("MPI",3)
         MPI_Sendrecv(aux_host_1[stid]+dest*NZ*myNx*myNy, NZ*myNx*myNy, MPI_DOUBLE, dest, 0, aux_host_2[stid]+dest*NZ*myNx*myNy, NZ*myNx*myNy, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
END_RANGE_ASYNC
         cudaCheck(cudaMemcpyAsync((float2*)aux_dev[stid]+dest*NZ*myNx*myNy,(float2*)aux_host_2[stid]+dest*NZ*myNx*myNy,NZ*myNx*myNy*sizeof(float2),cudaMemcpyHostToDevice,h2d_stream),"copy");
       }
#endif
#endif
        cudaEventRecord(events[20+stid],h2d_stream);
#endif
#else
//}else{
        cudaEventSynchronize(events[10+stid]);
START_RANGE_ASYNC("MPI",3)
        MPIErr = MPI_Alltoall(aux_host_1[stid],NZ*myNx*myNy,MPI_DOUBLE,
                              aux_host_2[stid],NZ*myNx*myNy,MPI_DOUBLE,
                              MPI_COMM_WORLD);
END_RANGE_ASYNC
        mpiCheck(MPIErr,"transpoze");
        cudaCheck(cudaMemcpyAsync((float2*)aux_dev[stid],(float2*)aux_host_2[stid],size,cudaMemcpyHostToDevice,h2d_stream),"copy");
        cudaEventRecord(events[20+stid],h2d_stream);
#endif
//}//pipe_xfer

        cudaStreamWaitEvent(compute_stream,events[20+stid],0);
        //transposeBatched(u1,(const float2*)aux_dev[stid],myNx*NZ,myNy,SIZE);
        //transposeBatched(aux_dev[stid],(const float2*)u1,myNy,NZ,SIZE*myNx);
        //transpose(u1,(const float2*)aux_dev[stid],myNy*NZ,SIZE*myNx);
        trans_zyx_yblock_to_yzx(aux_dev[stid], u1, compute_stream);

        cufftCheck(cufftExecC2C(fft1_c2c,u1,aux_dev[stid],CUFFT_FORWARD),"forward transform");
        //transpose_B(u1,aux_dev[stid]);
        trans_yzx_to_zyx(aux_dev[stid], u1, compute_stream);
}

void fftForw1T(float2* u1){

		//FFT 2D
START_RANGE("CUFFT",3)	
		cufftCheck(cufftExecR2C(fft2_r2c,(float*)u1,(float2*)u1),"forward transform");
END_RANGE
		//cudaCheck(cudaMemcpy((float2*)aux_host1,(float2*)u1,size,cudaMemcpyDeviceToHost),"copy");
			
		//Transpose from [y,x,z] to [x,z,y]
		
		//mpiCheck(chxyz2yzx((double *)aux_host1,(double*)aux_host2,NY,NX,NZ,RANK,SIZE),"T");

		
		//cudaCheck(cudaMemcpy((float2*)u1,(float2*)aux_host2,size,cudaMemcpyHostToDevice),"copy");
START_RANGE("transXYZ2YZX",1)
		transposeXYZ2YZX(u1,NY,NX,NZ,RANK,SIZE);
END_RANGE
		//FFT 1D
START_RANGE("CUFFT",3)
		cufftCheck(cufftExecC2C(fft1_c2c,u1,AUX,CUFFT_FORWARD),"forward transform");	 
END_RANGE
		//Transpose from [x,z,y] to [x,y,z]		
START_RANGE("transB",5)
		transpose_B(u1,AUX);
END_RANGE
}

void fftBackMultiple(float2* u1,float2* u2,float2* u3,float2* u4,float2* u5,float2* u6){

		//Handle buffers

		buffer[0]=u1;
		buffer[1]=u2;
		buffer[2]=u3;
		buffer[3]=u4;
		buffer[4]=u5;
		buffer[5]=u6;

		//FIRST SIX FFTS 1D


		for(int j=0;j<6;j++){

		//Transpose
		cublasCheck(cublasSetStream(cublasHandle,STREAMS[j]),"stream");
		transpose_A(AUX,buffer[j]);

	
		//FFT 1D

		cufftCheck(cufftSetStream(fft1_c2c,STREAMS[j]),"SetStream");
		cufftCheck(cufftExecC2C(fft1_c2c,AUX,buffer[j],CUFFT_INVERSE),"forward transform");	
		cudaCheck(cudaMemcpyAsync((float2*)aux_host_1[j],(float2*)buffer[j],size,cudaMemcpyDeviceToHost,STREAMS[j]),"copy");
		
		}


		//COPY TO CPU
		
		

		for(int j=0;j<6;j++){

		cudaCheck(cudaStreamSynchronize(STREAMS[j]),"event_synchronise"); 	
		mpiCheck(chyzx2xyz((double *)aux_host_1[j],(double*)aux_host_2[j],NY,NX,NZ,RANK,SIZE),"T");
 

		cudaCheck(cudaMemcpyAsync((float2*)buffer[j],(float2*)aux_host_2[j],size,cudaMemcpyHostToDevice,STREAMS[j]),"copy");
	
		cufftCheck(cufftSetStream(fft2_c2r,STREAMS[j]),"SetStream");
		cufftCheck(cufftExecC2R(fft2_c2r,buffer[j],(float*)buffer[j]),"forward transform");

		}

		//Device synchronise

		cudaCheck(cudaDeviceSynchronize(),"synchro"); 
		

		return;

}


void fftForwMultiple(float2* u1,float2* u2,float2* u3){


		//Handle buffers

		buffer[0]=u1;
		buffer[1]=u2;
		buffer[2]=u3;


		//FIRST THREE FFTS 2D

		

		for(int j=0;j<3;j++){

		//FFT 2D

		cufftCheck(cufftSetStream(fft2_r2c,STREAMS[j]),"SetStream");		
		cufftCheck(cufftExecR2C(fft2_r2c,(float*)buffer[j],(float2*)buffer[j]),"forward transform");

		cudaCheck(cudaMemcpyAsync((float2*)aux_host_1[j],(float2*)buffer[j],size,cudaMemcpyDeviceToHost,STREAMS[j]),"copy");
		
		}


		
		for(int j=0;j<3;j++){

		cudaCheck(cudaStreamSynchronize(STREAMS[j]),"event_synchronise"); 
		
		mpiCheck(chxyz2yzx((double *)aux_host_1[j],(double*)aux_host_2[j],NY,NX,NZ,RANK,SIZE),"T");

		//Copy to gpu
		cudaCheck(cudaMemcpyAsync((float2*)buffer[j],(float2*)aux_host_2[j],size,cudaMemcpyHostToDevice,STREAMS[j]),"copy"); 		

		//FFT 1D
		cufftCheck(cufftSetStream(fft1_c2c,STREAMS[j]),"SetStream");
		cufftCheck(cufftExecC2C(fft1_c2c,buffer[j],AUX,CUFFT_FORWARD),"forward transform");	 
		
		//Transpose
		cublasCheck(cublasSetStream(cublasHandle,STREAMS[j]),"stream");
		transpose_A(buffer[j],AUX);
		
		}

		//Device synchronise

		cudaCheck(cudaDeviceSynchronize(),"synchro"); 
		

		return;


}





void calcUmaxV2(vectorField t,float* ux,float* uy,float* uz)
{


	int size_l=2*NXSIZE*NY*NZ;
	int index;

	cublasCheck(cublasIsamax (cublasHandle,size_l, (const float *)ux,1,&index),"Isa");
	cudaCheck(cudaMemcpy(&ux,(float*)ux+index-1, sizeof(float), cudaMemcpyDeviceToHost),"MemInfo_isa");
	
	cublasCheck(cublasIsamax (cublasHandle,size_l, (const float *)uy,1,&index),"Isa");
	cudaCheck(cudaMemcpy(&uy,(float*)uy+index-1, sizeof(float), cudaMemcpyDeviceToHost),"MemInfo_isa");
	
	cublasCheck(cublasIsamax (cublasHandle,size_l, (const float *)uz,1,&index),"Isa");
	cudaCheck(cudaMemcpy(&uz,(float*)uz+index-1, sizeof(float), cudaMemcpyDeviceToHost),"MemInfo_isa");

	
	*ux=fabs(*ux);
	*uy=fabs(*uy);
	*uz=fabs(*uz);
	

	//MPI reduce
	reduceMAX(ux,uy,uz);

	return;

}



float sumElementsV2(float2* buffer_1){

	//destroza lo que haya en el buffer

	float sum_all=0;
	

	cufftCheck(cufftExecR2C(fft2_r2c,(float*)(buffer_1),buffer_1),"forward transform");


	for(int i=0;i<NXSIZE;i++){

	cudaCheck(cudaMemcpy((float2*)sum+i,(float2*)buffer_1+i*NY*NZ,sizeof(float2),cudaMemcpyDeviceToHost),"MemInfo1");

	};
	
	for(int i=1;i<NXSIZE;i++){

	sum[0].x+=sum[i].x;
	}

	//MPI SUM

	reduceSUM((float*)sum,&sum_all);


	return sum_all;

};



