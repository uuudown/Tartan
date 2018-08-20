//
//  b2r_halo_sync.cu
//  pack and unpack ghost zone data 
//
//  Created by Zhengchun Liu on 3/15/16.
//  Copyright © 2016 research. All rights reserved.
//

#ifndef b2r_halo_sync_h
#define b2r_halo_sync_h

#include <iostream>
#include <cstring>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "b2r_config.h"

#define CUDA_BLOCK_SIZE  32

using namespace std;

#define cudaErrchk(ans)  cudaAssert((ans), __FILE__, __LINE__) 
inline void cudaAssert(cudaError_t code, string file, int line){
    if (code != cudaSuccess){
        cerr << "CUDA Error: " << cudaGetErrorString(code) << "; file: " << file << ", line:" << line << endl;
        exit(-1);
    }
}

/*
***************************************************************************************************
*                                           Global Variables                                      *
***************************************************************************************************
*/
extern int B2R_B_X, B2R_B_Y, B2R_B_Z;
extern int B2R_BLOCK_SIZE_X, B2R_BLOCK_SIZE_Y, B2R_BLOCK_SIZE_Z;
extern int B2R_R;
extern int BLOCK_DIM_X, BLOCK_DIM_Y, BLOCK_DIM_Z;

__constant__ int d_B2R_B_X, d_B2R_B_Y, d_B2R_B_Z;
__constant__ int d_B2R_BLOCK_SIZE_X, d_B2R_BLOCK_SIZE_Y, d_B2R_BLOCK_SIZE_Z;
__constant__ int d_B2R_R;

// send/receive buffer, could be reused for different dimension
char * h_dp_pad_send_buf[2], * h_dp_pad_recv_buf[2];

CELL_DT ** d_pad_send_buf;
CELL_DT ** d_pad_recv_buf;

char * h_pad_send_buf[2];
char * h_pad_recv_buf[2];

extern void b2r_sync_wait(int sent_cnt, int recv_cnt);
extern int b2r_send_pad_x(int tstep, char **pad_send_buf);
extern int b2r_recv_pad_x(int tstep, char **pad_recv_buf);

extern int b2r_send_pad_y(int tstep, char **pad_send_buf);
extern int b2r_recv_pad_y(int tstep, char **pad_recv_buf);

extern int b2r_send_pad_z(int tstep, char **pad_send_buf);
extern int b2r_recv_pad_z(int tstep, char **pad_recv_buf);

extern char b2r_halo_update[6];
/*
 ***************************************************************************************************
 * func   name: b2r_sync_init
 * description: check the avaliability of MPICH_RDMA_ENABLED_CUDA
                allocate device memory for send/receive buffer
 parameters :
 *             none
 * return: none
 ***************************************************************************************************
 */
void b2r_sync_init()
{
    // Ensure that RDMA ENABLED CUDA is set correctly
    //int direct = getenv("MPICH_RDMA_ENABLED_CUDA")==NULL?0:atoi(getenv ("MPICH_RDMA_ENABLED_CUDA"));
    //if(direct != 1){
    //cout << "MPICH_RDMA_ENABLED_CUDA not enabled!" << endl;
    //exit (-1);
    //}

    const int Rd = B2R_R * B2R_D;
    int buffer_size = sizeof(CELL_DT) * max(max(Rd*B2R_B_Y*B2R_B_Z, B2R_BLOCK_SIZE_X*Rd*B2R_B_Z), 
                                            B2R_BLOCK_SIZE_X*B2R_BLOCK_SIZE_Y*Rd);                                                                          
    cudaErrchk( cudaMalloc((void**)&(h_dp_pad_send_buf[0]), buffer_size) ); 
    cudaErrchk( cudaMalloc((void**)&(h_dp_pad_send_buf[1]), buffer_size) ); 
    cudaErrchk( cudaMalloc((void**)&(h_dp_pad_recv_buf[0]), buffer_size) ); 
    cudaErrchk( cudaMalloc((void**)&(h_dp_pad_recv_buf[1]), buffer_size) );
    
    cudaErrchk( cudaMalloc((void**)&(d_pad_send_buf), 2*sizeof(CELL_DT *)) );
    cudaErrchk( cudaMalloc((void**)&(d_pad_recv_buf), 2*sizeof(CELL_DT *)) );
    cudaErrchk( cudaMemcpy((void *)d_pad_send_buf,  (void *)h_dp_pad_send_buf, 2*sizeof(CELL_DT *), cudaMemcpyHostToDevice) );
    cudaErrchk( cudaMemcpy((void *)d_pad_recv_buf,  (void *)h_dp_pad_recv_buf, 2*sizeof(CELL_DT *), cudaMemcpyHostToDevice) );

    //------------------------------------------------------------
    // sending buffer                                        
    h_pad_send_buf[0] = new char[buffer_size]();    // direction 1(to: left/up/infront)
    h_pad_send_buf[1] = new char[buffer_size]();    // direction 2(to: right/down/behind)
    
    //cudaErrchk( cudaMallocHost((void**)&(h_pad_send_buf[0]), buffer_size ) );
    //cudaErrchk( cudaMallocHost((void**)&(h_pad_send_buf[1]), buffer_size ) );

    // receiving buffer
    h_pad_recv_buf[0] = new char[buffer_size]();    // direction 1(from: left/up/infront)
    h_pad_recv_buf[1] = new char[buffer_size]();    // direction 2(from: right/down/behind)    

    //cudaErrchk( cudaMallocHost((void**)&(h_pad_recv_buf[0]), buffer_size ) );
    //cudaErrchk( cudaMallocHost((void**)&(h_pad_recv_buf[1]), buffer_size ) );

    //------------------------------------------------------------
    
    cudaErrchk( cudaMemcpyToSymbol(d_B2R_B_X, &B2R_B_X, sizeof(int), 0, cudaMemcpyHostToDevice) );
    cudaErrchk( cudaMemcpyToSymbol(d_B2R_B_Y, &B2R_B_Y, sizeof(int), 0, cudaMemcpyHostToDevice) );
    cudaErrchk( cudaMemcpyToSymbol(d_B2R_B_Z, &B2R_B_Z, sizeof(int), 0, cudaMemcpyHostToDevice) );
    
    cudaErrchk( cudaMemcpyToSymbol(d_B2R_BLOCK_SIZE_X, &B2R_BLOCK_SIZE_X, sizeof(int), 0, cudaMemcpyHostToDevice) );
    cudaErrchk( cudaMemcpyToSymbol(d_B2R_BLOCK_SIZE_Y, &B2R_BLOCK_SIZE_Y, sizeof(int), 0, cudaMemcpyHostToDevice) );
    cudaErrchk( cudaMemcpyToSymbol(d_B2R_BLOCK_SIZE_Z, &B2R_BLOCK_SIZE_Z, sizeof(int), 0, cudaMemcpyHostToDevice) );  
    
    cudaErrchk( cudaMemcpyToSymbol(d_B2R_R,            &B2R_R,            sizeof(int), 0, cudaMemcpyHostToDevice) );                                            
}
/*
 ***************************************************************************************************
 * func   name: b2r_sync_finalize
 * description: 
 parameters :
 *             none
 * return: none
 ***************************************************************************************************
 */
void b2r_sync_finalize(){
    for(int i=0; i<2; i++){
        cudaFree( h_dp_pad_recv_buf[i] );
        cudaFree( h_dp_pad_recv_buf[i] ); 
    }
    cudaFree(d_pad_send_buf);
    cudaFree(d_pad_recv_buf);

    //--------------------------------------
    //cudaErrchk( cudaFreeHost(h_pad_send_buf[0]) );
    //cudaErrchk( cudaFreeHost(h_pad_send_buf[1]) );
    //cudaErrchk( cudaFreeHost(h_pad_recv_buf[0]) );
    //cudaErrchk( cudaFreeHost(h_pad_recv_buf[1]) );
    //--------------------------------------
    free(h_pad_send_buf[0]);
    free(h_pad_send_buf[1]);
    free(h_pad_recv_buf[0]);
    free(h_pad_recv_buf[1]);

}
/*
 ***************************************************************************************************
 * func   name: b2r_sync_flag
 * description: check the avaliability of synchronizing halo subparts
                results will be wrote back to global variable
                b2r_halo_update, idex definition refer to SYNC_M_CNT
 parameters :
 *             rank number of MPI process
 * return: none
 ***************************************************************************************************
 */
void b2r_sync_flag(int rank)
{
    int block_id_x = (rank % (BLOCK_DIM_X * BLOCK_DIM_Y)) % BLOCK_DIM_X;
    int block_id_y = (rank % (BLOCK_DIM_X * BLOCK_DIM_Y)) / BLOCK_DIM_X;
    int block_id_z = (rank / (BLOCK_DIM_X * BLOCK_DIM_Y));
    
    memset(b2r_halo_update, 0, sizeof(char) * sizeof(b2r_halo_update)); 
    // left B x R*D
    if(block_id_x >= 1)
    {
        b2r_halo_update[0] = 1;
    }

    // right B x R*D
    if(block_id_x <  BLOCK_DIM_X-1)
    {
        b2r_halo_update[1] = 1;
    }

    // up
    if(block_id_y >= 1)
    {
        b2r_halo_update[2] = 1;
    }    
    
    // down
    if(block_id_y <  BLOCK_DIM_Y-1)
    {
        b2r_halo_update[3] = 1;
    }
#if _ENV_3D_    
    // in front
    if(block_id_z >= 1)
    {
        b2r_halo_update[4] = 1;
    }
    
    // behind
    if(block_id_z < BLOCK_DIM_Z-1)
    {
        b2r_halo_update[5] = 1;
    }
#endif
}

/*
 ***************************************************************************************************
 * func   name: sub_matrix_pack_x
 * description: pack sub cube in X direction to one dimensional array / buffer for sending(sync) 
                to other process / block 
 * parameters :
 *             none
 * return: none
 ***************************************************************************************************
 */
__global__ void sub_matrix_pack_x(CELL_DT *d_grid, CELL_DT **pad_send_buf, bool neg, bool pos)
{
    const int Rd = d_B2R_R * B2R_D;
    // distance between 2D slice (in elements)
    const int stride = d_B2R_BLOCK_SIZE_X * d_B2R_BLOCK_SIZE_Y;
    const int stride_vd = d_B2R_B_Z * d_B2R_B_Y;
    int ix = blockIdx.x*blockDim.x + threadIdx.x;
    int iy = blockIdx.y*blockDim.y + threadIdx.y;
    if(ix >= d_B2R_B_Z || iy >= d_B2R_B_Y){
        return;
    }
    int gy = iy + Rd;
    int gz = ix + Rd;
    // left R*D * d_B2R_B_Y * d_B2R_B_Z
    for (int gx = Rd; neg && gx < 2*Rd; ++gx){
        pad_send_buf[0][(gx-Rd)*stride_vd + iy*d_B2R_B_Z+ix] = d_grid[stride*gz + gy*d_B2R_BLOCK_SIZE_X + gx];
    }
    // right R*D * d_B2R_B_Y * d_B2R_B_Z
    for (int gx = d_B2R_B_X; pos && gx < d_B2R_B_X+Rd; ++gx){
        pad_send_buf[1][(gx-d_B2R_B_X)*stride_vd + iy*d_B2R_B_Z+ix] = d_grid[stride*gz + gy*d_B2R_BLOCK_SIZE_X + gx];
    }    
}
/*
 ***************************************************************************************************
 * func   name: sub_matrix_pack_y
 * description: pack sub cube in Y direction to one dimensional array / buffer for sending(sync) 
                to other process / block 
 * parameters :
 *             none
 * return: none
 ***************************************************************************************************
 */
__global__ void sub_matrix_pack_y(CELL_DT *d_grid, CELL_DT **pad_send_buf, bool neg, bool pos)
{
    const int Rd = d_B2R_R * B2R_D;
    // distance between 2D slice (in elements)
    const int stride = d_B2R_BLOCK_SIZE_X * d_B2R_BLOCK_SIZE_Y;
    const int stride_vd = d_B2R_BLOCK_SIZE_X * d_B2R_B_Z;
    int ix = blockIdx.x*blockDim.x + threadIdx.x;
    int iy = blockIdx.y*blockDim.y + threadIdx.y;
    if(ix >= d_B2R_BLOCK_SIZE_X || iy >= d_B2R_B_Z){
        return;
    }

    int gx = ix;
    int gz = iy + Rd;
    // up B2R_BLOCK_SIZE_X * R*D * d_B2R_B_Z
    for (int gy = Rd; neg && gy < 2*Rd; ++gy){
        pad_send_buf[0][(gy-Rd)*stride_vd + iy*d_B2R_BLOCK_SIZE_X+ix] = d_grid[stride*gz + gy*d_B2R_BLOCK_SIZE_X + gx];
    }
    // down B2R_BLOCK_SIZE_X * R*D * d_B2R_B_Z
    for (int gy = d_B2R_B_Y; pos && gy < d_B2R_B_Y+Rd; ++gy){
        pad_send_buf[1][(gy-d_B2R_B_Y)*stride_vd + iy*d_B2R_BLOCK_SIZE_X+ix] = d_grid[stride*gz + gy*d_B2R_BLOCK_SIZE_X + gx];
    }  
}
/*
 ***************************************************************************************************
 * func   name: sub_matrix_pack_z
 * description: pack sub cube in Z direction to one dimensional array / buffer for sending(sync) 
                to other process / block 
 * parameters :
 *             none
 * return: none
 ***************************************************************************************************
 */
__global__ void sub_matrix_pack_z(CELL_DT *d_grid, CELL_DT **pad_send_buf, bool neg, bool pos)
{
    const int Rd = d_B2R_R * B2R_D;
    // distance between 2D slice (in elements)
    const int stride = d_B2R_BLOCK_SIZE_X * d_B2R_BLOCK_SIZE_Y;
    int ix = blockIdx.x*blockDim.x + threadIdx.x;
    int iy = blockIdx.y*blockDim.y + threadIdx.y;
    if(ix >= d_B2R_BLOCK_SIZE_X || iy >= d_B2R_BLOCK_SIZE_Y){
        return;
    }

    int gx = ix;
    int gy = iy;
    // infront B2R_BLOCK_SIZE_X * R*D * B2R_BLOCK_SIZE_Z
    for (int gz = Rd; neg && gz < 2*Rd; ++gz){
        pad_send_buf[0][(gz-Rd)*stride + iy*d_B2R_BLOCK_SIZE_X+ix] = d_grid[stride*gz + gy*d_B2R_BLOCK_SIZE_X + gx];
    }
    
    // behind B2R_BLOCK_SIZE_X * R*D * B2R_BLOCK_SIZE_Z
    for (int gz = d_B2R_B_Z; pos && gz < d_B2R_B_Z+Rd; ++gz){
        pad_send_buf[1][(gz-d_B2R_B_Z)*stride + iy*d_B2R_BLOCK_SIZE_X+ix] = d_grid[stride*gz + gy*d_B2R_BLOCK_SIZE_X + gx];
    }
}
/*
 ***************************************************************************************************
 * func   name: sub_matrix_unpack_x
 * description: unpack received data (from left and right) to ghost region
 * parameters :
 *             none
 * return: none
 ***************************************************************************************************
 */
__global__ void sub_matrix_unpack_x(CELL_DT *d_grid, CELL_DT **pad_recv_buf, bool neg, bool pos)
{
    const int Rd = d_B2R_R * B2R_D;
    // distance between 2D slice (in elements)
    const int stride = d_B2R_BLOCK_SIZE_X * d_B2R_BLOCK_SIZE_Y;
    const int stride_vd = d_B2R_B_Z * d_B2R_B_Y;
    int ix = blockIdx.x*blockDim.x + threadIdx.x;
    int iy = blockIdx.y*blockDim.y + threadIdx.y;
    if(ix >= d_B2R_B_Z || iy >= d_B2R_B_Y){
        return;
    }
    int gy = iy + Rd;
    int gz = ix + Rd;
    // from left R*D * d_B2R_B_Y * d_B2R_B_Z
    for (int gx = 0; neg && gx < Rd; ++gx){
        d_grid[stride*gz + gy*d_B2R_BLOCK_SIZE_X + gx] = pad_recv_buf[0][gx*stride_vd + iy*d_B2R_B_Z+ix];
    }
    // from right R*D * d_B2R_B_Y * d_B2R_B_Z
    for (int gx = d_B2R_B_X+Rd; pos && gx < d_B2R_BLOCK_SIZE_X; ++gx){
        d_grid[stride*gz + gy*d_B2R_BLOCK_SIZE_X + gx] = pad_recv_buf[1][(gx-d_B2R_B_X-Rd)*stride_vd + iy*d_B2R_B_Z+ix];
    }   
}

/*
 ***************************************************************************************************
 * func   name: sub_matrix_unpack_y
 * description: unpack received data (from up and down) to ghost region
 * parameters :
 *             none
 * return: none
 ***************************************************************************************************
 */
__global__ void sub_matrix_unpack_y(CELL_DT *d_grid, CELL_DT **pad_recv_buf, bool neg, bool pos)
{
    const int Rd = d_B2R_R * B2R_D;
    // distance between 2D slice (in elements)
    const int stride = d_B2R_BLOCK_SIZE_X * d_B2R_BLOCK_SIZE_Y;
    const int stride_vd = d_B2R_BLOCK_SIZE_X * d_B2R_B_Z;
    int ix = blockIdx.x*blockDim.x + threadIdx.x;
    int iy = blockIdx.y*blockDim.y + threadIdx.y;
    if(ix >= d_B2R_BLOCK_SIZE_X || iy >= d_B2R_B_Z){
        return;
    }

    int gx = ix;
    int gz = iy + Rd;
    // from up B2R_BLOCK_SIZE_X * R*D * d_B2R_B_Z
    for (int gy = 0; neg && gy < Rd; ++gy){
        d_grid[stride*gz + gy*d_B2R_BLOCK_SIZE_X + gx] = pad_recv_buf[0][gy*stride_vd + iy*d_B2R_BLOCK_SIZE_X+ix];
    }
    // from down B2R_BLOCK_SIZE_X * R*D * d_B2R_B_Z
    for (int gy = d_B2R_B_Y+Rd; pos && gy < d_B2R_BLOCK_SIZE_Y; ++gy){
        d_grid[stride*gz + gy*d_B2R_BLOCK_SIZE_X + gx] = pad_recv_buf[1][(gy-d_B2R_B_Y-Rd)*stride_vd + iy*d_B2R_BLOCK_SIZE_X+ix];
    }      
}

/*
 ***************************************************************************************************
 * func   name: sub_matrix_unpack_z
 * description: unpack received data (from infront and behind) to ghost region
 * parameters :
 *             none
 * return: none
 ***************************************************************************************************
 */
__global__ void sub_matrix_unpack_z(CELL_DT *d_grid, CELL_DT **pad_recv_buf, bool neg, bool pos)
{
    const int Rd = d_B2R_R * B2R_D;
    // distance between 2D slice (in elements)
    const int stride = d_B2R_BLOCK_SIZE_X * d_B2R_BLOCK_SIZE_Y;
    int ix = blockIdx.x*blockDim.x + threadIdx.x;
    int iy = blockIdx.y*blockDim.y + threadIdx.y;
    if(ix >= d_B2R_BLOCK_SIZE_X || iy >= d_B2R_BLOCK_SIZE_Y){
        return;
    }

    int gx = ix;
    int gy = iy;
    // from infront B2R_BLOCK_SIZE_X * R*D * B2R_BLOCK_SIZE_Z
    for (int gz = 0; neg && gz < Rd; ++gz){
        d_grid[stride*gz + gy*d_B2R_BLOCK_SIZE_X + gx] = pad_recv_buf[0][gz*stride + iy*d_B2R_BLOCK_SIZE_X+ix];
    }
    
    // from behind B2R_BLOCK_SIZE_X * R*D * B2R_BLOCK_SIZE_Z
    for (int gz = d_B2R_B_Z+Rd; pos && gz < d_B2R_BLOCK_SIZE_Z; ++gz){
        d_grid[stride*gz + gy*d_B2R_BLOCK_SIZE_X + gx] = pad_recv_buf[1][(gz-d_B2R_B_Z-Rd)*stride + iy*d_B2R_BLOCK_SIZE_X+ix];
    }
}
/*
***************************************************************************************************
* func   name: boundary_updating_direct
* description: update subdomain boundaries via MPI
* parameters :
*             none
* return: none
***************************************************************************************************
*/
void boundary_updating_direct(int time_step, CELL_DT *d_grid)
{
    int send_req_cnt, recv_req_cnt;
    // updating ghost zone in X direction       
    // Launch configuration:
    // cout << "about to sync for " << time_step << endl;
    dim3 dimBlock_x(CUDA_BLOCK_SIZE, CUDA_BLOCK_SIZE, 1);
    dim3 dimGrid_x(ceil((float)B2R_B_Z/CUDA_BLOCK_SIZE), ceil((float)B2R_B_Y/CUDA_BLOCK_SIZE), 1);
    // invoke cuda threads to copy corresponding block data to halo sync buffer
    sub_matrix_pack_x<<<dimGrid_x, dimBlock_x>>>(d_grid, d_pad_send_buf, b2r_halo_update[0]==1, b2r_halo_update[1]==1);   
    cudaErrchk( cudaDeviceSynchronize() );
    send_req_cnt = b2r_send_pad_x(time_step, h_dp_pad_send_buf);               // send halos via non-block MPI 
    recv_req_cnt = b2r_recv_pad_x(time_step, h_dp_pad_recv_buf);               // receive halos via non-block MPI 
    b2r_sync_wait(send_req_cnt, recv_req_cnt);
    // invoke cuda threads to copy received block data to halos
    sub_matrix_unpack_x<<<dimGrid_x, dimBlock_x>>>(d_grid, d_pad_recv_buf, b2r_halo_update[0]==1, b2r_halo_update[1]==1);
    cudaErrchk( cudaDeviceSynchronize() );
    
    // updating ghost zone in Y direction
    dim3 dimBlock_y(CUDA_BLOCK_SIZE, CUDA_BLOCK_SIZE, 1);
    dim3 dimGrid_y(ceil((float)B2R_BLOCK_SIZE_X/CUDA_BLOCK_SIZE), ceil((float)B2R_B_Y/CUDA_BLOCK_SIZE), 1);    
    sub_matrix_pack_y<<<dimGrid_y, dimBlock_y>>>(d_grid, d_pad_send_buf, b2r_halo_update[2]==1, b2r_halo_update[3]==1);   
    cudaErrchk( cudaDeviceSynchronize() );
    send_req_cnt = b2r_send_pad_y(time_step, h_dp_pad_send_buf);               // send halos via non-block MPI 
    recv_req_cnt = b2r_recv_pad_y(time_step, h_dp_pad_recv_buf);               // receive halos via non-block MPI 
    b2r_sync_wait(send_req_cnt, recv_req_cnt);
    sub_matrix_unpack_y<<<dimGrid_y, dimBlock_y>>>(d_grid, d_pad_recv_buf, b2r_halo_update[2]==1, b2r_halo_update[3]==1);
    cudaErrchk( cudaDeviceSynchronize() );
    
    // updating ghost zone in Z direction
#if _ENV_3D_        
    dim3 dimBlock_z(CUDA_BLOCK_SIZE, CUDA_BLOCK_SIZE, 1);
    dim3 dimGrid_z(ceil((float)B2R_BLOCK_SIZE_X/CUDA_BLOCK_SIZE), ceil((float)B2R_BLOCK_SIZE_Y/CUDA_BLOCK_SIZE), 1);  
    
    sub_matrix_pack_z<<<dimGrid_z, dimBlock_z>>>(d_grid, d_pad_send_buf, b2r_halo_update[4]==1, b2r_halo_update[5]==1);   
    cudaErrchk( cudaDeviceSynchronize() );
    send_req_cnt = b2r_send_pad_z(time_step, h_dp_pad_send_buf);               // send halos via non-block MPI 
    recv_req_cnt = b2r_recv_pad_z(time_step, h_dp_pad_recv_buf);               // receive halos via non-block MPI 
    b2r_sync_wait(send_req_cnt, recv_req_cnt);
    sub_matrix_unpack_z<<<dimGrid_z, dimBlock_z>>>(d_grid, d_pad_recv_buf, b2r_halo_update[4]==1, b2r_halo_update[5]==1);
    cudaErrchk( cudaDeviceSynchronize() );
#endif    
}

/*
***************************************************************************************************
* func   name: boundary_updating
* description: update subdomain boundaries via MPI
* parameters :
*             none
* return: none
***************************************************************************************************
*/
void boundary_updating(int time_step, CELL_DT *d_grid)
{
    int send_req_cnt, recv_req_cnt;
    const int Rd = B2R_R * B2R_D;
    int buffer_size = sizeof(CELL_DT) * max(max(Rd*B2R_B_Y*B2R_B_Z, B2R_BLOCK_SIZE_X*Rd*B2R_B_Z), 
                                            B2R_BLOCK_SIZE_X*B2R_BLOCK_SIZE_Y*Rd); 
    // updating ghost zone in X direction       
    // Launch configuration:
    // cout << "about to sync for " << time_step << endl;
    dim3 dimBlock_x(CUDA_BLOCK_SIZE, CUDA_BLOCK_SIZE, 1);
    dim3 dimGrid_x(ceil((float)B2R_B_Z/CUDA_BLOCK_SIZE), ceil((float)B2R_B_Y/CUDA_BLOCK_SIZE), 1);
    // invoke cuda threads to copy corresponding block data to halo sync buffer
    sub_matrix_pack_x<<<dimGrid_x, dimBlock_x>>>(d_grid, d_pad_send_buf, b2r_halo_update[0]==1, b2r_halo_update[1]==1);   
    cudaErrchk( cudaDeviceSynchronize() );

    cudaErrchk( cudaMemcpy((void*) (h_pad_send_buf[0]), (void*) (h_dp_pad_send_buf[0]), buffer_size, cudaMemcpyDeviceToHost) );
    cudaErrchk( cudaMemcpy((void*) (h_pad_send_buf[1]), (void*) (h_dp_pad_send_buf[1]), buffer_size, cudaMemcpyDeviceToHost) );
    
    send_req_cnt = b2r_send_pad_x(time_step, h_pad_send_buf);               // send halos via non-block MPI 
    recv_req_cnt = b2r_recv_pad_x(time_step, h_pad_recv_buf);               // receive halos via non-block MPI 
    b2r_sync_wait(send_req_cnt, recv_req_cnt);
    
    cudaErrchk( cudaMemcpy((void*) (h_dp_pad_recv_buf[0]), (void*) (h_pad_recv_buf[0]), buffer_size, cudaMemcpyHostToDevice) );
    cudaErrchk( cudaMemcpy((void*) (h_dp_pad_recv_buf[1]), (void*) (h_pad_recv_buf[1]), buffer_size, cudaMemcpyHostToDevice) );
    // invoke cuda threads to copy received block data to halos
    sub_matrix_unpack_x<<<dimGrid_x, dimBlock_x>>>(d_grid, d_pad_recv_buf, b2r_halo_update[0]==1, b2r_halo_update[1]==1);
    cudaErrchk( cudaDeviceSynchronize() );
    
    // updating ghost zone in Y direction
    dim3 dimBlock_y(CUDA_BLOCK_SIZE, CUDA_BLOCK_SIZE, 1);
    dim3 dimGrid_y(ceil((float)B2R_BLOCK_SIZE_X/CUDA_BLOCK_SIZE), ceil((float)B2R_B_Y/CUDA_BLOCK_SIZE), 1);    
    sub_matrix_pack_y<<<dimGrid_y, dimBlock_y>>>(d_grid, d_pad_send_buf, b2r_halo_update[2]==1, b2r_halo_update[3]==1);   
    cudaErrchk( cudaDeviceSynchronize() );
    
    cudaErrchk( cudaMemcpy((void*) (h_pad_send_buf[0]), (void*) (h_dp_pad_send_buf[0]), buffer_size, cudaMemcpyDeviceToHost) );
    cudaErrchk( cudaMemcpy((void*) (h_pad_send_buf[1]), (void*) (h_dp_pad_send_buf[1]), buffer_size, cudaMemcpyDeviceToHost) );
        
    send_req_cnt = b2r_send_pad_y(time_step, h_pad_send_buf);               // send halos via non-block MPI 
    recv_req_cnt = b2r_recv_pad_y(time_step, h_pad_recv_buf);               // receive halos via non-block MPI 
    b2r_sync_wait(send_req_cnt, recv_req_cnt);
    
    cudaErrchk( cudaMemcpy((void*) (h_dp_pad_recv_buf[0]), (void*) (h_pad_recv_buf[0]), buffer_size, cudaMemcpyHostToDevice) );
    cudaErrchk( cudaMemcpy((void*) (h_dp_pad_recv_buf[1]), (void*) (h_pad_recv_buf[1]), buffer_size, cudaMemcpyHostToDevice) );    
    sub_matrix_unpack_y<<<dimGrid_y, dimBlock_y>>>(d_grid, d_pad_recv_buf, b2r_halo_update[2]==1, b2r_halo_update[3]==1);
    cudaErrchk( cudaDeviceSynchronize() );
    
    // updating ghost zone in Z direction
#if _ENV_3D_        
    dim3 dimBlock_z(CUDA_BLOCK_SIZE, CUDA_BLOCK_SIZE, 1);
    dim3 dimGrid_z(ceil((float)B2R_BLOCK_SIZE_X/CUDA_BLOCK_SIZE), ceil((float)B2R_BLOCK_SIZE_Y/CUDA_BLOCK_SIZE), 1);  
    
    sub_matrix_pack_z<<<dimGrid_z, dimBlock_z>>>(d_grid, d_pad_send_buf, b2r_halo_update[4]==1, b2r_halo_update[5]==1);   
    cudaErrchk( cudaDeviceSynchronize() );
    
    cudaErrchk( cudaMemcpy((void*) (h_pad_send_buf[0]), (void*) (h_dp_pad_send_buf[0]), buffer_size, cudaMemcpyDeviceToHost) );
    cudaErrchk( cudaMemcpy((void*) (h_pad_send_buf[1]), (void*) (h_dp_pad_send_buf[1]), buffer_size, cudaMemcpyDeviceToHost) );
        
    send_req_cnt = b2r_send_pad_z(time_step, h_pad_send_buf);               // send halos via non-block MPI 
    recv_req_cnt = b2r_recv_pad_z(time_step, h_pad_recv_buf);               // receive halos via non-block MPI 
    b2r_sync_wait(send_req_cnt, recv_req_cnt);
    
    cudaErrchk( cudaMemcpy((void*) (h_dp_pad_recv_buf[0]), (void*) (h_pad_recv_buf[0]), buffer_size, cudaMemcpyHostToDevice) );
    cudaErrchk( cudaMemcpy((void*) (h_dp_pad_recv_buf[1]), (void*) (h_pad_recv_buf[1]), buffer_size, cudaMemcpyHostToDevice) );   
    sub_matrix_unpack_z<<<dimGrid_z, dimBlock_z>>>(d_grid, d_pad_recv_buf, b2r_halo_update[4]==1, b2r_halo_update[5]==1);
    cudaErrchk( cudaDeviceSynchronize() );
#endif    
}

#endif 
