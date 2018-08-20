#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <iostream>
#include <fstream>
#include <string.h>
#include "b2r_config.h"

#define Dt            0.5
#define Dx            10.
#define Dy            10.
#define Dz            10.
#define INIT_Y        7
#define FRC_TIME      50
#define inv_rho       0.0005555555555555556 // (1./1800.)
#define E              84.293        // Young
#define Poi            0.31707       // poisson's ratio
#define lame1          32.00 // (E/(2 + 2*Poi))
#define lame2          174.87 //(E/((1+Poi) * (1-2*Poi)))
#define lame_122       (lame1 + 2*lame2)

#define coef_d2_a        (-1.f/12.f)
#define coef_d2_b        (2.f/3.f)

#define CUDA_BLOCK_SIZE    16

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
extern CELL_DT *d_vx, *d_vy, *d_vz;
extern CELL_DT *d_sigma_xx, *d_sigma_xy, *d_sigma_xz, *d_sigma_yy, *d_sigma_yz, *d_sigma_zz;

extern CELL_DT *h_vx, *h_vy, *h_vz;
extern CELL_DT *h_sigma_xx,  *h_sigma_xy, *h_sigma_xz, *h_sigma_yy, *h_sigma_yz, *h_sigma_zz;

CELL_DT *d_y;
float3 *d_init_force;

/*
***************************************************************************************************
* func   name: eqwp_fd4_vx
* description: update velocity in x direction with 4-order finite difference method
* parameters :
*             
* return: none
***************************************************************************************************
*/
__global__ void eqwp_fd4_vx(int Ngx, int Ngy, int Ngz, CELL_DT* vx, CELL_DT* sigma_xx, CELL_DT* sigma_xy, 
                            CELL_DT* sigma_xz, float3* f_init, int ts, int b2r_i, int b2r_R)
{
    // use shared memory because need differential to x or y
    __shared__ CELL_DT s_xx_cur[CUDA_BLOCK_SIZE+2*B2R_D][CUDA_BLOCK_SIZE+2*B2R_D]; 
    __shared__ CELL_DT s_xy_cur[CUDA_BLOCK_SIZE+2*B2R_D][CUDA_BLOCK_SIZE+2*B2R_D]; 
    
    CELL_DT s_xz_bh2, s_xz_bh1, s_xz_cur, s_xz_if1, s_xz_if2;
    
    int ix = blockIdx.x*blockDim.x + threadIdx.x;
    int iy = blockIdx.y*blockDim.y + threadIdx.y;
    if(ix >= Ngx-b2r_i*B2R_D || iy >= Ngy-b2r_i*B2R_D || ix < b2r_i*B2R_D || iy < b2r_i*B2R_D)
    {
        return;
    }

    int tx = threadIdx.x + B2R_D;   // physical id_x (due to halo storage)
    int ty = threadIdx.y + B2R_D;   // physical id_y (due to halo storage)

    int stride = Ngx*Ngy;
    // stride*b2r_i is important when R>1 which make sure the z direction is correct
    int in_2d  = stride*b2r_i + iy*Ngx + ix; 
    int out_2d;
    bool update_flag = ix >= B2R_D*(b2r_i+1) && ix <= Ngx-1-(1+b2r_i)*B2R_D && iy >= B2R_D*(1+b2r_i) && iy <= Ngy-1-(1+b2r_i)*B2R_D;

    s_xz_bh1 = sigma_xz[in_2d]; 
    in_2d += stride;

    s_xz_cur = sigma_xz[in_2d]; 
    out_2d = in_2d;
    in_2d += stride;

    s_xz_if1 = sigma_xz[in_2d]; 
    in_2d += stride;

    s_xz_if2 = sigma_xz[in_2d]; 
    in_2d += stride;

    for(int iz=B2R_D*(b2r_i+1); iz<Ngz-(1+b2r_i)*B2R_D; iz++)
    {
        // pipeline copy along z direction
        s_xz_bh2 = s_xz_bh1;     // behind2
        s_xz_bh1 = s_xz_cur;     // behind1
        s_xz_cur = s_xz_if1;     // current
        s_xz_if1 = s_xz_if2;     // infront1
        s_xz_if2 = sigma_xz[in_2d];      // infront2

        in_2d += stride;
        out_2d += stride;

        if (update_flag)
        {
            if(threadIdx.x < B2R_D){ // Halo left
                s_xx_cur[ty][tx-B2R_D] = sigma_xx[out_2d - B2R_D];
            }
            if(threadIdx.x >= CUDA_BLOCK_SIZE-B2R_D){ // Halo right
                s_xx_cur[ty][tx+B2R_D] = sigma_xx[out_2d + B2R_D];
            }
            if(threadIdx.y < B2R_D){ // Halo bottom
                s_xy_cur[ty-B2R_D][tx] = sigma_xy[out_2d - Ngx];
            }
            if(threadIdx.y >= CUDA_BLOCK_SIZE-B2R_D){ // Halo top
                s_xy_cur[ty+B2R_D][tx] = sigma_xy[out_2d + Ngx];
            }
        }
        
        s_xx_cur[ty][tx] = sigma_xx[out_2d];
        s_xy_cur[ty][tx] = sigma_xy[out_2d];
        __syncthreads();
        CELL_DT dsxx, dsxy, dsxz;
        if (update_flag){ 
            dsxx = (coef_d2_a * (s_xx_cur[ty][tx+2]-s_xx_cur[ty][tx-2]) + coef_d2_b * (s_xx_cur[ty][tx+1]-s_xx_cur[ty][tx-1])) / Dx;
            dsxy = (coef_d2_a * (s_xy_cur[ty+2][tx]-s_xy_cur[ty-2][tx]) + coef_d2_b * (s_xy_cur[ty+1][tx]-s_xy_cur[ty-1][tx])) / Dy;
            dsxz = (coef_d2_a * (s_xz_if2 -s_xz_bh2)   + coef_d2_b * (s_xz_if1 -s_xz_bh1))   / Dz;
            
            if(iy == (Ngy-INIT_Y-B2R_D*b2r_R) && ts < FRC_TIME){
                vx[out_2d] += Dt * inv_rho * (dsxx + dsxy + dsxz + f_init[iz*Ngx + ix].x);
            }else{
                vx[out_2d] += Dt * inv_rho * (dsxx + dsxy + dsxz);
            }               
        }
        __syncthreads();
    }    
}

/*
***************************************************************************************************
* func   name: eqwp_fd4_vy
* description: update velocity in y direction with 4-order finite difference method
* parameters :
*             
* return: none
***************************************************************************************************
*/
__global__ void eqwp_fd4_vy(int Ngx, int Ngy, int Ngz, CELL_DT* vy, CELL_DT* sigma_xy, CELL_DT* sigma_yy, 
                            CELL_DT* sigma_yz, float3* f_init, CELL_DT *disp, int ts, int b2r_i, int b2r_R)
{
    // use shared memory because need differential to x or y
    __shared__ CELL_DT s_xy_cur[CUDA_BLOCK_SIZE+2*B2R_D][CUDA_BLOCK_SIZE+2*B2R_D]; 
    __shared__ CELL_DT s_yy_cur[CUDA_BLOCK_SIZE+2*B2R_D][CUDA_BLOCK_SIZE+2*B2R_D]; 
    
    CELL_DT s_yz_bh2, s_yz_bh1, s_yz_cur, s_yz_if1, s_yz_if2;
    
    int ix = blockIdx.x*blockDim.x + threadIdx.x;
    int iy = blockIdx.y*blockDim.y + threadIdx.y;
    if(ix >= Ngx-b2r_i*B2R_D || iy >= Ngy-b2r_i*B2R_D || ix < b2r_i*B2R_D || iy < b2r_i*B2R_D)
    {
        return;
    }

    int tx = threadIdx.x + B2R_D;   // physical id_x (due to halo storage)
    int ty = threadIdx.y + B2R_D;   // physical id_y (due to halo storage)

    int stride = Ngx*Ngy;
    // stride*b2r_i is important when R>1 which make sure the z direction is correct
    int in_2d  = stride*b2r_i + iy*Ngx + ix; 
    int out_2d;
    bool update_flag = ix >= B2R_D*(b2r_i+1) && ix <= Ngx-1-(1+b2r_i)*B2R_D && iy >= B2R_D*(1+b2r_i) && iy <= Ngy-1-(1+b2r_i)*B2R_D;

    s_yz_bh1 = sigma_yz[in_2d]; 
    in_2d += stride;

    s_yz_cur = sigma_yz[in_2d]; 
    out_2d = in_2d;
    in_2d += stride;

    s_yz_if1 = sigma_yz[in_2d]; 
    in_2d += stride;

    s_yz_if2 = sigma_yz[in_2d]; 
    in_2d += stride;

    for(int iz=B2R_D*(b2r_i+1); iz<Ngz-(1+b2r_i)*B2R_D; iz++)
    {
        // pipeline copy along z direction
        s_yz_bh2 = s_yz_bh1;     // behind2
        s_yz_bh1 = s_yz_cur;     // behind1
        s_yz_cur = s_yz_if1;     // current
        s_yz_if1 = s_yz_if2;     // infront1
        s_yz_if2 = sigma_yz[in_2d];      // infront2

        in_2d += stride;
        out_2d += stride;

        if (update_flag)
        {
            if(threadIdx.x < B2R_D){ // Halo left
                s_xy_cur[ty][tx-B2R_D] = sigma_xy[out_2d - B2R_D];
            }
            if(threadIdx.x >= CUDA_BLOCK_SIZE-B2R_D){ // Halo right
                s_xy_cur[ty][tx+B2R_D] = sigma_xy[out_2d + B2R_D];
            }
            if(threadIdx.y < B2R_D){ // Halo bottom
                s_yy_cur[ty-B2R_D][tx] = sigma_yy[out_2d - Ngx];
            }
            if(threadIdx.y >= CUDA_BLOCK_SIZE-B2R_D){ // Halo top
                s_yy_cur[ty+B2R_D][tx] = sigma_yy[out_2d + Ngx];
            }
        }
        s_xy_cur[ty][tx] = sigma_xy[out_2d];
        s_yy_cur[ty][tx] = sigma_yy[out_2d];
        
        __syncthreads();
        CELL_DT dsxy, dsyy, dsyz;
        if (update_flag){ 
            dsxy = (coef_d2_a * (s_xy_cur[ty][tx+2]-s_xy_cur[ty][tx-2]) + coef_d2_b * (s_xy_cur[ty][tx+1]-s_xy_cur[ty][tx-1])) / Dx;
            dsyy = (coef_d2_a * (s_yy_cur[ty+2][tx]-s_yy_cur[ty-2][tx]) + coef_d2_b * (s_yy_cur[ty+1][tx]-s_yy_cur[ty-1][tx])) / Dy;
            dsyz = (coef_d2_a * (s_yz_if2 - s_yz_bh2)   + coef_d2_b * (s_yz_if1 - s_yz_bh1)) / Dz;
            
            if(iy == (Ngy-INIT_Y-B2R_D*b2r_R) && ts < FRC_TIME){
                vy[out_2d] += Dt * inv_rho * (dsxy + dsyy + dsyz + f_init[iz*Ngx + ix].y);
                disp[out_2d] += Dt * Dt * inv_rho * (dsxy + dsyy + dsyz + f_init[iz*Ngx + ix].y);
            }else{
                vy[out_2d] += Dt * inv_rho * (dsxy + dsyy + dsyz);
                disp[out_2d] += Dt * Dt * inv_rho * (dsxy + dsyy + dsyz);
            }            
        }
        __syncthreads();
    }    
}
/*
***************************************************************************************************
* func   name: eqwp_fd4_vz
* description: update velocity in z direction with 4-order finite difference method
* parameters :
*             
* return: none
***************************************************************************************************
*/
__global__ void eqwp_fd4_vz(int Ngx, int Ngy, int Ngz, CELL_DT* vz, CELL_DT* sigma_xz, CELL_DT* sigma_yz, 
                            CELL_DT* sigma_zz, float3* f_init, int ts, int b2r_i, int b2r_R)
{
    // use shared memory because need differential to x or y
    __shared__ CELL_DT s_xz_cur[CUDA_BLOCK_SIZE+2*B2R_D][CUDA_BLOCK_SIZE+2*B2R_D]; 
    __shared__ CELL_DT s_yz_cur[CUDA_BLOCK_SIZE+2*B2R_D][CUDA_BLOCK_SIZE+2*B2R_D]; 
    
    CELL_DT s_zz_bh2, s_zz_bh1, s_zz_cur, s_zz_if1, s_zz_if2;
    
    int ix = blockIdx.x*blockDim.x + threadIdx.x;
    int iy = blockIdx.y*blockDim.y + threadIdx.y;
    if(ix >= Ngx-b2r_i*B2R_D || iy >= Ngy-b2r_i*B2R_D || ix < b2r_i*B2R_D || iy < b2r_i*B2R_D)
    {
        return;
    }

    int tx = threadIdx.x + B2R_D;   // physical id_x (due to halo storage)
    int ty = threadIdx.y + B2R_D;   // physical id_y (due to halo storage)

    int stride = Ngx*Ngy;
    // stride*b2r_i is important when R>1 which make sure the z direction is correct
    int in_2d  = stride*b2r_i + iy*Ngx + ix; 
    int out_2d;
    bool update_flag = ix >= B2R_D*(b2r_i+1) && ix <= Ngx-1-(1+b2r_i)*B2R_D && iy >= B2R_D*(1+b2r_i) && iy <= Ngy-1-(1+b2r_i)*B2R_D;

    s_zz_bh1 = sigma_zz[in_2d]; 
    in_2d += stride;

    s_zz_cur = sigma_zz[in_2d]; 
    out_2d = in_2d;
    in_2d += stride;

    s_zz_if1 = sigma_zz[in_2d]; 
    in_2d += stride;

    s_zz_if2 = sigma_zz[in_2d]; 
    in_2d += stride;

    for(int iz=B2R_D*(b2r_i+1); iz<Ngz-(1+b2r_i)*B2R_D; iz++)
    {
        // pipeline copy along z direction
        s_zz_bh2 = s_zz_bh1;     // behind2
        s_zz_bh1 = s_zz_cur;     // behind1
        s_zz_cur = s_zz_if1;     // current
        s_zz_if1 = s_zz_if2;     // infront1
        s_zz_if2 = sigma_zz[in_2d];      // infront2

        in_2d += stride;
        out_2d += stride;

        if (update_flag)
        {
            if(threadIdx.x < B2R_D){ // Halo left
                s_xz_cur[ty][tx-B2R_D] = sigma_xz[out_2d - B2R_D];
            }
            if(threadIdx.x >= CUDA_BLOCK_SIZE-B2R_D){ // Halo right
                s_xz_cur[ty][tx+B2R_D] = sigma_xz[out_2d + B2R_D];
            }
            if(threadIdx.y < B2R_D){ // Halo bottom
                s_yz_cur[ty-B2R_D][tx] = sigma_yz[out_2d - Ngx];
            }
            if(threadIdx.y >= CUDA_BLOCK_SIZE-B2R_D){ // Halo top
                s_yz_cur[ty+B2R_D][tx] = sigma_yz[out_2d + Ngx];
            }
        }
        s_xz_cur[ty][tx] = sigma_xz[out_2d];
        s_yz_cur[ty][tx] = sigma_yz[out_2d];
        
        __syncthreads();
        CELL_DT dsxz, dsyz, dszz;
        if (update_flag){ 
            dsxz = (coef_d2_a * (s_xz_cur[ty][tx+2]-s_xz_cur[ty][tx-2]) + coef_d2_b * (s_xz_cur[ty][tx+1]-s_xz_cur[ty][tx-1])) / Dx;
            dsyz = (coef_d2_a * (s_yz_cur[ty+2][tx]-s_yz_cur[ty-2][tx]) + coef_d2_b * (s_yz_cur[ty+1][tx]-s_yz_cur[ty-1][tx])) / Dy;
            dszz = (coef_d2_a * (s_zz_if2  - s_zz_bh2)   + coef_d2_b * (s_zz_if1  - s_zz_bh1)) / Dz;
            if(iy == (Ngy-INIT_Y-B2R_D*b2r_R) && ts < FRC_TIME){
                vz[out_2d] += Dt * inv_rho * (dsxz + dsyz + dszz + f_init[iz*Ngx + ix].z);
            }else{
                vz[out_2d] += Dt * inv_rho * (dsxz + dsyz + dszz);
            }
        }
        __syncthreads();
    }    
}
/*
***************************************************************************************************
* func   name: eqwp_fd4_stress
* description: update all the stree tensor, with 4-order finite difference method
* parameters :
*             
* return: none
***************************************************************************************************
*/
__global__ void eqwp_fd4_stress(int Ngx, int Ngy, int Ngz, CELL_DT* sigma_xx, CELL_DT* sigma_xy, CELL_DT* sigma_xz, CELL_DT* sigma_yy, 
                             CELL_DT* sigma_yz, CELL_DT* sigma_zz, CELL_DT* vx, CELL_DT* vy, CELL_DT* vz, int b2r_i)
{    
    __shared__ CELL_DT vx_bh2[CUDA_BLOCK_SIZE+2*B2R_D][CUDA_BLOCK_SIZE+2*B2R_D]; 
    __shared__ CELL_DT vx_bh1[CUDA_BLOCK_SIZE+2*B2R_D][CUDA_BLOCK_SIZE+2*B2R_D]; 
    __shared__ CELL_DT vx_cur[CUDA_BLOCK_SIZE+2*B2R_D][CUDA_BLOCK_SIZE+2*B2R_D]; 
    __shared__ CELL_DT vx_if1[CUDA_BLOCK_SIZE+2*B2R_D][CUDA_BLOCK_SIZE+2*B2R_D]; 
    __shared__ CELL_DT vx_if2[CUDA_BLOCK_SIZE+2*B2R_D][CUDA_BLOCK_SIZE+2*B2R_D]; 

    __shared__ CELL_DT vy_bh2[CUDA_BLOCK_SIZE+2*B2R_D][CUDA_BLOCK_SIZE+2*B2R_D]; 
    __shared__ CELL_DT vy_bh1[CUDA_BLOCK_SIZE+2*B2R_D][CUDA_BLOCK_SIZE+2*B2R_D]; 
    __shared__ CELL_DT vy_cur[CUDA_BLOCK_SIZE+2*B2R_D][CUDA_BLOCK_SIZE+2*B2R_D]; 
    __shared__ CELL_DT vy_if1[CUDA_BLOCK_SIZE+2*B2R_D][CUDA_BLOCK_SIZE+2*B2R_D]; 
    __shared__ CELL_DT vy_if2[CUDA_BLOCK_SIZE+2*B2R_D][CUDA_BLOCK_SIZE+2*B2R_D]; 
    
    __shared__ CELL_DT vz_bh2[CUDA_BLOCK_SIZE+2*B2R_D][CUDA_BLOCK_SIZE+2*B2R_D]; 
    __shared__ CELL_DT vz_bh1[CUDA_BLOCK_SIZE+2*B2R_D][CUDA_BLOCK_SIZE+2*B2R_D]; 
    __shared__ CELL_DT vz_cur[CUDA_BLOCK_SIZE+2*B2R_D][CUDA_BLOCK_SIZE+2*B2R_D]; 
    __shared__ CELL_DT vz_if1[CUDA_BLOCK_SIZE+2*B2R_D][CUDA_BLOCK_SIZE+2*B2R_D]; 
    __shared__ CELL_DT vz_if2[CUDA_BLOCK_SIZE+2*B2R_D][CUDA_BLOCK_SIZE+2*B2R_D]; 
        
    int ix = blockIdx.x*blockDim.x + threadIdx.x;
    int iy = blockIdx.y*blockDim.y + threadIdx.y;
    if(ix >= Ngx-b2r_i*B2R_D || iy >= Ngy-b2r_i*B2R_D || ix < b2r_i*B2R_D || iy < b2r_i*B2R_D)
    {
        return;
    }

    int tx = threadIdx.x + B2R_D;   // physical id_x (due to halo storage)
    int ty = threadIdx.y + B2R_D;   // physical id_y (due to halo storage)

    int stride = Ngx*Ngy;
    // stride*b2r_i is important when R>1 which make sure the z direction is correct
    int in_2d  = stride*b2r_i + iy*Ngx + ix; 
    int out_2d;
    bool update_flag = ix >= B2R_D*(b2r_i+1) && ix <= Ngx-1-(1+b2r_i)*B2R_D && iy >= B2R_D*(1+b2r_i) && iy <= Ngy-1-(1+b2r_i)*B2R_D;

    vx_bh1[ty][tx] = vx[in_2d]; 
    vy_bh1[ty][tx] = vy[in_2d]; 
    vz_bh1[ty][tx] = vz[in_2d]; 
    in_2d += stride;

    vx_cur[ty][tx] = vx[in_2d];
    vy_cur[ty][tx] = vy[in_2d];
    vz_cur[ty][tx] = vz[in_2d]; 
    out_2d = in_2d;
    in_2d += stride;

    vx_if1[ty][tx] = vx[in_2d];
    vy_if1[ty][tx] = vy[in_2d];
    vz_if1[ty][tx] = vz[in_2d]; 
    in_2d += stride;

    vx_if2[ty][tx] = vx[in_2d]; 
    vy_if2[ty][tx] = vy[in_2d]; 
    vz_if2[ty][tx] = vz[in_2d]; 
    in_2d += stride;

    for(int i=B2R_D*(b2r_i+1); i<Ngz-(1+b2r_i)*B2R_D; i++)
    {
        // pipeline copy along z direction
        vx_bh2[ty][tx] = vx_bh1[ty][tx];     // behind2
        vx_bh1[ty][tx] = vx_cur[ty][tx];     // behind1
        vx_cur[ty][tx] = vx_if1[ty][tx];     // current
        vx_if1[ty][tx] = vx_if2[ty][tx];     // infront1
        vx_if2[ty][tx] = vx[in_2d];          // infront2

        vy_bh2[ty][tx] = vy_bh1[ty][tx];     // behind2
        vy_bh1[ty][tx] = vy_cur[ty][tx];     // behind1
        vy_cur[ty][tx] = vy_if1[ty][tx];     // current
        vy_if1[ty][tx] = vy_if2[ty][tx];     // infront1
        vy_if2[ty][tx] = vy[in_2d];          // infront2
        
        vz_bh2[ty][tx] = vz_bh1[ty][tx];     // behind2
        vz_bh1[ty][tx] = vz_cur[ty][tx];     // behind1
        vz_cur[ty][tx] = vz_if1[ty][tx];     // current
        vz_if1[ty][tx] = vz_if2[ty][tx];     // infront1
        vz_if2[ty][tx] = vz[in_2d];          // infront2
        
        in_2d += stride;
        out_2d += stride;

        if (update_flag)
        {
            if(threadIdx.x < B2R_D){ // Halo left
                vx_cur[ty][tx-B2R_D] = vx[out_2d - B2R_D];
                vy_cur[ty][tx-B2R_D] = vy[out_2d - B2R_D];
                vz_cur[ty][tx-B2R_D] = vz[out_2d - B2R_D];
            }
            if(threadIdx.x >= CUDA_BLOCK_SIZE-B2R_D){ // Halo right
                vx_cur[ty][tx+B2R_D] = vx[out_2d + B2R_D];
                vy_cur[ty][tx+B2R_D] = vy[out_2d + B2R_D];
                vz_cur[ty][tx+B2R_D] = vz[out_2d + B2R_D];
            }
            if(threadIdx.y < B2R_D){ // Halo bottom
                vx_cur[ty-B2R_D][tx] = vx[out_2d - Ngx];
                vy_cur[ty-B2R_D][tx] = vy[out_2d - Ngx];
                vz_cur[ty-B2R_D][tx] = vz[out_2d - Ngx];
            }
            if(threadIdx.y >= CUDA_BLOCK_SIZE-B2R_D){ // Halo top
                vx_cur[ty+B2R_D][tx] = vx[out_2d + Ngx];
                vy_cur[ty+B2R_D][tx] = vy[out_2d + Ngx];
                vz_cur[ty+B2R_D][tx] = vz[out_2d + Ngx];
            }
        }
        __syncthreads();
        CELL_DT dsxx, dsxy, dsxz, dsyy, dsyz, dszz;
        CELL_DT dvxx, dvyy, dvzz, dvxy, dvyx, dvxz, dvzx, dvyz, dvzy;
        if (update_flag){ 
            dvxx = (coef_d2_a * (vx_cur[ty][tx+2]-vx_cur[ty][tx-2]) + coef_d2_b * (vx_cur[ty][tx+1]-vx_cur[ty][tx-1])) / Dx;
            dvyy = (coef_d2_a * (vy_cur[ty+2][tx]-vy_cur[ty-2][tx]) + coef_d2_b * (vy_cur[ty+1][tx]-vy_cur[ty-1][tx])) / Dy;
            dvzz = (coef_d2_a * (vz_if2[ty][tx]  -vz_bh2[ty][tx])   + coef_d2_b * (vz_if1[ty][tx]  -vz_bh1[ty][tx]))   / Dz;
            dvxy = (coef_d2_a * (vx_cur[ty+2][tx]-vx_cur[ty-2][tx]) + coef_d2_b * (vx_cur[ty+1][tx]-vx_cur[ty-1][tx])) / Dy;
            dvyx = (coef_d2_a * (vy_cur[ty][tx+2]-vy_cur[ty][tx-2]) + coef_d2_b * (vy_cur[ty][tx+1]-vy_cur[ty][tx-1])) / Dx;
            dvxz = (coef_d2_a * (vx_if2[ty][tx]  -vx_bh2[ty][tx])   + coef_d2_b * (vx_if1[ty][tx]  -vx_bh1[ty][tx]))   / Dz;
            dvzx = (coef_d2_a * (vz_cur[ty][tx+2]-vz_cur[ty][tx-2]) + coef_d2_b * (vz_cur[ty][tx+1]-vz_cur[ty][tx-1])) / Dx;
            dvyz = (coef_d2_a * (vy_if2[ty][tx]  -vy_bh2[ty][tx])   + coef_d2_b * (vy_if1[ty][tx]  -vy_bh1[ty][tx]))   / Dz;
            dvzy = (coef_d2_a * (vz_cur[ty+2][tx]-vz_cur[ty-2][tx]) + coef_d2_b * (vz_cur[ty+1][tx]-vz_cur[ty-1][tx])) / Dy;
            
            dsxx = (lame1 + 2*lame2) * dvxx + lame1 * (dvyy + dvzz);
            dsyy = (lame1 + 2*lame2) * dvyy + lame1 * (dvxx + dvzz);
            dszz = (lame1 + 2*lame2) * dvzz + lame1 * (dvxx + dvyy);
            
            dsxy = lame2 * (dvxy + dvyx);
            dsxz = lame2 * (dvxz + dvzx);
            dsyz = lame2 * (dvyz + dvzy);
            
            sigma_xx[out_2d] += Dt * dsxx;
            sigma_xy[out_2d] += Dt * dsxy;
            sigma_xz[out_2d] += Dt * dsxz;
            sigma_yy[out_2d] += Dt * dsyy;
            sigma_yz[out_2d] += Dt * dsyz;
            sigma_zz[out_2d] += Dt * dszz;
        }
        __syncthreads();
    }       
}

/*
***************************************************************************************************
* func   name: pinned_alloc
* description: allocate pinned memory
* parameters :
*             
* return: none
***************************************************************************************************
*/
CELL_DT* pinned_alloc(int bytes){
    CELL_DT* res;
    cudaErrchk( cudaMallocHost((void**)&res, bytes) );
    return res;
}

/*
***************************************************************************************************
* func   name: free_pinned_mem
* description: free allocate pinned memory
* parameters :
*             
* return: none
***************************************************************************************************
*/
void free_pinned_mem(CELL_DT *p){
    cudaFreeHost(p);
}
/*
***************************************************************************************************
* func   name: eqwp_cuda_init
* description: initialize cuda related things
* parameters :
*             
* return: none
***************************************************************************************************
*/
void eqwp_cuda_init(int Ngx, int Ngy, int Ngz)
{
    // allocate device memory
    int grid_size = Ngx * Ngy * Ngz * sizeof(CELL_DT);
    cudaErrchk( cudaMalloc((void**)&d_vx, grid_size) );
    cudaErrchk( cudaMalloc((void**)&d_vy, grid_size) );
    cudaErrchk( cudaMalloc((void**)&d_vz, grid_size) );
    
    cudaErrchk( cudaMalloc((void**)&d_sigma_xx, grid_size) );
    cudaErrchk( cudaMalloc((void**)&d_sigma_xy, grid_size) );
    cudaErrchk( cudaMalloc((void**)&d_sigma_xz, grid_size) );
    cudaErrchk( cudaMalloc((void**)&d_sigma_yy, grid_size) );
    cudaErrchk( cudaMalloc((void**)&d_sigma_yz, grid_size) );
    cudaErrchk( cudaMalloc((void**)&d_sigma_zz, grid_size) );
    // allocate pinned host memory
    cudaErrchk( cudaMallocHost((void**)&h_vx, grid_size) );
    memset((void*)h_vx, 0, grid_size);
    
    cudaErrchk( cudaMallocHost((void**)&h_vy, grid_size) );
    memset((void*)h_vy, 0, grid_size);
    
    cudaErrchk( cudaMallocHost((void**)&h_vz, grid_size) );
    memset((void*)h_vz, 0, grid_size);
    
    cudaErrchk( cudaMallocHost((void**)&h_sigma_xx, grid_size) );
    memset((void*)h_sigma_xx, 0, grid_size);
    
    cudaErrchk( cudaMallocHost((void**)&h_sigma_xy, grid_size) );
    memset((void*)h_sigma_xy, 0, grid_size);
    
    cudaErrchk( cudaMallocHost((void**)&h_sigma_xz, grid_size) );
    memset((void*)h_sigma_xz, 0, grid_size);
    
    cudaErrchk( cudaMallocHost((void**)&h_sigma_yy, grid_size) );
    memset((void*)h_sigma_yy, 0, grid_size);
    
    cudaErrchk( cudaMallocHost((void**)&h_sigma_yz, grid_size) );
    memset((void*)h_sigma_yz, 0, grid_size);
    
    cudaErrchk( cudaMallocHost((void**)&h_sigma_zz, grid_size) );
    memset((void*)h_sigma_zz, 0, grid_size);
    // for displacement
    cudaErrchk( cudaMalloc((void**)&d_y, grid_size) );
    cudaErrchk( cudaMemset((void*)d_y, 0, grid_size) );
    
    // cpy inital force
    float3 *init_f = new float3[Ngx*Ngz]();
    memset((void*)init_f, 0, sizeof(float3)*Ngx*Ngz);
    for(int x=Ngx/2; x<Ngx/2+Ngx/10; x++)
        for(int z=Ngz/2; z<Ngz/2+Ngz/10; z++){
            init_f[z*Ngx+x].y = -10000.;
        }
    cudaErrchk( cudaMalloc((void**)&d_init_force, sizeof(float3)*Ngx*Ngz) );
    cudaErrchk( cudaMemcpy((void *)d_init_force, (void *)init_f, sizeof(float3)*Ngx*Ngz, cudaMemcpyHostToDevice) );
    
    // copy inital data to device if necessary
    // copy (ghost zone)updated velocity state
    cudaErrchk( cudaMemcpy((void *)d_vx, (void *)h_vx, grid_size, cudaMemcpyHostToDevice) );
    cudaErrchk( cudaMemcpy((void *)d_vy, (void *)h_vy, grid_size, cudaMemcpyHostToDevice) );
    cudaErrchk( cudaMemcpy((void *)d_vz, (void *)h_vz, grid_size, cudaMemcpyHostToDevice) );
    // copy (ghost zone)updated stress state
    cudaErrchk( cudaMemcpy((void *)d_sigma_xx, (void *)h_sigma_xx, grid_size, cudaMemcpyHostToDevice) );
    cudaErrchk( cudaMemcpy((void *)d_sigma_xy, (void *)h_sigma_xy, grid_size, cudaMemcpyHostToDevice) );
    cudaErrchk( cudaMemcpy((void *)d_sigma_xz, (void *)h_sigma_xz, grid_size, cudaMemcpyHostToDevice) );
    cudaErrchk( cudaMemcpy((void *)d_sigma_yy, (void *)h_sigma_yy, grid_size, cudaMemcpyHostToDevice) );
    cudaErrchk( cudaMemcpy((void *)d_sigma_yz, (void *)h_sigma_yz, grid_size, cudaMemcpyHostToDevice) );
    cudaErrchk( cudaMemcpy((void *)d_sigma_zz, (void *)h_sigma_zz, grid_size, cudaMemcpyHostToDevice) );    
}
/*
***************************************************************************************************
* func   name: eqwp_cuda_finalize
* description: finalize cuda related things
* parameters :
*             
* return: none
***************************************************************************************************
*/
void eqwp_cuda_finalize()
{
    cudaFree(d_vx);
    cudaFree(d_vy);
    cudaFree(d_vz);
    cudaFree(d_sigma_xx);
    cudaFree(d_sigma_xy);
    cudaFree(d_sigma_xz);
    cudaFree(d_sigma_yy);
    cudaFree(d_sigma_yz);
    cudaFree(d_sigma_zz);
    cudaFree(d_y);
    cudaFree(d_init_force);
}

/*
***************************************************************************************************
* func   name: write_block_data_bin
* description: write results to file for visualizing
* parameters :
*             none
* return: none
***************************************************************************************************
*/
void write_block_data_bin(int time_step, CELL_DT * p_data, int Ngx, int Ngy, int Ngz)
{
    ofstream output_file;
    char filename[100];
    sprintf( filename, "displacement-%d.bin", time_step);
    output_file.open(filename, iostream::binary);
    int f_header[3] = {Ngx, Ngy, Ngz};
    output_file.write((char*)f_header, sizeof(f_header)); 
    output_file.write((char*)p_data, sizeof(CELL_DT) * Ngx * Ngy * Ngz);  
    output_file.close();
}
/*
***************************************************************************************************
* func   name: write_block_data
* description: write results to file for visualizing
* parameters :
*             none
* return: none
***************************************************************************************************
*/
void write_block_data(int time_step, CELL_DT * p_data, int Ngx, int Ngy, int Ngz)
{
    ofstream output_file;
    char filename[100];
    sprintf( filename, "displacement-%d.txt", time_step);
    output_file.open(filename, iostream::binary);
    for(int z = 0; z < Ngz; z++)
        for(int y = 0; y < Ngy; y++){
            for(int x = 0; x < Ngx; x++){
                int idx = z*Ngx*Ngy + y*Ngx + x;
                output_file << p_data[idx] << ",";
            }
            output_file << endl;  
        }  
    output_file.close();
}
/*
***************************************************************************************************
* func   name: eqwp_gpu_main
* description: main entrance for device computing
* parameters :
*             
* return: none
***************************************************************************************************
*/
double eqwp_gpu_main(int ts, int b2r_R, int Ngx, int Ngy, int Ngz, bool op_flag)
{
    CELL_DT *h_y;    
    // Launch configuration:
    cudaFuncSetCacheConfig(eqwp_fd4_stress, cudaFuncCachePreferShared);
    cudaFuncSetCacheConfig(eqwp_fd4_vx, cudaFuncCachePreferShared);
    cudaFuncSetCacheConfig(eqwp_fd4_vy, cudaFuncCachePreferShared);
    cudaFuncSetCacheConfig(eqwp_fd4_vz, cudaFuncCachePreferShared);

    // count computation time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);    
    dim3 dimBlock(CUDA_BLOCK_SIZE, CUDA_BLOCK_SIZE, 1);
    dim3 dimGrid(ceil((float)Ngx/CUDA_BLOCK_SIZE), ceil((float)Ngy/CUDA_BLOCK_SIZE), 1);
    for (int i = 0; i < b2r_R; i++)
    {
        eqwp_fd4_stress<<<dimGrid, dimBlock>>>(Ngx, Ngy, Ngz, d_sigma_xx, d_sigma_xy, d_sigma_xz, 
                                               d_sigma_yy, d_sigma_yz, d_sigma_zz, 
                                               d_vx, d_vy, d_vz, i);
        eqwp_fd4_vx<<<dimGrid, dimBlock>>>(Ngx, Ngy, Ngz, d_vx, d_sigma_xx, d_sigma_xy, d_sigma_xz, d_init_force, ts, i, b2r_R);
        eqwp_fd4_vy<<<dimGrid, dimBlock>>>(Ngx, Ngy, Ngz, d_vy, d_sigma_xy, d_sigma_yy, d_sigma_yz, d_init_force, d_y, ts, i, b2r_R);
        eqwp_fd4_vz<<<dimGrid, dimBlock>>>(Ngx, Ngy, Ngz, d_vz, d_sigma_xz, d_sigma_yz, d_sigma_zz, d_init_force, ts, i, b2r_R);      
        cudaErrchk( cudaDeviceSynchronize() );          
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
       
    // copy results for synchronizing, only copy data in valid z range
    int ghs_sh = b2r_R*B2R_D*Ngx*Ngy;                       // address shift for the first rd layers
    int cp4s_size = Ngx*Ngy*(Ngz-2*b2r_R*B2R_D) * sizeof(CELL_DT);  // bytes to copy
    if(false && op_flag){    
        cudaErrchk( cudaMemcpy((void*) (h_vx + ghs_sh), (void*) (d_vx + ghs_sh), cp4s_size, cudaMemcpyDeviceToHost) );
        cudaErrchk( cudaMemcpy((void*) (h_vy + ghs_sh), (void*) (d_vy + ghs_sh), cp4s_size, cudaMemcpyDeviceToHost) );
        cudaErrchk( cudaMemcpy((void*) (h_vz + ghs_sh), (void*) (d_vz + ghs_sh), cp4s_size, cudaMemcpyDeviceToHost) );    
        cudaErrchk( cudaMemcpy((void*) (h_sigma_xx + ghs_sh), (void*) (d_sigma_xx + ghs_sh), cp4s_size, cudaMemcpyDeviceToHost) );
        cudaErrchk( cudaMemcpy((void*) (h_sigma_xy + ghs_sh), (void*) (d_sigma_xy + ghs_sh), cp4s_size, cudaMemcpyDeviceToHost) );
        cudaErrchk( cudaMemcpy((void*) (h_sigma_xz + ghs_sh), (void*) (d_sigma_xz + ghs_sh), cp4s_size, cudaMemcpyDeviceToHost) );
        cudaErrchk( cudaMemcpy((void*) (h_sigma_yy + ghs_sh), (void*) (d_sigma_yy + ghs_sh), cp4s_size, cudaMemcpyDeviceToHost) );
        cudaErrchk( cudaMemcpy((void*) (h_sigma_yz + ghs_sh), (void*) (d_sigma_yz + ghs_sh), cp4s_size, cudaMemcpyDeviceToHost) );
        cudaErrchk( cudaMemcpy((void*) (h_sigma_zz + ghs_sh), (void*) (d_sigma_zz + ghs_sh), cp4s_size, cudaMemcpyDeviceToHost) );

        cudaErrchk( cudaMallocHost((void**)&h_y, sizeof(CELL_DT)*Ngx*Ngy*Ngz) );
        cudaErrchk( cudaMemcpy((void*)h_y, (void*)d_y, sizeof(CELL_DT)*Ngx*Ngy*Ngz, cudaMemcpyDeviceToHost) );   
        //write_block_data(ts, h_y, Ngx, Ngy, Ngz);  
        write_block_data_bin(ts, h_y, Ngx, Ngy, Ngz); 
        cudaFreeHost(h_y);
        eqwp_cuda_finalize();
    }
    float gpu_compu_elapsed_time_ms;
    cudaEventElapsedTime(&gpu_compu_elapsed_time_ms, start, stop);
    return (double)gpu_compu_elapsed_time_ms / 1000.0;
}
