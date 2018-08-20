//
//  b2r_comm.hpp
//  b2r
//
//  Created by Zhengchun Liu on 1/15/16.
//  Copyright © 2016 research. All rights reserved.
//
#ifndef b2r_comm_h
#define b2r_comm_h

#include <iostream>
#include "b2r.h"

using namespace std;

/*
 ***************************************************************************************************
 * func   name: b2r_3D_get_rank
 * description: return the global rank/id via Oxyz coordinates
 * parameters :
 *             x, y, z the 3D coordinate
 * return: the global id / process rank in mpi
 ***************************************************************************************************
 */
int b2r_3D_get_rank(int x, int y, int z)
{
    return z * BLOCK_DIM_X * BLOCK_DIM_Y + y * BLOCK_DIM_X + x;
}

/*
 ***************************************************************************************************
 * func   name: b2r_recv_pad_x
 * description: when finished local updating (for R time steps), have to wait the new data from
                neighbourhoods, this function handle the nonblocking receiving in X direction.
 * parameters :
 *             tstep: current simulation time step
 * return: none
 ***************************************************************************************************
 */
int b2r_recv_pad_x(int tstep, char **pad_recv_buf)
{
    int rank, req_cnt = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int block_id_x = (rank % (BLOCK_DIM_X * BLOCK_DIM_Y)) % BLOCK_DIM_X;
    int block_id_y = (rank % (BLOCK_DIM_X * BLOCK_DIM_Y)) / BLOCK_DIM_X;
    int block_id_z = (rank / (BLOCK_DIM_X * BLOCK_DIM_Y));
    int Rd = B2R_R * B2R_D;
       
    // left 
    if(b2r_halo_update[0] == 1)
    {
        MPI_Irecv(pad_recv_buf[0], B2R_B_Y*Rd*B2R_B_Z*item_size, MPI_BYTE,
                  b2r_3D_get_rank(block_id_x-1, block_id_y, block_id_z),
                  tstep + 0, MPI_COMM_WORLD, &b2r_req_recv[req_cnt++]);
    }    
    // right 
    if(b2r_halo_update[1] == 1)
    {
        MPI_Irecv(pad_recv_buf[1], B2R_B_Y*Rd*B2R_B_Z*item_size, MPI_BYTE,
                  b2r_3D_get_rank(block_id_x+1, block_id_y, block_id_z),
                  tstep + 1, MPI_COMM_WORLD, &b2r_req_recv[req_cnt++]);
    }
    return req_cnt;
}

/*
 ***************************************************************************************************
 * func   name: b2r_send_pad_x
 * description: when finished local updating (for R time steps), needs to pack the new block, 
                sync to neighbourhoods, this function handle the nonblocking sending in X direction.
 * parameters :
 *             tstep: current simulation time step
 * return: none
 ***************************************************************************************************
 */
int b2r_send_pad_x(int tstep, char **pad_send_buf)
{
    int rank, req_cnt = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int block_id_x = (rank % (BLOCK_DIM_X * BLOCK_DIM_Y)) % BLOCK_DIM_X;
    int block_id_y = (rank % (BLOCK_DIM_X * BLOCK_DIM_Y)) / BLOCK_DIM_X;
    int block_id_z = (rank / (BLOCK_DIM_X * BLOCK_DIM_Y));
    int Rd = B2R_R * B2R_D;
       
    // left 
    if(b2r_halo_update[0] == 1)
    {
        MPI_Isend(pad_send_buf[0], B2R_B_Y*Rd*B2R_B_Z*item_size, MPI_BYTE,
                  b2r_3D_get_rank(block_id_x-1, block_id_y, block_id_z),
                  tstep + 1, MPI_COMM_WORLD, &b2r_req_send[req_cnt++]);
    }    
    // right 
    if(b2r_halo_update[1] == 1)
    {
        MPI_Isend(pad_send_buf[1], B2R_B_Y*Rd*B2R_B_Z*item_size, MPI_BYTE,
                  b2r_3D_get_rank(block_id_x+1, block_id_y, block_id_z),
                  tstep + 0, MPI_COMM_WORLD, &b2r_req_send[req_cnt++]);
    }
    return req_cnt;
}
/*
 ***************************************************************************************************
 * func   name: b2r_recv_pad_y
 * description: when finished local updating (for R time steps), have to wait the new data from
                neighbourhoods, this function handle the nonblocking receiving in Y direction.
 * parameters :
 *             tstep: current simulation time step
 * return: none
 ***************************************************************************************************
 */
int b2r_recv_pad_y(int tstep, char **pad_recv_buf)
{
    int rank, req_cnt = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int block_id_x = (rank % (BLOCK_DIM_X * BLOCK_DIM_Y)) % BLOCK_DIM_X;
    int block_id_y = (rank % (BLOCK_DIM_X * BLOCK_DIM_Y)) / BLOCK_DIM_X;
    int block_id_z = (rank / (BLOCK_DIM_X * BLOCK_DIM_Y));
    int Rd = B2R_R * B2R_D;
       
    // left 
    if(b2r_halo_update[2] == 1)
    {
        MPI_Irecv(pad_recv_buf[0], B2R_BLOCK_SIZE_X*Rd*B2R_B_Z*item_size, MPI_BYTE,
                  b2r_3D_get_rank(block_id_x, block_id_y-1, block_id_z),
                  tstep + 0, MPI_COMM_WORLD, &b2r_req_recv[req_cnt++]);
    }    
    // right 
    if(b2r_halo_update[3] == 1)
    {
        MPI_Irecv(pad_recv_buf[1], B2R_BLOCK_SIZE_X*Rd*B2R_B_Z*item_size, MPI_BYTE,
                  b2r_3D_get_rank(block_id_x, block_id_y+1, block_id_z),
                  tstep + 1, MPI_COMM_WORLD, &b2r_req_recv[req_cnt++]);
    }
    return req_cnt;
}

/*
 ***************************************************************************************************
 * func   name: b2r_send_pad_y
 * description: when finished local updating (for R time steps), needs to pack the new block, 
                sync to neighbourhoods, this function handle the nonblocking sending in Y direction.
 * parameters :
 *             tstep: current simulation time step
 * return: none
 ***************************************************************************************************
 */
int b2r_send_pad_y(int tstep, char **pad_send_buf)
{
    int rank, req_cnt = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int block_id_x = (rank % (BLOCK_DIM_X * BLOCK_DIM_Y)) % BLOCK_DIM_X;
    int block_id_y = (rank % (BLOCK_DIM_X * BLOCK_DIM_Y)) / BLOCK_DIM_X;
    int block_id_z = (rank / (BLOCK_DIM_X * BLOCK_DIM_Y));
    int Rd = B2R_R * B2R_D;
       
    // left 
    if(b2r_halo_update[2] == 1)
    {
        MPI_Isend(pad_send_buf[0], B2R_BLOCK_SIZE_X*Rd*B2R_B_Z*item_size, MPI_BYTE,
                  b2r_3D_get_rank(block_id_x, block_id_y-1, block_id_z),
                  tstep + 1, MPI_COMM_WORLD, &b2r_req_send[req_cnt++]);
    }    
    // right 
    if(b2r_halo_update[3] == 1)
    {
        MPI_Isend(pad_send_buf[1], B2R_BLOCK_SIZE_X*Rd*B2R_B_Z*item_size, MPI_BYTE,
                  b2r_3D_get_rank(block_id_x, block_id_y+1, block_id_z),
                  tstep + 0, MPI_COMM_WORLD, &b2r_req_send[req_cnt++]);
    }
    return req_cnt;
}
/*
 ***************************************************************************************************
 * func   name: b2r_recv_pad_z
 * description: when finished local updating (for R time steps), have to wait the new data from
                neighbourhoods, this function handle the nonblocking receiving in Z direction.
 * parameters :
 *             tstep: current simulation time step
 * return: none
 ***************************************************************************************************
 */
int b2r_recv_pad_z(int tstep, char **pad_recv_buf)
{
    int rank, req_cnt = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int block_id_x = (rank % (BLOCK_DIM_X * BLOCK_DIM_Y)) % BLOCK_DIM_X;
    int block_id_y = (rank % (BLOCK_DIM_X * BLOCK_DIM_Y)) / BLOCK_DIM_X;
    int block_id_z = (rank / (BLOCK_DIM_X * BLOCK_DIM_Y));
    int Rd = B2R_R * B2R_D;
       
    // left 
    if(b2r_halo_update[4] == 1)
    {
        MPI_Irecv(pad_recv_buf[0], B2R_BLOCK_SIZE_X*B2R_BLOCK_SIZE_Y*Rd*item_size, MPI_BYTE,
                  b2r_3D_get_rank(block_id_x, block_id_y, block_id_z-1),
                  tstep + 0, MPI_COMM_WORLD, &b2r_req_recv[req_cnt++]);
    }    
    // right 
    if(b2r_halo_update[5] == 1)
    {
        MPI_Irecv(pad_recv_buf[1], B2R_BLOCK_SIZE_X*B2R_BLOCK_SIZE_Y*Rd*item_size, MPI_BYTE,
                  b2r_3D_get_rank(block_id_x, block_id_y, block_id_z+1),
                  tstep + 1, MPI_COMM_WORLD, &b2r_req_recv[req_cnt++]);
    }
    return req_cnt;
}

/*
 ***************************************************************************************************
 * func   name: b2r_send_pad_z
 * description: when finished local updating (for R time steps), needs to pack the new block, 
                sync to neighbourhoods, this function handle the nonblocking sending in Z direction.
 * parameters :
 *             tstep: current simulation time step
 * return: none
 ***************************************************************************************************
 */
int b2r_send_pad_z(int tstep, char **pad_send_buf)
{
    int rank, req_cnt = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int block_id_x = (rank % (BLOCK_DIM_X * BLOCK_DIM_Y)) % BLOCK_DIM_X;
    int block_id_y = (rank % (BLOCK_DIM_X * BLOCK_DIM_Y)) / BLOCK_DIM_X;
    int block_id_z = (rank / (BLOCK_DIM_X * BLOCK_DIM_Y));
    int Rd = B2R_R * B2R_D;
       
    // left 
    if(b2r_halo_update[4] == 1)
    {
        MPI_Isend(pad_send_buf[0], B2R_BLOCK_SIZE_X*B2R_BLOCK_SIZE_Y*Rd*item_size, MPI_BYTE,
                  b2r_3D_get_rank(block_id_x, block_id_y, block_id_z-1),
                  tstep + 1, MPI_COMM_WORLD, &b2r_req_send[req_cnt++]);
    }    
    // right 
    if(b2r_halo_update[5] == 1)
    {
        MPI_Isend(pad_send_buf[1], B2R_BLOCK_SIZE_X*B2R_BLOCK_SIZE_Y*Rd*item_size, MPI_BYTE,
                  b2r_3D_get_rank(block_id_x, block_id_y, block_id_z+1),
                  tstep + 0, MPI_COMM_WORLD, &b2r_req_send[req_cnt++]);
    }
    return req_cnt;
}

/*
 ***************************************************************************************************
 * func   name: b2r_sync_wait
 * description: when finished local updating (for R time steps), needs to pack the new block, 
                sync to neighbourhoods, this function handle the nonblocking sending in Z direction.
 * parameters :
 *             tstep: current simulation time step
 * return: none
 ***************************************************************************************************
 */
void b2r_sync_wait(int sent_cnt, int recv_cnt){
    MPI_Waitall(sent_cnt, b2r_req_send, b2r_stt_send);  // wait untill all sent data have been received
    MPI_Waitall(recv_cnt, b2r_req_recv, b2r_stt_recv);  // wait untill all requested receive data arrived     
}
#endif /* b2r_comm_h */
