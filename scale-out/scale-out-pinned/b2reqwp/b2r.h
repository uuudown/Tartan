#ifndef _B2R_
#define _B2R_

#include <mpi.h>
#include <omp.h>
#include <time.h>
#include <sys/time.h>
#include "b2r_config.h"

#define RAND_SEED 10000

#define N_ITER    600
/*
+---------+----------+-----+-----------+------+-------+-----------+------+------------+---------+--------+
| index   | 0        | 1   | 2         | 3    | 4     | 5         | 6    | 7          | 8       | 9      |
+---------+----------+-----+-----------+------+-------+-----------+------+------------+---------+--------+
| block   | top-left | top | top-right | left | right | down-left | down | down-right | infront | behind |
+---------+----------+-----+-----------+------+-------+-----------+------+------------+---------+--------+
+---------+----------+-----+-----------+------+-------+-----------+------+------------+---------+--------+
| index   | 10       | 11  | 12        | 13   | 14    | 15        | 16   | 17         |         |        |
+---------+----------+-----+-----------+------+-------+-----------+------+------------+---------+--------+
| infront | top-left | top | top-right | left | right | down-left | down | down-right |         |        |
+---------+----------+-----+-----------+------+-------+-----------+------+------------+---------+--------+
+---------+----------+-----+-----------+------+-------+-----------+------+------------+---------+--------+
| index   | 18       | 19  | 20        | 21   | 22    | 23        | 24   | 25         |         |        |
+---------+----------+-----+-----------+------+-------+-----------+------+------------+---------+--------+
| behind  | top-left | top | top-right | left | right | down-left | down | down-right |         |        |
+---------+----------+-----+-----------+------+-------+-----------+------+------------+---------+--------+
*/

// update direction by direction, i.e., X -> Y -> Z
MPI_Request b2r_req_send[2];
MPI_Request b2r_req_recv[2];
MPI_Status  b2r_stt_send[2];
MPI_Status  b2r_stt_recv[2];

// 0 - unavailable, 1 - available, -1 periodical
#if _ENV_3D_
    char b2r_halo_update[6];
#else
    char b2r_halo_update[4];
#endif

#if _ENV_3D_
struct cell_data_structure
{
    int x, y, z;
};
#else
struct cell_data_structure
{
    int x, y;
};
#endif

/* data structure of cells, i.e., patch*/
//typedef struct cell_data_structure CELL_DT;

// will be [BLOCK_DIM_X][BLOCK_DIM_Y][BLOCK_DIM_Z];
CELL_DT *h_vx, *h_vy, *h_vz;
CELL_DT *h_sigma_xx,  *h_sigma_xy, *h_sigma_xz, *h_sigma_yy, *h_sigma_yz, *h_sigma_zz;

CELL_DT *d_vx, *d_vy, *d_vz;
CELL_DT *d_sigma_xx, *d_sigma_xy, *d_sigma_xz, *d_sigma_yy, *d_sigma_yz, *d_sigma_zz;

/*
*********************************************************************
                    Global Variables
                    all capital variables denotes parameter
*********************************************************************
*/
// these will be determined by ENV grid size and number of blocks in each dimension
// B2R_B_ = ENV_ / BLOCK_DIM_
int B2R_B_X, B2R_B_Y, B2R_B_Z;

// number of blocks, these will be given by users through command execution parameter
int BLOCK_DIM_X;
int BLOCK_DIM_Y;
int BLOCK_DIM_Z;

// local block size, (B+2*R*delta)
int B2R_BLOCK_SIZE_X;   // (B2R_B_X + 2*B2R_D*B2R_R)
int B2R_BLOCK_SIZE_Y;   // (B2R_B_Y + 2*B2R_D*B2R_R)
int B2R_BLOCK_SIZE_Z;   // (B2R_B_Z + 2*B2R_D*B2R_R)
int B2R_R;              // specified by user via executing parameter

int item_size = sizeof(CELL_DT);  // size of memory for one cell

#endif
