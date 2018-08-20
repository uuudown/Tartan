/*
 *  file name: b2r.cpp
 *
 *  b2r.cpp is the main of a B2R based simulation. 
 *  model update entry function should be in the for-loop in main function
 *
 *  openmp was used to parallel handle halo pack/unpack, environment variable OMP_NUM_THREADS must be set
 *  when compile the openmp supporting version, e.g., export OMP_NUM_THREADS=4, otherwise, each of the MPI
 *  process will invoke n (n = number of cores) threads, which may exceed number of threads limitation
 *  and also inefficient.
 *
 */
#include <mpi.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <unistd.h>
#include "b2r.h"
#include "b2r_comm.hpp"
#include "eqwp.hpp"

using namespace std;

/*
***************************************************************************************************
*                          Global Variables and external functions                                *
***************************************************************************************************
*/
extern void boundary_updating_direct(int time_step, CELL_DT *d_grid);
extern void boundary_updating(int time_step, CELL_DT *d_grid);
extern void b2r_sync_init();
extern void b2r_sync_flag(int rank);
extern void eqwp_cuda_init(int Ngx, int Ngy, int Ngz);
extern void b2r_sync_finalize();

/*
***************************************************************************************************
* func   name: simu_env_init
* description: this function response the simulation / B2R environment, e.g., sending/receiving 
               buffer allocating, global variables/parameters initialization
* parameters :
*             none
* return: none
***************************************************************************************************
*/
void simu_env_init(int np, int rank)
{
    B2R_B_X = ENV_DIM_X / BLOCK_DIM_X;
    B2R_B_Y = ENV_DIM_Y / BLOCK_DIM_Y;
    B2R_B_Z = ENV_DIM_Z / BLOCK_DIM_Z;

#if _ENV_3D_
    if(B2R_R*B2R_D > min(B2R_B_X, min(B2R_B_Y, B2R_B_Z))/2){
        if(rank == 0){
            cout << "should meet condition: 1 <= B2R_R*B2R_D <= min(B2R_B_X, B2R_B_Y, B2R_B_Z)/2" << endl;
        }
        exit(1);
    }
#endif
    B2R_BLOCK_SIZE_X = (B2R_B_X + 2*B2R_D*B2R_R);
    B2R_BLOCK_SIZE_Y = (B2R_B_Y + 2*B2R_D*B2R_R);
#if _ENV_3D_
    B2R_BLOCK_SIZE_Z = (B2R_B_Z + 2*B2R_D*B2R_R);
#else
    B2R_BLOCK_SIZE_Z = 1;
#endif

/*
    int Rd = B2R_R * B2R_D;
    int buffer_size = sizeof(CELL_DT) * max(max(Rd*B2R_B_Y*B2R_B_Z, B2R_BLOCK_SIZE_X*Rd*B2R_B_Z), 
                                            B2R_BLOCK_SIZE_X*B2R_BLOCK_SIZE_Y*Rd);
    // sending buffer                                        
    pad_send_buf[0] = new char[buffer_size]();    // direction 1(to: left/up/infront)
    pad_send_buf[1] = new char[buffer_size]();    // direction 2(to: right/down/behind)
    // receiving buffer
    pad_recv_buf[0] = new char[buffer_size]();    // direction 1(from: left/up/infront)
    pad_recv_buf[1] = new char[buffer_size]();    // direction 2(from: right/down/behind)    
*/
    b2r_sync_init(); // allocate send/recv buffer in device memory 

    char filename[100];
#if _DEBUG_
    sprintf( filename, "R-%d-stdout-%d.txt", B2R_R, rank );
    freopen( filename, "w", stdout );
#else
    if(rank == 0)
    {
#if _MDL_DEV_ == 0
        sprintf(filename, "R-%d-D-%d-ENV-%dX%dX%d-BLOCK-%dX%dX%d-CPU.txt", B2R_R, B2R_D, 
                B2R_B_X*BLOCK_DIM_X, B2R_B_Y*BLOCK_DIM_Y, B2R_B_Z*BLOCK_DIM_Z, BLOCK_DIM_X, BLOCK_DIM_Y, BLOCK_DIM_Z );
#else
        sprintf(filename, "R-%d-D-%d-ENV-%dX%dX%d-BLOCK-%dX%dX%d-GPU.txt", B2R_R, B2R_D, 
                B2R_B_X*BLOCK_DIM_X, B2R_B_Y*BLOCK_DIM_Y, B2R_B_Z*BLOCK_DIM_Z, BLOCK_DIM_X, BLOCK_DIM_Y, BLOCK_DIM_Z );
#endif                
        freopen( filename, "w", stdout );
    }
#endif
}
/*
***************************************************************************************************
* func   name: simu_env_finalize
* description: this function release memroy resources in head (allocated by new operator),
* parameters :
*             none
* return: none
***************************************************************************************************
*/
void simu_env_finalize()
{

}
/*
***************************************************************************************************
* func   name: main
* description: the main access function, it has several arguments. must be given in execution, 
                i.e., the arguments, R in B2R (not the same as halo size, halo = Delta*R),
                number of blocks (MPI sync level) in x, y and z dimension (np == x*y*z). 
                i.e., four in total.
* parameters : 
*             none
* return: none
***************************************************************************************************
*/
int main(int argc, char *argv[])
{
    int rank, np;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if(argc < 4){
        if(rank == 0){
            cout << "not sufficient parameter given for execution." << endl;
            cout << "please set parameter B2R_B_R, BLOCK_DIM_(X,Y,Z)" << endl;
        }
        exit(1);
    }else{
        B2R_R = atoi(argv[1]);
        BLOCK_DIM_X = atoi(argv[2]);
        BLOCK_DIM_Y = atoi(argv[3]);
#if _ENV_3D_
        BLOCK_DIM_Z = atoi(argv[4]);
#else
        BLOCK_DIM_Z = 1;
#endif        
        if(rank == 0){
            cout << "arguments received, where: B2R_R = " << B2R_R << ", BLOCK_DIM_X = " << BLOCK_DIM_X 
                 << ", BLOCK_DIM_Y = " << BLOCK_DIM_Y << ", BLOCK_DIM_Z = " << BLOCK_DIM_Z << endl;
        }
    }
    // comment for weak scaling testing

    if(ENV_DIM_X % BLOCK_DIM_X != 0 || ENV_DIM_Y % BLOCK_DIM_Y != 0 || ENV_DIM_Z % BLOCK_DIM_Z != 0){
        if(rank == 0){
            cout << "GBL_ENV_(X,Y,Z) should be perfectly divisible by the given block size BLOCK_DIM_(X,Y,Z) separately!" << endl;
        }
        exit(1);
    }

    if(np != BLOCK_DIM_X*BLOCK_DIM_Y*BLOCK_DIM_Z){
        if(rank == 0){
            cout << "improper np=" << np << ", np should equal with the total number of blocks=" 
                 << BLOCK_DIM_X*BLOCK_DIM_Y*BLOCK_DIM_Z << endl;
        }
        exit(1);
    }
    
    b2r_sync_flag(rank);                               // update global variable b2r_halo_update, determine halos need updating
    simu_env_init(np, rank);
    eqwp_cuda_init(B2R_BLOCK_SIZE_X, B2R_BLOCK_SIZE_Y, B2R_BLOCK_SIZE_Z);
    
    struct timeval t_all_s, t_all_e, compu_s, compu_e, sync_s, sync_e;
    double elapsedtime, mpi_sync_time = 0.0, hd_time = 0.0, device_compu_time = 0.0;
    double device_time1, device_compu_time1;
    MPI_Barrier( MPI_COMM_WORLD );
    gettimeofday(&t_all_s, NULL);                // start to count the time takes on simulation
    
    for (int i = 0; i < N_ITER; i+=B2R_R)
    {
        //MPI_Barrier( MPI_COMM_WORLD );
        gettimeofday(&sync_s, NULL);       
        
        // update subdomain boundaries via MPI 
        /*
        boundary_updating_direct(i, d_vx);                           
        boundary_updating_direct(i, d_vy);
        boundary_updating_direct(i, d_vz);
        
        boundary_updating_direct(i, d_sigma_xx);             
        boundary_updating_direct(i, d_sigma_xy);
        boundary_updating_direct(i, d_sigma_xz);
        boundary_updating_direct(i, d_sigma_yy);
        boundary_updating_direct(i, d_sigma_yz);
        boundary_updating_direct(i, d_sigma_zz);       
        */ 
        boundary_updating(i, d_vx);                           
        boundary_updating(i, d_vy);
        boundary_updating(i, d_vz);
        
        boundary_updating(i, d_sigma_xx);             
        boundary_updating(i, d_sigma_xy);
        boundary_updating(i, d_sigma_xz);
        boundary_updating(i, d_sigma_yy);
        boundary_updating(i, d_sigma_yz);
        boundary_updating(i, d_sigma_zz);  
                
        gettimeofday(&sync_e, NULL);
        
        mpi_sync_time += sync_e.tv_sec - sync_s.tv_sec + (sync_e.tv_usec - sync_s.tv_usec) / 1e6;
        
        gettimeofday(&compu_s, NULL);            // to record all the time on GPU (include computatio and data transfering)
        // computation here

        device_compu_time1 = eqwp_gpu_main(i, B2R_R, B2R_BLOCK_SIZE_X, B2R_BLOCK_SIZE_Y, B2R_BLOCK_SIZE_Z, (i+B2R_R) >= (N_ITER));
        
        gettimeofday(&compu_e, NULL);            // to record all the time on GPU (include computatio and data transfering)
        device_time1 = compu_e.tv_sec - compu_s.tv_sec + (compu_e.tv_usec - compu_s.tv_usec) / 1e6;
        hd_time += ( device_time1 - device_compu_time1); // accumulation of host to device and device to host time
        device_compu_time += device_compu_time1;
    } 
    // end of simulation
    MPI_Barrier( MPI_COMM_WORLD );
    
#if _DEBUG_ == 0
    if(rank == 0)
    {
        gettimeofday(&t_all_e, NULL);
        elapsedtime = t_all_e.tv_sec - t_all_s.tv_sec + (t_all_e.tv_usec - t_all_s.tv_usec) / 1e6;
        cout << "Execution/Model Parameters: B2R_R = " << B2R_R << ", B2R_D: " << B2R_D << ", ENV: " << B2R_B_X*BLOCK_DIM_X
             << " X "<< B2R_B_Y*BLOCK_DIM_Y << " X "<< B2R_B_Z*BLOCK_DIM_Z << " X "
             << ", BLOCK_DIM_X = " << BLOCK_DIM_X << ", BLOCK_DIM_Y = " 
             << BLOCK_DIM_Y << ", BLOCK_DIM_Z = " << BLOCK_DIM_Z << endl;
        cout << N_ITER << " Iterations with B2R_R=" << B2R_R << " took: " << elapsedtime << " seconds" << endl;
        cout << "h <-> d took: " << hd_time << " seconds" << endl;
        cout << "device computation took: " << device_compu_time << " seconds" << endl;
        cout << "mpi synchronization took: " << mpi_sync_time << " seconds" << endl;
    }
#endif

#if _DEBUG_ == 1
    print_block_data(grid_displ_y);
#endif
    // free newed memory
    model_finalize();
    simu_env_finalize();
    MPI_Barrier( MPI_COMM_WORLD );
    MPI_Finalize();
    return 0;
}
