/*
 * =====================================================================================
 *
 *       Filename:  main.cpp
 *
 *    Description:  This microbenchmark is to obtain the latency, bandwidth and efficiency
 *                  for P2P InfiniBand (IB) communication among GPU-integrated HPC nodes. 
 *                  The objective is to measure the difference when GPU-Direct RDMA is 
 *                  setting and when pinned host memory is utilized. The test platform
 *                  is SummitDev supercomputer in Oak Ridge National Laboratory (ORNL).
 *                  The code is modified from "MPI-GPU-BW" in GitHub.
 *
 *        Version:  1.0
 *        Created:  01/24/2018 04:36:31 PM
 *       Revision:  none
 *       Compiler:  mpicxx (Tested with IBM spectrum-mpi) 
 *
 *         Author:  Ang Li, PNNL
 *        Website:  http://www.angliphd.com  
 *
 * =====================================================================================
 */




#include <mpi.h>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <chrono>
#include <assert.h>
#include <stdio.h>

#define NODIRECT 1

int main(int argc, char **argv) {
    
    MPI_Init(&argc, &argv);
    int rank;
    int nprocs;

    int maxnamelen;
    char myname[128];
    
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    MPI_Get_processor_name(myname, &maxnamelen);
    printf("MPI task %d starts on hosts %s\n",rank, myname);

    if (nprocs != 2) 
    {
        if (rank == 0) 
            std::cout << "Error: You need to run on exactly 2 ranks or GPUs" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    auto rounds = std::atoi(argv[3]);
    auto maxLen = std::atoi(argv[4]);
    int dev0 = std::atoi(argv[1]);
    int dev1 = std::atoi(argv[2]);

    if(rank == 0)
        cudaSetDevice(dev0);
    else
        cudaSetDevice(dev1);

    FILE *outfile = NULL;
    if (rank == 0)
    {
        outfile = fopen("/ccs/home/angli/tartan/MPI-GPU-BW/out.txt","w");
    }

    int *a = NULL;
    int *b = NULL;

    cudaMalloc((void**)&a, sizeof(int)*maxLen); 		
    cudaMalloc((void**)&b, sizeof(int)*maxLen); 		

#ifdef NODIRECT
    int *c;
    int *d;
    c = new int[maxLen];
    d = new int[maxLen];

    //cudaMallocHost((void**)&c, sizeof(int)*maxLen); 		
    //cudaMallocHost((void**)&d, sizeof(int)*maxLen); 		
#endif

    for (int len = 1; len <= maxLen; len *= 2) {
        std::chrono::duration<double> best = std::chrono::duration<double>::max();
        
        for (int round = 0; round < rounds; round++) {
        // timer start
            MPI_Barrier(MPI_COMM_WORLD);
            auto start = std::chrono::system_clock::now();
            if(rank == 0)
            {
                // MPI send
#ifdef NODIRECT
                cudaMemcpy(c,a,len*sizeof(int), cudaMemcpyDeviceToHost);
                MPI_Send(c, len, MPI_INT, 1, 0, MPI_COMM_WORLD);
#else
                MPI_Send(a, len, MPI_INT, 1, 0, MPI_COMM_WORLD);
#endif
            }
            else
            {
                // MPI recv
#ifdef NODIRECT
                MPI_Recv(d, len, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                cudaMemcpy(b,d,len*sizeof(int),cudaMemcpyHostToDevice);
#else
                MPI_Recv(b, len, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
#endif
            }
            auto end = std::chrono::system_clock::now();
            std::chrono::duration<double> elapsed_seconds = end-start;
            best = std::min(best, elapsed_seconds);

        }
        MPI_Barrier(MPI_COMM_WORLD);

        if(rank == 0)
        {
            std::cout << len*sizeof(int) << " BW : " << (len*4.0/1E9)/best.count() << std::endl;
            if (outfile != NULL)
            {
                fprintf(outfile, "SZ:%lu,BW:%.6lf,LT:%.3lf\n", len*sizeof(int), (len*4.0/1E9)/best.count(), best.count()*1E3); 
            }	

        }

    }

    if (rank == 0)
    {
        fclose(outfile);
    }

#ifdef NODIRECT
    free(c);
    free(d);
    
    //cudaFreeHost(c);
    //cudaFreeHost(d);
#endif

    cudaFree(a);
    cudaFree(b);

    MPI_Finalize();

    return 0;
}
