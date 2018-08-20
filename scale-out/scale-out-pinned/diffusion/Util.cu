//
//  Util.cu
//  Burgers3d-GPU-MPI
//
//  Created by Manuel Diaz on 7/26/16.
//  Copyright © 2016 Manuel Diaz. All rights reserved.
//

#include "DiffusionMPICUDA.h"

#define checkCuda(error) __checkCuda(error, __FILE__, __LINE__)

/*********************************************/
/* A method for checking error in CUDA calls */
/*********************************************/
inline void __checkCuda(cudaError_t error, const char *file, const int line)
{
	#if defined(DEBUG) || defined(_DEBUG)
	if (error != cudaSuccess)
	{
		printf("checkCuda error at %s:%i: %s\n", file, line, cudaGetErrorString(cudaGetLastError()));
		exit(-1);
	}
	#endif

	return;
}

/****************************************************/
/* Function that scans for devices on a single node */
/****************************************************/
extern "C" int DeviceScan()
{
	int numberOfDevices;
	checkCuda(cudaGetDeviceCount(&numberOfDevices));

	return numberOfDevices;
}

/*******************************************************************/
/* Function that checks if multiple GPUs are available on the node */
/*******************************************************************/
extern "C" void MPIDeviceCheck(int rank, int numberOfProcesses, int numberOfDevices)
{
	if (numberOfDevices < 2)
	{
		printf("Less than two devices were found.\n");
		printf("Exiting...\n");
		FinalizeMPI();
		exit(-1);

	}

	if (numberOfProcesses > numberOfDevices)
	{
		printf("Number of processors exceeds the number of GPUs\n");
		printf("Exiting...\n");
		FinalizeMPI();
		exit(-1);
	}
}

/*****************************************************************/
/* Function that assigns a single device to a single MPI process */
/*****************************************************************/
extern "C" void AssignDevices(int rank)
{
	int numberOfDevices = 0;

	checkCuda(cudaGetDeviceCount(&numberOfDevices));
	checkCuda(cudaSetDevice(rank % numberOfDevices));

	printf("Process %d -> GPU%d\n", rank, rank % numberOfDevices);
}

/************************************************************************/
/* Function that checks if ECC is turned on for the devices on the node */
/************************************************************************/
extern "C" void ECCCheck(int rank)
{
	cudaDeviceProp properties;

    checkCuda(cudaGetDeviceProperties(&properties, rank));

    if (properties.ECCEnabled == 1)
    {
        printf("ECC is turned on for device #%d\n", rank);
    }
    else
    {
        printf("ECC is turned off for device #%d\n", rank);
    }
}

/**********************************/
/* Computes the thread block size */
/**********************************/
extern "C" int getBlock(int n, int block)
{
	return (n+2)/block + ((n+2)%block == 0?0:1);
}
