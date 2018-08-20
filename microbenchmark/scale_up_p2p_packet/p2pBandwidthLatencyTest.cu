/*
 * =====================================================================================
 *
 *       Filename:  p2pBandwidthLatencyTest.cu
 *
 *    Description:  This microbenchmark is to obtain the latency & uni/bi-directional
 *                  bandwidth for PCI-e and NVLink with increased message size. 
 *                  Please see our IISWC-18 paper titled "Tartan: Evaluating Modern GPU
 *                  Interconnect via a Multi-GPU Benchmark Suite". 
 *                  The code is modified from the p2pBandwidthLatencyTest app in 
 *                  NVIDIA CUDA-SDK. Please follow NVIDIA's EULA for end usage. 
 *
 *        Version:  1.0
 *        Created:  01/24/2018 02:12:31 PM
 *       Revision:  none
 *       Compiler:  nvcc
 *
 *         Author:  Ang Li, PNNL
 *        Website:  http://www.angliphd.com  
 *
 * =====================================================================================
 */


/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#include <cstdio>
#include <vector>

#include <helper_cuda.h>

using namespace std;

const char *sSampleName = "P2P (Peer-to-Peer) GPU Bandwidth Latency Test";

//Macro for checking cuda errors following a cuda launch or api call
#define cudaCheckError() {                                          \
        cudaError_t e=cudaGetLastError();                                 \
        if(e!=cudaSuccess) {                                              \
            printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    }
__global__ void delay(int * null) {
  float j=threadIdx.x;
  for(int i=1;i<10000;i++)
      j=(j+1)/j;

  if(threadIdx.x == j) null[0] = j;
}

void checkP2Paccess(int numGPUs)
{
    for (int i=0; i<numGPUs; i++)
    {
        cudaSetDevice(i);

        for (int j=0; j<numGPUs; j++)
        {
            int access;
            if (i!=j)
            {
                cudaDeviceCanAccessPeer(&access,i,j);
                printf("Device=%d %s Access Peer Device=%d\n", i, access ? "CAN" : "CANNOT", j);
            }
        }
    }
    printf("\n***NOTE: In case a device doesn't have P2P access to other one, it falls back to normal memcopy procedure.\nSo you can see lesser Bandwidth (GB/s) in those cases.\n\n");
}

void outputBandwidthMatrix(int numGPUs, bool p2p, FILE* outfile, int datasize)
{
    int numElems=datasize;
    int repeat=5;
    vector<int *> buffers(numGPUs);
    vector<cudaEvent_t> start(numGPUs);
    vector<cudaEvent_t> stop(numGPUs);

    for (int d=0; d<numGPUs; d++)
    {
        cudaSetDevice(d);
        //cudaMalloc(&buffers[d],numElems*sizeof(int));
        cudaMalloc(&buffers[d],numElems);
        cudaCheckError();
        cudaEventCreate(&start[d]);
        cudaCheckError();
        cudaEventCreate(&stop[d]);
        cudaCheckError();
    }

    vector<double> bandwidthMatrix(numGPUs*numGPUs);

    for (int i=0; i<numGPUs; i++)
    {
        cudaSetDevice(i);

        for (int j=0; j<numGPUs; j++)
        {
            int access;
            if(p2p) {
                cudaDeviceCanAccessPeer(&access,i,j);
                if (access)
                {
                    cudaDeviceEnablePeerAccess(j,0 );
                    cudaCheckError();
                }
            }

            cudaDeviceSynchronize();
            cudaCheckError();
            delay<<<1,1>>>(NULL);
            cudaEventRecord(start[i]);

            for (int r=0; r<repeat; r++)
            {
                //cudaMemcpyPeerAsync(buffers[i],i,buffers[j],j,sizeof(int)*numElems);
                cudaMemcpyPeerAsync(buffers[i],i,buffers[j],j,numElems);
            }

            cudaEventRecord(stop[i]);
            cudaDeviceSynchronize();
            cudaCheckError();

            float time_ms;
            cudaEventElapsedTime(&time_ms,start[i],stop[i]);
            double time_s=time_ms/1e3;

            //double gb=numElems*sizeof(int)*repeat/(double)1e9;
            double gb=numElems*repeat/(double)1e9;


            if(i==j) gb*=2;  //must count both the read and the write here
            bandwidthMatrix[i*numGPUs+j]=gb/time_s;
            if (p2p && access)
            {
                cudaDeviceDisablePeerAccess(j);
                cudaCheckError();
            }
        }
    }

    fprintf(outfile, "   D\\D");

    for (int j=0; j<numGPUs; j++)
    {
        fprintf(outfile, "%6d ", j);
    }

    fprintf(outfile, "\n");

    for (int i=0; i<numGPUs; i++)
    {
        fprintf(outfile, "%6d ",i);

        for (int j=0; j<numGPUs; j++)
        {
            fprintf(outfile, "%6.02f ", bandwidthMatrix[i*numGPUs+j]);
        }

        fprintf(outfile, "\n");
    }

    for (int d=0; d<numGPUs; d++)
    {
        cudaSetDevice(d);
        cudaFree(buffers[d]);
        cudaCheckError();
        cudaEventDestroy(start[d]);
        cudaCheckError();
        cudaEventDestroy(stop[d]);
        cudaCheckError();
    }
}

void outputBidirectionalBandwidthMatrix(int numGPUs, bool p2p, FILE* outfile, int datasize)
{
    int numElems=datasize;
    int repeat=5;
    vector<int *> buffers(numGPUs);
    vector<cudaEvent_t> start(numGPUs);
    vector<cudaEvent_t> stop(numGPUs);
    vector<cudaStream_t> stream0(numGPUs);
    vector<cudaStream_t> stream1(numGPUs);

    for (int d=0; d<numGPUs; d++)
    {
        cudaSetDevice(d);
        //cudaMalloc(&buffers[d],numElems*sizeof(int));
        cudaMalloc(&buffers[d],numElems);
        cudaCheckError();
        cudaEventCreate(&start[d]);
        cudaCheckError();
        cudaEventCreate(&stop[d]);
        cudaCheckError();
        cudaStreamCreate(&stream0[d]);
        cudaCheckError();
        cudaStreamCreate(&stream1[d]);
        cudaCheckError();
    }

    vector<double> bandwidthMatrix(numGPUs*numGPUs);

    for (int i=0; i<numGPUs; i++)
    {
        cudaSetDevice(i);

        for (int j=0; j<numGPUs; j++)
        {
            int access;
            if(p2p) {
                cudaDeviceCanAccessPeer(&access,i,j);
                if (access)
                {
                    cudaSetDevice(i);
                    cudaDeviceEnablePeerAccess(j,0);
                    cudaCheckError();
                    cudaSetDevice(j);
                    cudaDeviceEnablePeerAccess(i,0);
                    cudaCheckError();
                }
            }

            cudaSetDevice(i);
            cudaDeviceSynchronize();
            cudaCheckError();
            delay<<<1,1>>>(NULL);
            cudaEventRecord(start[i]);

            for (int r=0; r<repeat; r++)
            {
                //cudaMemcpyPeerAsync(buffers[i],i,buffers[j],j,sizeof(int)*numElems,stream0[i]);
                //cudaMemcpyPeerAsync(buffers[j],j,buffers[i],i,sizeof(int)*numElems,stream1[i]);
                cudaMemcpyPeerAsync(buffers[i],i,buffers[j],j,numElems,stream0[i]);
                cudaMemcpyPeerAsync(buffers[j],j,buffers[i],i,numElems,stream1[i]);
            }

            cudaEventRecord(stop[i]);
            cudaDeviceSynchronize();
            cudaCheckError();

            float time_ms;
            cudaEventElapsedTime(&time_ms,start[i],stop[i]);
            double time_s=time_ms/1e3;

            //double gb=2.0*numElems*sizeof(int)*repeat/(double)1e9;
            double gb=2.0*numElems*repeat/(double)1e9;

            if(i==j) gb*=2;  //must count both the read and the write here
            bandwidthMatrix[i*numGPUs+j]=gb/time_s;
            if(p2p && access)
            {
                cudaSetDevice(i);
                cudaDeviceDisablePeerAccess(j);
                cudaSetDevice(j);
                cudaDeviceDisablePeerAccess(i);
            }
        }
    }

    fprintf(outfile, "   D\\D");

    for (int j=0; j<numGPUs; j++)
    {
        fprintf(outfile, "%6d ", j);
    }

    fprintf(outfile, "\n");

    for (int i=0; i<numGPUs; i++)
    {
        fprintf(outfile, "%6d ",i);

        for (int j=0; j<numGPUs; j++)
        {
            fprintf(outfile, "%6.02f ", bandwidthMatrix[i*numGPUs+j]);
        }

        fprintf(outfile, "\n");
    }

    for (int d=0; d<numGPUs; d++)
    {
        cudaSetDevice(d);
        cudaFree(buffers[d]);
        cudaCheckError();
        cudaEventDestroy(start[d]);
        cudaCheckError();
        cudaEventDestroy(stop[d]);
        cudaCheckError();
        cudaStreamDestroy(stream0[d]);
        cudaCheckError();
        cudaStreamDestroy(stream1[d]);
        cudaCheckError();
    }
}

void outputLatencyMatrix(int numGPUs, bool p2p, FILE* outfile, int datasize)
{
    int repeat=100;
    vector<int *> buffers(numGPUs);
    vector<cudaEvent_t> start(numGPUs);
    vector<cudaEvent_t> stop(numGPUs);

    for (int d=0; d<numGPUs; d++)
    {
        cudaSetDevice(d);
        cudaMalloc(&buffers[d],datasize);
        cudaCheckError();
        cudaEventCreate(&start[d]);
        cudaCheckError();
        cudaEventCreate(&stop[d]);
        cudaCheckError();
    }

    vector<double> latencyMatrix(numGPUs*numGPUs);

    for (int i=0; i<numGPUs; i++)
    {
        cudaSetDevice(i);

        for (int j=0; j<numGPUs; j++)
        {
            int access;
            if(p2p) {
                cudaDeviceCanAccessPeer(&access,i,j);
                if (access)
                {
                    cudaDeviceEnablePeerAccess(j,0);
                    cudaCheckError();
                }
            }
            cudaDeviceSynchronize();
            cudaCheckError();
            delay<<<1,1>>>(NULL);
            cudaEventRecord(start[i]);

            for (int r=0; r<repeat; r++)
            {
                cudaMemcpyPeerAsync(buffers[i],i,buffers[j],j,datasize);
            }

            cudaEventRecord(stop[i]);
            cudaDeviceSynchronize();
            cudaCheckError();

            float time_ms;
            cudaEventElapsedTime(&time_ms,start[i],stop[i]);

            latencyMatrix[i*numGPUs+j]=time_ms*1e3/repeat;
            if(p2p && access)
            {
                cudaDeviceDisablePeerAccess(j);
            }
        }
    }

    fprintf(outfile, "   D\\D");

    for (int j=0; j<numGPUs; j++)
    {
        fprintf(outfile, "%6d ", j);
    }

    fprintf(outfile, "\n");

    for (int i=0; i<numGPUs; i++)
    {
        fprintf(outfile, "%6d ",i);

        for (int j=0; j<numGPUs; j++)
        {
            fprintf(outfile, "%6.02f ", latencyMatrix[i*numGPUs+j]);
        }

        fprintf(outfile, "\n");
    }

    for (int d=0; d<numGPUs; d++)
    {
        cudaSetDevice(d);
        cudaFree(buffers[d]);
        cudaCheckError();
        cudaEventDestroy(start[d]);
        cudaCheckError();
        cudaEventDestroy(stop[d]);
        cudaCheckError();
    }
}

int main(int argc, char **argv)
{

    int numGPUs;
    cudaGetDeviceCount(&numGPUs);

    printf("[%s]\n", sSampleName);

    //output devices
    for (int i=0; i<numGPUs; i++)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop,i);
        printf("Device: %d, %s, pciBusID: %x, pciDeviceID: %x, pciDomainID:%x\n",i,prop.name, prop.pciBusID, prop.pciDeviceID, prop.pciDomainID);
    }

    checkP2Paccess(numGPUs);

    //Check peer-to-peer connectivity
    printf("P2P Connectivity Matrix\n");
    printf("     D\\D");

    for (int j=0; j<numGPUs; j++)
    {
        printf("%6d", j);
    }
    printf("\n");

    for (int i=0; i<numGPUs; i++)
    {
        printf("%6d\t", i);
        for (int j=0; j<numGPUs; j++)
        {
            if (i!=j)
            {
               int access;
               cudaDeviceCanAccessPeer(&access,i,j);
               printf("%6d", (access) ? 1 : 0);
            }
            else
            {
                printf("%6d", 1);
            }
        }
        printf("\n");
    }
    FILE *outfile = NULL;
    outfile = fopen("result.txt","w");
    if (outfile == NULL)
    {
        printf("Error! Cannot open file.");
        return -1;
    }

    for (int datasize = 1; datasize < 67108868*4; datasize *= 2)
    //for (int datasize = 32; datasize < 64; datasize *= 2)
    {
        fprintf(outfile, "===== %d =====\n", datasize);
        printf("===== %d =====\n", datasize);

        fprintf(outfile, "Uni-PCIe-Bandwidth\n");
        printf("Uni-PCIe-Bandwidth\n");
        outputBandwidthMatrix(numGPUs, false, outfile, datasize);

        fprintf(outfile, "Uni-NVLink-Bandwidth\n");
        printf("Uni-NVLink-Bandwidth\n");
        outputBandwidthMatrix(numGPUs, true, outfile, datasize);

        fprintf(outfile, "Bi-PCIe-Bandwidth\n");
        printf("Bi-PCIe-Bandwidth\n");
        outputBidirectionalBandwidthMatrix(numGPUs, false, outfile, datasize);

        fprintf(outfile, "Bi-NVLink-Bandwidth\n");
        printf("Bi-NVLink-Bandwidth\n");
        outputBidirectionalBandwidthMatrix(numGPUs, true, outfile, datasize);

        fprintf(outfile, "PCIe-Latency\n");
        printf("PCIe-Latency\n");
        outputLatencyMatrix(numGPUs, false, outfile, datasize);

        fprintf(outfile, "NVLink-Latency\n");
        printf("NVLink-Latency\n");
        outputLatencyMatrix(numGPUs, true, outfile, datasize);
    }


    fclose(outfile);


    exit(EXIT_SUCCESS);
}
