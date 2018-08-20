==========================================================================================
 This microbenchmark belongs to the Tartan Benchmark Suite, which contains microbenchmarks
 and benchmark applications for evaluating multi-GPUs and multi-GPU interconnect. Please see
 our IISWC-18 paper titled "Tartan: Evaluating Modern GPU Interconnect via a Multi-GPU 
 Benchmark Suite" for detail.

        Version:  1.0
       Created:  01/24/2018 04:40:11 PM

         Author:  Ang Li, PNNL
        Website:  http://www.angliphd.com  
==========================================================================================

##Description: 

This microbenchmark is to obtain the latency, bandwidth and efficiency
for P2P InfiniBand (IB) communication among GPU-integrated HPC nodes. The objective is to 
measure the difference when GPU-Direct RDMA is setting and when pinned host memory is utilized.
The test platform is SummitDev supercomputer in Oak Ridge National Laboratory (ORNL).

##Compile: (on ORNL-SummitDev)

```shell
    $ ./compile.sh
```
##Run: 

On ORNL-SummitDev, you need to config runtest.lsf first.
```shell
    $ bsub -o results.txt runtest.lsf
```

##Configuration: 

There are several options you can tune. 
(1) Using pinned/unpinned host memory for MPI communication buffer or directly access GPU memory
(2) Enable/disable GPU-RDMA for InfiniBand

For (1): Pinned host memory: you can uncomment "cudaMallocHost" and "cudaFreeHost" in main.cpp;
         Unpinned host memory: you can uncomment "new" and "free" in main.cpp
         GPU device memory: coment the definition of "NODIRECT"
For (2): Enable RDMA: "source $OLCF_SPECTRUM_MPI_ROOT/jsm_pmix/bin/export_smpi_env -gpu"
                      in runtest.lsf
         Disable RDMA: As default

### Note: 

The combination of (1)GPU device memory with (2) Disable RDMA leads to segmentation error.
      Other combinations work well. Please see our paper for detail.

