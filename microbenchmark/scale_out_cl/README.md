==========================================================================================
 This microbenchmark belongs to the Tartan Benchmark Suite, which contains microbenchmarks
 and benchmark applications for evaluating multi-GPUs and multi-GPU interconnect. Please see
 our IISWC-18 paper titled "Tartan: Evaluating Modern GPU Interconnect via a Multi-GPU 
 Benchmark Suite" for detail.

        Version:  1.0
        Created:  01/24/2018 05:07:38 PM

         Author:  Ang Li, PNNL
        Website:  http://www.angliphd.com  
==========================================================================================

##Description:

This microbenchmark is to measure the latency, bandwidth and communication
efficiency (with increased message size) for Collective Communication (CL wrsp P2P) 
among GPU-integrated HPC nodes. The test platform is SummitDev supercomputer 
in Oak Ridge National Laboratory (ORNL).

##Compile:  

ORNL-SummitDev: we have already modified the Makefile under src for summitdev env.
```shell
    $ make
```

##Run:  

ORNL-SummitDev: you need to specify your project id in run.lsf and runtest.lsf first.
```shell
    $ cd build
    $ bsub -o result.txt run.lsf 
```

For complete measure of latency, bandwidth and efficiency, run script:
```shell
    $ python runtest.py
```

##Configuration: 

NCCL-V1 does not support multi-nodes so we can only test NCCL-V2. In addition, 
NCCL-V2 will automatically enable GPU-RDMA (based on our observation that enabling RDMA or not 
does not make any difference for performance), so there is no opportunity for us to observe
the difference by enabling RDMA or not. But still, we can measure the latency, bandwidth and
communication efficiency for collective operations NCCL supported among increased node number.
Just modify the "runtest.py" under build dir accordingly. For detailed observation, please see
our paper. 

