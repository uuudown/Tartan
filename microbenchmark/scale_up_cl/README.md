==========================================================================================
 This microbenchmark belongs to the Tartan Benchmark Suite, which contains microbenchmarks
 and benchmark applications for evaluating multi-GPUs and multi-GPU interconnect. Please see
 our IISWC-18 paper titled "Tartan: Evaluating Modern GPU Interconnect via a Multi-GPU
 Benchmark Suite" for detail.

        Version:  1.0
        Created:  01/24/2018 03:52:11 PM

         Author:  Ang Li, PNNL
        Website:  http://www.angliphd.com  
==========================================================================================

##Description:  

This microbenchmark is to measure the latency, bandwidth and communication
efficiency (with increased message size) for Collective Communication (CL w.r.s.p P2P) 
for PCI-e, NVLink-V1 in P100 DGX-1 and NVLink-V2 in V100 DGX-1 with increased number of GPUs. 
Please refer to NCCL_TEST_README.md for compilation and run.

##Configuration:

NCCL-V1 is opensourced but only adopts PCI-e communication. NCCL-V2 leverages
both PCI-e and NVLink. However, it picks the appropriate interconnect all by itself (i.e., not
configurable. In addition, it is closedsourced. So our plan to evaluate the difference among
PCI-e and NVLink is that: for PCI-e, we set $NCCL_HOME in the Makefile under src to link to 
NCCL-V1; for NVLink, we set $NCCL_HOME to NCCL-V2. We conduct both test on P100 DGX-1 and 
V100 DGX-1 to compare the difference among NVLink-V1 and NVLink-V2. 
Please refer to our paper for detail.

##Testing:

The NCCL-V1 linked binaries are put into nccl_v1 folder. The NCCL-V2 linked binaries
are put into nccl_v2 folder. We provide a python script called "run_test.py" to measure the 
latency, bandwidth and communication efficiency. 

