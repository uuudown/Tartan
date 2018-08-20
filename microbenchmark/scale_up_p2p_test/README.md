#==========================================================================================
 This microbenchmark belongs to the Tartan Benchmark Suite, which contains microbenchmarks
 and benchmark applications for evaluating multi-GPUs and multi-GPU interconnect. Please see
 our IISWC-18 paper titled "Tartan: Evaluating Modern GPU Interconnect via a Multi-GPU
 Benchmark Suite" for detail.

        Version:  1.0
        Created:  03/19/2018 02:12:31 PM

        Author:  Ang Li, PNNL
        Website:  http://www.angliphd.com  
#==========================================================================================

##Description:  

This microbenchmark is to measure the latency & uni/bi-directional 
bandwidth for PCI-e, NVLink-V1 in NVIDIA P100 DGX-1 and NVLink-V2 in V100 DGX-1. 
The code is modified from the p2pBandwidthLatencyTest app in NVIDIA CUDA-SDK. 

##Compile: 

Please modify shared.mk according to your machine environment, then compile using
```shell
    $ make clean
    $ make
```

##Run: 
```shell
    $ ./p2pTest 
```

##Configuration: 

This microbenchmark is able to measure the latency, uni/bi-direction bandwidth 
for P2P communication among GPUs in a single node (i.e., scale-up) via PCI-e and NVLink. Differnt
from the original SDK implementation, when two nodes are not directly connected via NVLink, we 
manually pick up a routing node. You may define or undefine "ASCENDING" in 
p2pBandwidthLatencyTest.cu to test different routing choice for evaluating the NUMA effect. 

##Results: 

dgx1_ascending.txt and dx1_decending.txt are the results we used for the paper.


