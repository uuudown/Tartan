==========================================================================================
 This microbenchmark belongs to the Tartan Benchmark Suite, which contains microbenchmarks
 and benchmark applications for evaluating multi-GPUs and multi-GPU interconnect. Please see
 our IISWC-18 paper titled "Tartan: Evaluating Modern GPU Interconnects via a Multi-GPU 
 Benchmark Suite" for detail.

        Version:  1.0
        Created:  01/24/2018 04:13:15 PM

         Author:  Ang Li, PNNL
        Website:  http://www.angliphd.com  
==========================================================================================

##Description:

This microbenchmark is to measure the latency & uni/bi-directional 
bandwidth for PCI-e and NVLink with different message size so as to see when the interconnect
got saturated. The code is modified from the p2pBandwidthLatencyTest app in NVIDIA CUDA-SDK. 

##Compile: 

Please modify shared.mk according to your machine environment, then compile using
```shell
    $ make clean
    $ make
```

##Run: 
```shell
    $ ./p2pPacket 
```

##Configuration: 

This microbenchmark measures the latency, uni/bi-direction bandwidth 
for P2P communication among GPUs in a single node (i.e., scale-up) via PCI-e and NVLink 
with increased data size. For NVLink, we only measure the nodes that are directly 
interconnected (which means we do not perform manually routing). However, you may still
observe the bandwidth NUMA effect on NVLink-V2 in V100 DGX-1.

##Results:

dgx-1_packet.txt is the result we used for the paper.


