###========================================================================
###         Author:  Ang Li, PNNL
###        Website:  http://www.angliphd.com  
###        Created:  01/24/2018 05:40:38 PM, Richland, WA, USA.
###========================================================================

##Introduction:

 This microbenchmark package is to measure the performance metrics, including latency, 
 uni/bi-direction bandwidth, routing, NUMA effect, communication efficiency, etc for 
 both P2P and Collective communications on modern multi-GPU interconnect, including PCI-e
 NVLink-V1, NVLink-V2 for single-node scale-up, and IB with/without GPUDirect-RDMA for 
 multi-nodes scale-out.  

##File Description: 
```shell
            shared.mk: Overall configuration file for all Makefile. Please config your env here.

               common: Commonly-shared header

    scale_up_p2p_test: Latency, uni/bi-direction bandwidth benchmarking for P2P 
                       communication on single node multi-GPU platforms, such as NVIDIA DGX-1     

  scale_up_p2p_packet: Communication efficiency benchmarking for P2P on single-node multi-GPUs

          scale_up_cl: Latency, bandwidth, efficiency benchmarking for Collective communication
                       on single node multi-GPU platforms, such as DGX-1.

        scale_out_p2p: Latency, bandwidth and efficiency benchmarking for P2P communication on 
                       GPU integrated HPC nodes (multi-nodes). 

         scale_out_cl: Latency, bandwidth and efficiency benchmarking for CL communication on 
                       GPU integrated HPC nodes.
```

## Note:

    All the evaluations are discussed and analyzed in detail in our IISWC-18 paper titled  
      "Tartan: Evaluating Modern GPU Interconnect via a Multi-GPU Benchmark Suite"
      Please cite our paper if you find this package useful.  
    
    
    All Rights Reserved.
