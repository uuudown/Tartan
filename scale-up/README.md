###========================================================================
####         Author:  Ang Li, PNNL
####        Website:  http://www.angliphd.com  
####        Created:  03/19/2018 02:44:40 PM, Richland, WA, USA.
###========================================================================

- Introduction:
 This directory contains the 7 multi-GPU benchmarks for intr-node scale-up with PCI-e 
 and NVLink-V1/V2 interconnect.

- Config:
 Set env in "shared.mk". You need to install NCCL library before building the NVLink version.

- Run:
 Execute the python script: 
 ```
   $ python run_scale_up.py 
 ```
 or enter into each app dir, make, and run:
 ```
   $ cd scale-up/$app$/
   $ make
   $ chmod +x run.sh
   $ ./run.sh
 ```
##Scaling Test:
  Strong scaling:
  ```
   $ ./run_1g_strong.sh
   $ ./run_2g_strong.sh
   $ ./run_4g_strong.sh
   $ ./run_8g_strong.sh
  ```
  
  Weak scaling: 
  ```
   $ ./run_1g_weak.sh
   $ ./run_2g_weak.sh
   $ ./run_4g_weak.sh
   $ ./run_8g_weak.sh
  ```
##File Description: 
```shell
            shared.mk: Overall configuration file for all Makefile. Please config your env here.

               common: Commonly-shared header and/or dependent third-party library

             scale-up: Benchmarks based on PCI-e interconnect

      scale-up-nvlink: Benchmarks based on NVLink interconnect

      run_scale_up.py: Python script for testing

```

##Applications:
```shell
            ConvNet2: Convolution neural networks via data, model and hybrid parallelism

            Cusimann: global optmization via parallel simulated annealing algorithm.

                 GMM: multivariate data clustering via Gaussian mixture model

              Kmeans: Kmeans-Clustering for double-precision data.

          MonteCarlo: Monte-Carlo option pricing from CUDA-SDK

              Planar: Depth-First-Search (DFS) and backtracing to solve Planar Langford's Sequence

              Trueke: Exchange Monte-Carlo for 3D random field Ising model

```


## Note:

    Please see our IISWC-18 paper "Tartan: Evaluating Modern GPU Interconnect via a 
      Multi-GPU Benchmark Suite" for detail.

    Please cite our paper if you find this package useful.  
    
