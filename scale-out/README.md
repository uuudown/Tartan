###========================================================================
####         Author:  Ang Li, PNNL
####        Website:  http://www.angliphd.com  
####        Created:  03/19/2018 03:44:40 PM, Richland, WA, USA.
###========================================================================

##Introduction:
 This directory contains the 7 multi-GPU benchmarks for inter-node scale-out on InfiniBand with GPUDirect-RDMA. This package have been tested on ORNL's SummitDev supercomputer. "scale-out" has been tested on DGX-1.

##Config:
 Set env in "shared.mk". You may need to install NCCL and libconfig library.

##Run:
 Execute the python script: 
 ```
   $ python run\_scale\_out.py
 ```
 or enter into each app dir, make, and run:
 ```  
   $ cd scale-out/$app$/
   $ make
   $ chmod +x run.sh
   $ ./run.sh
```

##Scaling Test:
  Strong scaling:
  ```
   $ ./run\_1g\_strong.sh
   $ ./run\_2g\_strong.sh
   $ ./run\_4g\_strong.sh
   $ ./run\_8g\_strong.sh
```

  Weak scaling: 
```
   $ ./run\_1g\_weak.sh
   $ ./run\_2g\_weak.sh
   $ ./run\_4g\_weak.sh
   $ ./run\_8g\_weak.sh
```
##Acknowledge:
 Our scale-out test is conducted on SummitDev of Oak-Ridge National Laboratory (ORNL). The
 research and evaluation is supported by Exascale Computing Project (17-SC-20-SC), a joint 
 project of the U.S. Department of Energy's Office of Science and National Nuclear Security
 Administration, under the Application Assessment Project.


##File Description: 
```shell
            shared.mk: Overall configuration file for all Makefile. Please config your env here.

               common: Commonly-shared header and/or dependent third-party library

            scale-out: Baseline; Unpinned-main memory as inter-node MPI communication buffer 

     scale-out-pinned: Pinned-main memory as inter-node MPI communication buffer

       scale-out-rdma: Directly use GPU device memory as inter-node MPI communication buffer

      run_scale_up.py: Python script for testing

```

##Applications:
```shell
             B2rEqwp: 3D earthquake wave-propogation model simulation using finite difference method

           Diffusion: 3D heat equation and inviscid burgers' equation

              Lulesh: Livermore unstructured lagrangian explicit shock hydrodynamics

                CoMD: A reference implmentation of classical molecular dynamics algorithms 

             Prbench: Page rank computation

                 Hit: Simulating homogeneous isotropic turbulence by solving navier-stokes equations

              Matvec: Matrix multiplication via mpi-scatter, broadcast and gather

```


## Note:

    Please see our IISWC-18 paper "Tartan: Evaluating Modern GPU Interconnect via a 
      Multi-GPU Benchmark Suite" for detail.

    Please cite our paper if you find this package useful.  
    
    All Rights Reserved.
