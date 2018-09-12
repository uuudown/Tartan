####========================================================================
####         Author:  Ang Li, PNNL
####        Website:  http://www.angliphd.com  
####        Created:  03/19/2018 04:12:50 PM, Richland, WA, USA.
####========================================================================

##Introduction:

 Tartan is a multi-GPU benchmark suite. It is proposed to evaluate modern GPU interconnect 
 in our IISWC-18 paper **"Tartan: Evaluating Modern GPU Interconnect via a Multi-GPU Benchmark
 Suite"**. Please see our paper (as included) for more details.

 Tartan contains three sub-directories:

   **microbenchmark** The microbenmarking routines to measure the startup latency, sustainable 
                    uni-/bi-direction bandwidth, bandwidth with message size, etc. for 
                    Peer-to-Peer (P2P) and and Collective Communication (CL) on PCI-e, 
                    NVLink-V1, NVLink-V2 and InfiniBand.

  **scale-up** Applications for intra-node scale-up (i.e., single node with multiple GPUs)

 **scale-out** Applications for inter-node scale-out (i.e., GPU-accelerated multi-node system)

##Acknowledge:

The applications in Tartan were modified from their original implementations. It is an joint effort of the community. Here we list the source of the original design.

**Microbenchmark:**

- Scale-up P2P: NVIDIA, “CUDA SDK Code Samples”, 2015.
- Scale-up Collective: NVIDIA, “NCCL Tests”, http://github.com/NVIDIA/nccl-tests
- Scale-out P2P: “MPI-GPU-BW”, https://github.com/frroberts/MPI-GPU-BW
- Scale-out Collective: NVIDIA, “NCCL Tests”, http://github.com/NVIDIA/nccl-tests

**Scale-up:**
- ConvNet2: Google, “High-Performance C++/CUDA Implementation of Convolutional Neural Networks Version-2”, https://github.com/akrizhevsky/cuda-convnet2
- Cusimann: A.M.F. Ferreiro, J.A.G. Rodrıguez, J.G.L. Salas, and C.V. Cendon, “CUSIMANN: An optimized simulated annealing software for GPUs”, https://github.com/palmalcheg/cusimann
- GMM: A.D. Pangborn, “Expectation Maximization with a Gaussian Mixture Model using CUDA”, https://github.com/corv/cuda-gmm-multigpu
- Kmeans: NVIDIA, “Kmeans Clustering with Multi-GPU Capabilities”, https://github.com/NVIDIA/kmeans
- MonteCarlo: NVIDIA, “CUDA SDK Code Samples,” 2015.
- Planar: B. Dimitrov, “Multi-GPU Code to Count all PLANAR Langford Sequences”, https://github.com/boris-dimitrov/z4_planar_langford_multigpu
- Trueke: C. Navarro, “Multi-GPU Exchange Monte Carlo for 3D Random Field Ising Model”, https://github.com/crinavar/trueke

**Scale-out:**

- B2rEqwp: Z. Liu, “Efficient Large-scale Parallel Stencil Computation on Multi-Core and Multi-GPU Accelerated Clusters”, https://github.com/lzhengchun/b2r.
- Diffusion: M. A. Diaz, “Multi-GPU (CUDA-MPI) baseline implementation of Heat Equation and the inviscid Burgers’ equation”, https://github.com/wme7/MultiGPU_AdvectionDiffusion.
- Lulesh: LLNL, “Livermore Unstructured Lagrangian Explicit Shock Hydrodynamics”, https://codesign.llnl.gov/lulesh.php
- CoMD: NVIDIA, “GPU implementation of classical molecular dynamics proxy application”, https://github.com/NVIDIA/CoMD-CUDA
- Prbench: NVIDIA, “A CUDA implementation of the PageRank Pipeline Benchmark”, https://github.com/NVIDIA/PRBench
- HIT: M.V.Martin, “HIT: a parallel GPGPU code to simulate Homogeneous Isotropic Turbulence”, https://github.com/albertovelam/HIT_MPI.
- Mavtec: T. Agarwal, “Multi-GPU Matrix Multiplication using CUDA and MPI”, https://github.com/tejaswiagarwal/multigpumatmul

The Tartan Benchmark Suite and the evaluation research was supported by the U.S. DOE Office of 
Science, Office of Advanced Scientific Computing Research, under award 66150: "CENATE - Center
for Advanced Architecture Evaluation". The Pacific Northwest National Laboratory is operated
by Battelle for the U.S. Department of Energy under contract DE-AC05-76RL01830. 
This research was also supported by the Exascale Computing Project (17-SC-20-SC), 
a collaborative effort of the U.S. DOE Office of Science and National Nuclear Security 
Administration. Part of the computing resources is from the Oak Ridge Leadership Computing 
Facility. The Oak Ridge National Laboratory is supported by the Office of Science of 
the U.S. Department of Energy under Contract No. DE-AC05-00OR22725.

##MIT License

This benchmark suite is modified from individual applications. Please refer to each application's license requirement.

Copyright 2018 Ang Li, PNNL.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software 
and associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute, 
sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or 
substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR 
PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE 
FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

