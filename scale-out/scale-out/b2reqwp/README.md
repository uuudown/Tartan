# Introduction
Source code for article 'Efficient Large-scale Parallel Stencil Computation on Multi-Core and Multi-GPU Accelerated Clusters'. 

This repo host the framework template that updates/synchronizes arbitrarily defined subdomain boundaries, and the benckmark models to study the effectiveness of our B2R latency hiding scheme. 

# Main
 
The main entry is in b2r.cpp. Configurations are in b2r.hpp.

# Data movement way, in b2r.cpp 

1. use function *boundary\_updating\_direct* will transfer data directly from device memory to device memory via MPI-direct (requires hardware support).

2. use function *boundary\_updating* will pack/unpack in device but copy the halo to host to send, the same for receiving (does not need hardware support).

# Benchmark model

## Earthquake wave propogation model
eqwp.cu and eqwp.hpp

## Game of life
gol-3D.cu and gol-3D.hpp
## Heat diffusion model

CPU codes are in heat-3D.cpp, cuda implementation codes are in heat-3D.cu

There are four implementations (fdm_heat_diffuse_delta1-4) for this model with different degree of numerical accuracy. 

You need to change model entry function accordingly in the main function in b2r.cc.

# Run
mpirun -np N ./mpi.out (mpi.out -> mpi-gou.out) R Bx By Bz

where, R is the B2R parameter R that must fullfil the condition: 1 <= R <= (R*Delta)/2, Bxyz is the block size to be used to split the simulation environment. all of which should be perfectly devisiable with the corresponding dimension size of the environemnt. The total number of process, N, should be the same as total number of block, i.e., N = Bx * By * Bz.