##
 # =====================================================================================
 #
 #       Filename:  shared.mk
 #
 #    Description:  This is the commonly shared configuration file for all Makefile in 
 #                  directory. Please change the items according to your machine.
 #
 #        Version:  1.0
 #        Created:  03/19/2018 03:43:25 PM
 #       Revision:  none
 #       Compiler:  GNU-Make
 #
 #         Author:  Ang Li, PNNL
 #        Website:  http://www.angliphd.com  
 #
 #       Please cite our IISWC-18 paper "Tartan: Evaluating Modern GPU Interconnect 
 #          via a Multi-GPU Benchmark Suite"
 #
 # =====================================================================================
##


SHELL = /bin/bash

ARCH=sm_60
# CUDA toolkit installation path
CUDA_DIR = /sw/summitdev/cuda/8.0.54/

# CUDA SDK installation path
SDK_DIR = /sw/summitdevcuda/8.0.54/samples/

# CUDA toolkit libraries
LIB_DIR = $(CUDA_DIR)/lib64

# MPI 
MPI_DIR = /autofs/nccs-svm1_sw/summitdev/.swci/1-compute/opt/spack/20171006/linux-rhel7-ppc64le/xl-20170914-beta/spectrum-mpi-10.1.0.4-20170915-nmlgpsufnxxal2wv64hh7zfisabr56ry/

# compiler
CC = gcc
CC_FLAGS = -O3 

MPICC = mpicc
MPICC_FLAGS = -O5

MPICXX = mpic++

# CUDA compiler
NVCC = $(CUDA_DIR)/bin/nvcc
NVCC_FLAGS = -arch=$(ARCH)  -O3 
# Link
NVCC_INCLUDE = -I. -I$(CUDA_DIR)/include -I$(SDK_DIR)/C/common/inc -I../../common/inc/ -I$(SDK_DIR)/shared/inc -I$(MPI_DIR)/include -I/ccs/home/angli/tartan/Collective/nccl_2.0/include/ -I../../common/libconfig-1.4.9/
NVCC_LIB =-lcuda -lmpi_ibm # -lnccl
NVCC_LIB_PATH = -L. -L$(SDK_DIR)/C/lib -L$(LIB_DIR)/ -L$(SDK_DIR)/shared/lib -L$(MPI_DIR)/lib -L/usr/lib/ -L/usr/lib64  -L/ccs/home/angli/tartan/Collective/nccl_2.0/lib -L../../common/libconfig-1.4.9/.libs/


LINK_FLAG = $(NVCC_INCLUDE) $(NVCC_LIB_PATH) $(NVCC_LIB) -lstdc++ -lm
