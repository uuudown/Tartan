##
 # =====================================================================================
 #
 #       Filename:  shared.mk
 #
 #    Description:  This is the commonly shared configuration file for all Makefile in 
 #                  directory. Please change the items according to your machine.
 #
 #        Version:  1.0
 #        Created:  03/19/2018 02:19:15 PM
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
CUDA_DIR = /usr/local/cuda-9.1/

# CUDA driver path
CUDA_DRIVER_DIR = /usr/lib/nvidia-390/

# CUDA SDK installation path
SDK_DIR = $(HOME)/NVIDIA_GPU_Computing_SDK/

# CUDA toolkit libraries
LIB_DIR = $(CUDA_DIR)/lib64

# MPI 
MPI_DIR = $(HOME)/opt/miniconda2/pkgs/mpich2-1.4.1p1-0/

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
NVCC_INCLUDE = -I. -I$(CUDA_DIR)/include -I$(SDK_DIR)/C/common/inc -I../../common/inc/ -I$(SDK_DIR)/shared/inc -I$(MPI_DIR)/include -I/home/lian599/include/
NVCC_LIB = -lcutil_x86_64 -lcuda -lmpich -lmpl -lnccl
NVCC_LIB_PATH = -L. -L$(SDK_DIR)/C/lib -L$(LIB_DIR)/ -L$(SDK_DIR)/shared/lib -L$(MPI_DIR)/lib -L/home/lian599/lib/ -L/usr/lib/ -L/usr/lib64 


LINK_FLAG = $(NVCC_INCLUDE) $(NVCC_LIB_PATH) $(NVCC_LIB) -lstdc++ -lm
