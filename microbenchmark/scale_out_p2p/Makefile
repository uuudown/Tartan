##
 # =====================================================================================
 #
 #       Filename:  Makefile
 #
 #    Description:  This microbenchmark is to obtain the latency, bandwidth and efficiency
 #                  for P2P InfiniBand (IB) communication among GPU-integrated HPC nodes. 
 #                  The objective is to measure the difference when GPU-Direct RDMA is 
 #                  setting and when pinned host memory is utilized. The test platform
 #                  is SummitDev supercomputer in Oak Ridge National Laboratory (ORNL).
 #
 #        Version:  1.0
 #        Created:  01/24/2018 04:13:15 PM
 #       Revision:  none
 #       Compiler:  GNU-Make
 #
 #         Author:  Ang Li, PNNL
 #        Website:  http://www.angliphd.com  
 #
 # =====================================================================================
##

CC = mpicxx
FLAGS = -O3 -std=c++11 # -lmpi -I/usr/include/openmpi-x86_64 -L/opt/mpich3/gnu/lib 

mpibw: main.cpp
	$(CC) $(FLAGS) $^ -o  $@ -lcudart -o mpibw -I/sw/summitdev/cuda/8.0.54/include/ -L/sw/summitdev/cuda/8.0.54/lib64/

clean:
	rm -f mpibw
	
