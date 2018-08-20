#!/bin/bash
#
# Warning: You must adapt this script to your runtime environment.

nvcc -Xcompiler -fopenmp -arch=sm_60 -I$HOME/NVIDIA_GPU_Computing_SDK/C/common/inc/ -I../include  -I$HOME/install/include minimize$1.cu -L$HOME/NVIDIA_GPU_Computing_SDK/C/lib -lcutil_x86_64 -lcurand -L$HOME/install/lib -lnlopt -lm
