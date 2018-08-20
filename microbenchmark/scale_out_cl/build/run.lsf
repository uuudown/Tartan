#!/bin/bash
#BSUB -P YOUR_PROJECT_ID
#BSUB -W 2
#BSUB -nnodes 4

module load cuda/8.0.54
module load spectrum-mpi

source $OLCF_SPECTRUM_MPI_ROOT/jsm_pmix/bin/export_smpi_env -gpu
export LD_LIBRARY_PATH=/sw/summitdev/cuda/8.0.54/lib64/:$LD_LIBRARY_PATH

jsrun -n4 -a1 -g1 -c1 -r1 ./broadcast_perf -b 16M -e 16M -f 2 -g 1 -c 0
