#!/bin/bash
#BSUB -P PROJNUM
#BSUB -W 4
#BSUB -nnodes 2

module load cuda/8.0.54
module load spectrum-mpi

export LD_LIBRARY_PATH=/sw/summitdev/cuda/8.0.54/lib64/:$LD_LIBRARY_PATH
#source $OLCF_SPECTRUM_MPI_ROOT/jsm_pmix/bin/export_smpi_env -gpu

jsrun -n2 -a1 -g1 -c1 -r1 ./mpibw 0 0 20 268435456 device
