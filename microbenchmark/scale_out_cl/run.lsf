#!/bin/bash
#BSUB -P CSC249ADCD502 
#BSUB -W 1
#BSUB -nnodes 2

module purge
module load gcc/5.4.0
module load cuda/8.0.54
module load spectrum-mpi

source $OLCF_SPECTRUM_MPI_ROOT/jsm_pmix/bin/export_smpi_env -gpu
export LD_LIBRARY_PATH=/sw/summitdev/cuda/8.0.54/lib64/:$LD_LIBRARY_PATH
#LD_LIBRARY_PATH=/sw/summitdev/cuda/8.0.61-1/lib64/:/autofs/nccs-svm1_sw/summitdev/.swci/1-compute/opt/spack/20171006/linux-rhel7-ppc64le/xl-20170914-beta/spectrum-mpi-10.1.0.4-20170915-nmlgpsufnxxal2wv64hh7zfisabr56ry/lib:/ccs/home/angli/tartan/Collective/nccl_2.0/lib/

jsrun -n1 -a1 -g1 -c1 -r1 ./broadcast_perf -b 16M -e 16M -f 2 -g 1 -c 0
#jsrun -n1 -a1 -g1 -c1 -r1 ./all_reduce_perf -b 8 -e 128M -f 2 -g 1 -c 0
