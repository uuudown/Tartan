#!/bin/bash
#BSUB -P #PROJECTNUMBER
#BSUB -W 10
#BSUB -nnodes 2
module load gcc/5.4.0
module load hdf5
module load cuda/8.0.54
module load spectrum-mpi
source $OLCF_SPECTRUM_MPI_ROOT/jsm_pmix/bin/export_smpi_env -gpu
/usr/bin/time -f 'ExE_Time: %e' jsrun -n2 -a1 -g1 -c1 -r1 ./run_2g_weak.sh
/usr/bin/time -f 'ExE_Time: %e' jsrun -n2 -a1 -g1 -c1 -r1 ./run_2g_weak.sh
/usr/bin/time -f 'ExE_Time: %e' jsrun -n2 -a1 -g1 -c1 -r1 ./run_2g_weak.sh
/usr/bin/time -f 'ExE_Time: %e' jsrun -n2 -a1 -g1 -c1 -r1 ./run_2g_weak.sh
/usr/bin/time -f 'ExE_Time: %e' jsrun -n2 -a1 -g1 -c1 -r1 ./run_2g_weak.sh
