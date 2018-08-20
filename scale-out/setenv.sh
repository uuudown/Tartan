#!/bin/bash

module load cuda/8.0.54
module load spectrum-mpi
module load hdf5

export LD_LIBRARY_PATH=/ccs/home/angli/tartan/Tartan/common/lib/:/ccs/home/angli/tartan/Tartan/common/libconfig-1.4.9/.libs/:$LD_LIBRARY_PATH
