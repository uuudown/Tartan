# trueke: multi-GPU Monte Carlo for the 3D RFIM

# (1) Hardware requirements:
- A CUDA capable GPU, we recommend Kepler+
- 4GB of RAM
- Multi-core X86_64 CPU




# (2) Software requirements:
- Linux OS
- GCC.
- Nvidia CUDA runtime library
- Nvidia nvcc compiler
- [optional] Nvidia nvlm (to query the device).
- OpenMP 4.0 Implementation






# (3) Check Makefile and CUDA instalation. 
Make sure the bin, inc and lib paths to the corresponding ones.







# (4) compilation (edit Makefile if necessary)
 - make clean
 - make






# (5) how to run
./bin/trueke -l \<L\> \<R\> -t \<T\> \<dT\> -a \<tri\> <ins\> \<pts\> \<ms\> -h \<h\> -s \<pts\> \<mz\> \<eq\> \<ms\> \<meas\> \<per\> -br \<b\> \<r\> -z \<seed\> -g \<x\>




# (6) parameters
- <b>Lattice (-l)</b>: size \<L\>, \<R\> replicas.
- <b>Temperature (-t)</b>: high-temp \<T\>, delta \<dT\>.
- <b>Adaptive (-a)</b>: \<tri\> trials, \<ins\> inserts/trial, \<pts\> exchange steps, \<ms\> sweeps per exchange.
- <b>External Field (-h)</b>: magnetic field strength \<h\> \(tipically, 0 \< h \< 3\).
- <b>Simulation (-s)</b>: \<pts\> ex, meas at \<mz\>, equil \<eq\> sweeps, \<ms\> sweeps/ex, \<per\> ex/\<meas\>
- <b>Repetition (-br)</b>: \<b\> blocks of \<ms\>, \<r\> disorder realizations.
- <b>Seed (-z)</b>: use \<seed\> as base seed for the PRNGs.
- <b>Multi-GPU (-g)</b>: use \<x\> GPUs.






# (4) Example execution using two GPUs 
- ./bin/trueke -l 64 11 -t 4.7 0.1 -a 58 2 2000 10 -h 1.0 -s 5000 3000 100 5 1 1 -br 1 2000 -z 7919 -g 2
