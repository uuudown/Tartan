# profile heat3d
mpirun -np 2 nvprof -o Diffusion3d.%q{OMPI_COMM_WORLD_RANK}.nvprof ./Diffusion3d.run 256 256 256 100 64 4 1
