CoMD-CUDA
====

This is GPU implementation of CoMD 1.1 proxy application. The GPU code supports both cell-based and neighbor-lists methods for molecular dynamics and includes various parallelization strategies for both. Distributed multi-GPU implementation is supported as well by using one GPU per MPI rank.

Build instructions
------------------------

Use src-mpi/Makefile.EAM for EAM forces and src-mpi/Makefile.NAMD for LJ forces. Modify Makefiles according to your environment, then make.

Requirements:
* CUDA toolkit (6.5, 7.0 and later versions are preferred)
* MPI library (if building with DO_MPI = ON)

Notes:
* When building with MPI you need to update MPI_INCLUDE variable in Makefile.
* You might need to do make clean/make after you modify header files.

Sample run commands
------------------------

Single-GPU run with EAM forces and 49x49x49 grid size using default method (cell-based, thread per atom):
~~~ sh
$ ./bin/CoMD-cuda-mpi -e -x 49 -y 49 -z 49
~~~

Multi-GPU run with 2 GPUs, EAM forces and 98x49x49 overall grid size divided between GPUs along X dimension:
~~~ sh
$ mpirun -np 2 ./bin/CoMD-cuda-mpi -e -x 98 -y 49 -z 49 -i 2
~~~

Multi-GPU run with 2 GPUs, EAM forces, 98x49x49 overall grid size and neighbor lists method:
~~~ sh
$ mpirun -np 2 ./bin/CoMD-cuda-mpi -e -x 98 -y 49 -z 49 -i 2 -m thread_atom_nl
~~~

Best single-GPU configuration using warp per atom approach with neighbor lists:
~~~ sh
$ ./bin/CoMD-cuda-mpi -e -x 49 -y 49 -z 49 -m warp_atom_nl
~~~

To view all available options please check:
* original CoMD Doxygen documentation at <a href="http://exmatex.github.io/CoMD/doxygen-mpi/index.html">exmatex.github.io/CoMD/doxygen-mpi/index.html</a>
* for any GPU-only options in <a href="https://github.com/nsakharnykh/CoMD-CUDA/blob/master/src-mpi/mycommand.c">mycommand.c</a>

Output explanation
------------------------

Below is a sample output which you can use for the validation of the results. When modifying the code please check that all energies and # of atoms remain the same as in the original code.

~~~ sh
#                                                                                         Performance
#  Loop   Time(fs)       Total Energy   Potential Energy     Kinetic Energy  Temperature   (us/atom)     # Atoms
      0       0.00    -3.460523233086    -3.538079224686     0.077555991600     600.0000     0.0000       470596
     10      10.00    -3.460522622766    -3.529929454580     0.069406831814     536.9553     0.0707       470596
     20      20.00    -3.460524220490    -3.509740515517     0.049216295027     380.7543     0.0711       470596
     30      30.00    -3.460527806915    -3.488529040692     0.028001233777     216.6272     0.0660       470596
     40      40.00    -3.460532196608    -3.477523402265     0.016991205657     131.4498     0.0662       470596
     50      50.00    -3.460536497383    -3.479780609997     0.019244112614     148.8791     0.0709       470596
     60      60.00    -3.460538213894    -3.488976046432     0.028437832538     220.0049     0.0665       470596
     70      70.00    -3.460536800219    -3.496688002423     0.036151202204     279.6782     0.0663       470596
     80      80.00    -3.460533977439    -3.498984084647     0.038450107208     297.4633     0.0713       470596
     90      90.00    -3.460531463100    -3.497356126200     0.036824663100     284.8883     0.0664       470596
    100     100.00    -3.460530040624    -3.495833910540     0.035303869916     273.1230     0.0666       470596
~~~

Performance metric for CoMD is atoms/us (processed atoms per time), which is printed at the end of the execution. 

~~~ sh
---------------------------------------------------
 Average atom rate:             14.66 atoms/us
---------------------------------------------------
~~~

Results and publications
------------------------

EAM code can achieve 34 atoms/us on NVIDIA Tesla K40m with the boost clocks.

GTC 2014: <a href="http://on-demand-gtc.gputechconf.com/gtc-quicklink/hzgVvB">Optimizing CoMD: A Molecular Dynamics Proxy Application Study</a>

CoMD version 1.1
------------------------

CoMD is a reference implementation of typical classical molecular
dynamics algorithms and workloads.  It is created and maintained by
ExMatEx: Exascale Co-Design Center for Materials in Extreme Environments
(<a href="http://exmatex.org">exmatex.org</a>).  The
code is intended to serve as a vehicle for co-design by allowing
others to extend and/or reimplement it as needed to test performance of 
new architectures, programming models, etc.

Original CoMD code is available at <a href="https://github.com/exmatex/CoMD">github.com/exmatex/CoMD</a>.

To view the generated Doxygen documentation for CoMD, please visit
<a href="http://exmatex.github.io/CoMD/doxygen-mpi/index.html">exmatex.github.io/CoMD/doxygen-mpi/index.html</a>.

To contact the developers of CoMD send email to exmatex-comd@llnl.gov.
