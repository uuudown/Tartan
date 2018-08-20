Expectation Maximization with a Gaussian Mixture Model using CUDA

CUDA Data Clustering

Developed by Andrew D. Pangborn:

2009 - 2010

Department of Computer Engineering
Kate Gleason College of Engineering
Rochester Institute of Technology
Rochester, New York

=== About ===
This software does multivariate data clustering using Expectation Maximization with a Gaussian mixture model algorithms using NVIDIA's CUDA framework. A CUDA-capable GPU is required to run the programs.
For a list of CUDA-capable GPUs please consult the NVIDIA documentation available here: http://www.nvidia.com/object/cuda_gpus.html

The algorithm has two separate versions, one for a single workstation (which supports multiple GPUs), and another that is designed for HPC clusters with a GPU on each processing node. The current MPI implementation was developed and tested on the "Lincoln" cluster (part of NCSA) on the TeraGrid using MVAPICH2 compiled with the GNU toolchain.

The program achieves nearly 2 orders of magnitude (100x) speedup using a NVIDIA GTX260, compared to an optimized single threaded version on a modern Intel CPU.

For additional information about the algorithms, the implementations, and the performance results please consult my thesis documentation.
http://apangborn.com/thesis/

=== Software Installation ===

1) Install NVIDIA CUDA Driver (available from http://www.nvidia.com/object/cuda_get.html)
2) Install NVIDIA CUDA Toolkit (available from http://www.nvidia.com/object/cuda_get.html)
3) Install NVIDIA CUDA SDK (available from http://www.nvidia.com/object/cuda_get.html)
Linux/Mac users, pay attention to the instructions at the end of the SDK installer. You must add a couple entries to the C_INCLUDE_PATH and LD_LIBRARY_PATH environment variables

4) Checkout source code from SVN repository into your NVIDIA CUDA SDK folder
For CUDA 2.3 the default for Linux is ~/NVIDIA_GPU_Computing_SDK

Single GPU or multiples GPUs on one system (including single cards with 2 GPU cores like the GTX 295)
    $ svn checkout http://cyberaide.googlecode.com/svn/trunk/project/biostatistics/projects/mixtureModelMultiGPU/ ~/NVIDIA_GPU_Computing_SDK/C/src/gmm

With MPI support for GPU clusters. The Makefile will need to updated to compile correctly against your MPI distribution
    $ svn checkout http://cyberaide.googlecode.com/svn/trunk/project/biostatistics/projects/mixtureModelMPI/ ~/NVIDIA_GPU_Computing_SDK/C/src/gmm

=== Compiling the CUDA Code (Linux) ===

    $ cd ~/NVIDIA_GPU_Computing_SDK/C/src/gmm
    $ make
    
This utilizes the common Makefile (common.mk) in the NVIDIA SDK, so the project must be installed as a subfolder of the "<sdk root>/C/src/" folder to work properly. If future versoins of the CUDA SDK change the directory layout (again), following these instructions verbatim will probably not work.

Some parameters must be adjusted at compile-time. You can edit "gaussian.h" and then recompile to change these features

Parameters of note:

MAX_ITERS - defines the maximum number of iterations (number of E-step + M-step iterations). If reached, iterating will stop and the current solution will be output. The epsilon value used for converge is computed based on the input parameter, but it may need adjustment depending on the nature of your data. "epsilon" can be found in "gaussian.cu"
COVARIANCE_DYNAMIC_RANGE - the program adds (Average_variance/COVARIANCE_DYNAMIC_RANGE) to the diagonal of the covariance matrices. This helps prevent the matrices from becoming singular (un-invertable). If you see "NaN" values appearing in your output, try reduces this value, but it may introduce a little bit more error into your result.
ENABLE_DEBUG - prints out a bunch of extra debugging information
ENABLE_PRINT - prints some basic program status whlie running and the final clustering parameters. Typically only disabled for doing performance tests.
ENABLE_OUTPUT - outputs the gaussian mixture parameters and the membership probabilities for every data point


=== Compiling the CUDA Code (Windows) ===
Windows users should be able to use the included ".sln" Visual Studio solution file. Ensure that the $(CUDA_INC_PATH) and $(CUDA_LIB_PATH) environment variables are set properly.
It was developed and tested on Windows 7 with Visual Studio 2008 Professional.

=== Running the Code ===
Binaries on a linux system are placed in ~/NVIDIA_GPU_Computing_SDK/C/bin/linux/release

Usage: ../../bin/linux/release/gmm num_clusters infile outfile [target_num_clusters]
         num_clusters: The number of starting clusters
         infile: ASCII space-delimited FCS data file
         outfile: Clustering results output file
         target_num_clusters: A desired number of clusters. Must be less than or equal to num_clusters

I usually find it useful to create a symbolic link to the bin folder in my source folder

    $ ln -s ../../bin/linux/release bin

An example from the source folder..
    $ ./bin/gmm 100 mydata.csv mydata
    
This will produce "mydata.summary" and "mydata.results". The former contains the gaussian mixture parameters, and the latter contains the data and the cluster membership probabilities for each data point. The data values and the probabilities are separated by the tab, and the individual dimensions are separated by commas.

Example of output file 3 data point with 4 dimensions and 2 clusters:
1.0,2.0,3.0,4.0 0.01,0.99
4.0,3.0,2.0,1.0 0.88,0.01
2.5,2.5,2.5,2.5 0.5,0.5
    

=== Acknowledgement ===
Thanks to Dr. Gregor von Laszewski, Dr. James S. Cavenaugh, Dr. Muhammad Shaaban, Dr. Roy Melton, and Jeremy Espenshade for their contributions to this project. 

The theory and sequential code for Gaussian mixture model application was based on the "cluster" application by Charles Bouman from the University of Purdue.
https://engineering.purdue.edu/~bouman/software/cluster/

Thanks to the TeraGrid and the National Center for Supercomputing Applications (NCSA) for their help and resources under project grant number TG-MIP050001.

=== License ===
Copyright (c) 2010, Andrew D. Pangborn
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

