###========================================================================
###         Author:  Ang Li, PNNL
###        Website:  http://www.angliphd.com  
###        Created:  03/19/2018 04:12:50 PM, Richland, WA, USA.
###========================================================================

##Introduction:

 Tartan is a multi-GPU benchmark suite. It is proposed to evaluate modern GPU interconnect 
 in our IISWC-18 paper "Tartan: Evaluating Modern GPU Interconnect via a Multi-GPU Benchmark
 Suite". Please see our paper for more details.

 Tartan contains three sub-directories:

   *microbenchmark* The microbenmarking routines to measure the startup latency, sustainable 
                    uni-/bi-direction bandwidth, bandwidth with message size, etc. for 
                    Peer-to-Peer (P2P) and and Collective Communication (CL) on PCI-e, 
                    NVLink-V1, NVLink-V2 and InfiniBand.

  *scale-up* Applications for intra-node scale-up (i.e., single node with multiple GPUs)

 *scale-out* Applications for inter-node scale-out (i.e., GPU-accelerated multi-node system)

##Acknowledge:

The Tartan Benchmark Suite and the evaluation research was supported by the U.S. DOE Office of 
Science, Office of Advanced Scientific Computing Research, under award 66150: "CENATE - Center
for Advanced Architecture Evaluation". The Pacific Northwest National Laboratory is operated
by Battelle for the U.S. Department of Energy under contract DE-AC05-76RL01830. 
This research was also supported by the Exascale Computing Project (17-SC-20-SC), 
a collaborative effort of the U.S. DOE Office of Science and National Nuclear Security 
Administration. Part of the computing resources is from the Oak Ridge Leadership Computing 
Facility. The Oak Ridge National Laboratory is supported by the Office of Science of 
the U.S. Department of Energy under Contract No. DE-AC05-00OR22725.


##MIT License

Copyright 2018 Ang Li, PNNL.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software 
and associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute, 
sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or 
substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR 
PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE 
FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

