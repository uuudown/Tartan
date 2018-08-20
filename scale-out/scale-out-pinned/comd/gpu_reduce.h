/*************************************************************************
 * Copyright (c) 2013, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 ************************************************************************/

#ifndef __GPU_REDUCE_H_
#define __GPU_REDUCE_H_

__global__ void ReduceEnergy(SimGpu sim, real_t *e_pot, real_t *e_kin)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  int iAtom = sim.a_list.atoms[tid];
  int iBox = sim.a_list.cells[tid];

  int iOff = iBox * MAXATOMS + iAtom;

  real_t ep = 0;
  real_t ek = 0; 
  if (tid < sim.a_list.n) {
    int iSpecies = sim.atoms.iSpecies[iOff];
    real_t invMass = 0.5/sim.species_mass[iSpecies];
    ep = sim.atoms.e[iOff]; 
    ek = (sim.atoms.p.x[iOff] * sim.atoms.p.x[iOff] + sim.atoms.p.y[iOff] * sim.atoms.p.y[iOff] + sim.atoms.p.z[iOff] * sim.atoms.p.z[iOff]) * invMass;
  }
  
  // reduce in smem
  __shared__ real_t sp[THREAD_ATOM_CTA];
  __shared__ real_t sk[THREAD_ATOM_CTA];
  sp[threadIdx.x] = ep;
  sk[threadIdx.x] = ek;
  __syncthreads();
  for (int i = THREAD_ATOM_CTA / 2; i >= 32; i /= 2) {
    if (threadIdx.x < i) {
      sp[threadIdx.x] += sp[threadIdx.x + i];
      sk[threadIdx.x] += sk[threadIdx.x + i];
    }
    __syncthreads();
  }
  
  // reduce in warp
  if (threadIdx.x < 32) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
    ep = sp[threadIdx.x];
    ek = sk[threadIdx.x];
    for (int i = WARP_SIZE / 2; i > 0; i /= 2) {
      ep += __shfl_xor(ep, i);
      ek += __shfl_xor(ek, i);
    }
#else
    if (threadIdx.x < 16) sp[threadIdx.x] += sp[threadIdx.x+16];
    if (threadIdx.x < 8) sp[threadIdx.x] += sp[threadIdx.x+8];
    if (threadIdx.x < 4) sp[threadIdx.x] += sp[threadIdx.x+4];
    if (threadIdx.x < 2) sp[threadIdx.x] += sp[threadIdx.x+2];
    if (threadIdx.x < 1) sp[threadIdx.x] += sp[threadIdx.x+1];

    if (threadIdx.x < 16) sk[threadIdx.x] += sk[threadIdx.x+16];
    if (threadIdx.x < 8) sk[threadIdx.x] += sk[threadIdx.x+8];
    if (threadIdx.x < 4) sk[threadIdx.x] += sk[threadIdx.x+4];
    if (threadIdx.x < 2) sk[threadIdx.x] += sk[threadIdx.x+2];
    if (threadIdx.x < 1) sk[threadIdx.x] += sk[threadIdx.x+1];

    if (threadIdx.x == 0) {
      ep = sp[threadIdx.x];
      ek = sk[threadIdx.x];
    }
#endif
  }

  // one thread adds to gmem
  if (threadIdx.x == 0) {
    atomicAdd(e_pot, ep);
    atomicAdd(e_kin, ek);
  }
}
#endif
