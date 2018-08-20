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

#include "defines.h"

__global__ void AdvanceVelocity(SimGpu sim, real_t dt)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= sim.a_list.n) return;

  int iAtom = sim.a_list.atoms[tid];
  int iBox = sim.a_list.cells[tid];

  int iOff = iBox * MAXATOMS + iAtom;

  sim.atoms.p.x[iOff] += dt * sim.atoms.f.x[iOff]; 
  sim.atoms.p.y[iOff] += dt * sim.atoms.f.y[iOff]; 
  sim.atoms.p.z[iOff] += dt * sim.atoms.f.z[iOff]; 
}

__global__ void AdvancePosition(SimGpu sim, real_t dt)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= sim.a_list.n) return;

  int iAtom = sim.a_list.atoms[tid];
  int iBox = sim.a_list.cells[tid];

  int iOff = iBox * MAXATOMS + iAtom;
  
  int iSpecies = sim.atoms.iSpecies[iOff];
  real_t invMass = 1.0/sim.species_mass[iSpecies];

  sim.atoms.r.x[iOff] += dt * sim.atoms.p.x[iOff] * invMass;
  sim.atoms.r.y[iOff] += dt * sim.atoms.p.y[iOff] * invMass;
  sim.atoms.r.z[iOff] += dt * sim.atoms.p.z[iOff] * invMass;
}
