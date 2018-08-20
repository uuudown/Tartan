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

__global__
__launch_bounds__(THREAD_ATOM_CTA, THREAD_ATOM_ACTIVE_CTAS)
void LJ_Force_thread_atom(SimGpu sim, AtomListGpu list)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x; 
  if (tid >= list.n) return;

  // compute box ID and local atom ID
  int iAtom = list.atoms[tid];
  int iBox = list.cells[tid]; 

  // common constants for LJ potential
  // TODO: this can be precomputed
  real_t sigma = sim.lj_pot.sigma;
  real_t epsilon = sim.lj_pot.epsilon;
  real_t rCut = sim.lj_pot.cutoff;
  real_t rCut2 = rCut*rCut;

  real_t s6 = sigma*sigma*sigma*sigma*sigma*sigma;

  real_t rCut6 = s6 / (rCut2*rCut2*rCut2);
  real_t eShift = rCut6 * (rCut6 - 1.0f);

  // zero out forces and energy
  real_t ifx = 0;
  real_t ify = 0;
  real_t ifz = 0;
  real_t ie = 0;

  // fetch position
  int iOff = iBox * MAXATOMS + iAtom;
  real_t irx = sim.atoms.r.x[iOff];
  real_t iry = sim.atoms.r.y[iOff];
  real_t irz = sim.atoms.r.z[iOff];
 
  { 
    const int jBox = iBox;
    
    int jOff = jBox * MAXATOMS;
    // loop over all atoms in the neighbor cell 
    for (int jAtom = 0; jAtom < sim.boxes.nAtoms[jBox]; jAtom++) 
    {  

      real_t dx = irx - sim.atoms.r.x[jOff];
      real_t dy = iry - sim.atoms.r.y[jOff];
      real_t dz = irz - sim.atoms.r.z[jOff];

      // distance^2
      real_t r2 = dx*dx + dy*dy + dz*dz;

      // no divide by zero
      if (r2 <= rCut2 && r2 > 0.0f) 
      {
        r2 = 1.0f/r2;
        real_t r6 = s6 * (r2*r2*r2);
        real_t eLocal = r6 * (r6 - 1.0f) - eShift;

        // update energy
        ie += 0.5f * eLocal;
        // different formulation to avoid sqrt computation
        real_t fr = r6*r2*(48.0f*r6 - 24.0f);

        // update forces
        ifx += fr * dx;
        ify += fr * dy;
        ifz += fr * dz;
      }
      ++jOff;
    } // loop over all atoms
  } // loop over neighbor cells

  // loop over my neighbor cells
  for (int j = 1; j < N_MAX_NEIGHBORS; j++) 
  { 
    const int jBox = sim.neighbor_cells[iBox * N_MAX_NEIGHBORS + j];
    
    int jOff = jBox * MAXATOMS;
    // loop over all atoms in the neighbor cell 
    for (int jAtom = 0; jAtom < sim.boxes.nAtoms[jBox]; jAtom++) 
    {  

      real_t dx = irx - sim.atoms.r.x[jOff];
      real_t dy = iry - sim.atoms.r.y[jOff];
      real_t dz = irz - sim.atoms.r.z[jOff];

      // distance^2
      real_t r2 = dx*dx + dy*dy + dz*dz;

      // no divide by zero
      if (r2 <= rCut2) 
      {
        r2 = 1.0f/r2;
        real_t r6 = s6 * (r2*r2*r2);
        real_t eLocal = r6 * (r6 - 1.0f) - eShift;

	// update energy
        ie += 0.5f * eLocal;
        // different formulation to avoid sqrt computation
        real_t fr = r6*r2*(48.0f*r6 - 24.0f);

        // update forces
        ifx += fr * dx;
        ify += fr * dy;
        ifz += fr * dz;
      }
      ++jOff;
    } // loop over all atoms
  } // loop over neighbor cells

  sim.atoms.f.x[iOff] = ifx * epsilon;
  sim.atoms.f.y[iOff] = ify * epsilon;
  sim.atoms.f.z[iOff] = ifz * epsilon;

  sim.atoms.e[iOff] = ie * 4 * epsilon;
}

__global__
__launch_bounds__(THREAD_ATOM_CTA, THREAD_ATOM_ACTIVE_CTAS)
void LJ_Force_thread_atom_interpolation(SimGpu sim, AtomListGpu list)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x; 
  if (tid >= list.n) return;

  // compute box ID and local atom ID
  int iAtom = list.atoms[tid];
  int iBox = list.cells[tid]; 

  real_t rCut = sim.lj_pot.cutoff;
  real_t rCut2 = rCut*rCut;

  // zero out forces and energy
  real_t ifx = 0;
  real_t ify = 0;
  real_t ifz = 0;
  real_t ie = 0;

  real_t *const __restrict__ rx = sim.atoms.r.x;
  real_t *const __restrict__ ry = sim.atoms.r.y;
  real_t *const __restrict__ rz = sim.atoms.r.z;


  // fetch position
  int iOff = iBox * MAXATOMS + iAtom;
  real_t irx = sim.atoms.r.x[iOff];
  real_t iry = sim.atoms.r.y[iOff];
  real_t irz = sim.atoms.r.z[iOff];
 
  // loop over my neighbor cells
#pragma unroll
  for (int j = 0; j < N_MAX_NEIGHBORS; j++) 
  { 
    int jBox = sim.neighbor_cells[iBox * N_MAX_NEIGHBORS + j];

    // loop over all atoms in the neighbor cell 
    for (int jAtom = 0; jAtom < sim.boxes.nAtoms[jBox]; jAtom++) 
    {  
      int jOff = jBox * MAXATOMS + jAtom; 

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
      real_t dx = irx - __ldg(&rx[jOff]);
      real_t dy = iry - __ldg(&ry[jOff]);
      real_t dz = irz - __ldg(&rz[jOff]);
#else
      real_t dx = irx - rx[jOff];
      real_t dy = iry - ry[jOff];
      real_t dz = irz - rz[jOff];
#endif
      // distance^2
      real_t r2 = dx*dx + dy*dy + dz*dz;

      // no divide by zero
      if (r2 <= rCut2 && r2 > 0.0) 
      {
        real_t r = sqrt(r2);
        
        real_t etmp, ftmp;
        interpolate(sim.lj_pot.lj_interpolation, r, etmp, ftmp);

	// update energy
        ie += 0.5 * etmp;

        // different formulation to avoid sqrt computation
        real_t fr = -ftmp/r;

        // update forces
        ifx += fr * dx;
        ify += fr * dy;
        ifz += fr * dz;
      } 
    } // loop over all atoms
  } // loop over neighbor cells

  sim.atoms.f.x[iOff] = ifx;
  sim.atoms.f.y[iOff] = ify;
  sim.atoms.f.z[iOff] = ifz;

  sim.atoms.e[iOff] = ie;
}

