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

#include <assert.h>

// templated for the 1st and 3rd EAM passes
template<int step, bool spline>
__global__
__launch_bounds__(THREAD_ATOM_CTA, THREAD_ATOM_ACTIVE_CTAS)
void EAM_Force_thread_atom(SimGpu sim, AtomListGpu list)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x; 
  if (tid >= list.n) return;

  // compute box ID and local atom ID
  int iAtom = list.atoms[tid];
  int iBox = list.cells[tid]; 
  int iOff = iBox * MAXATOMS + iAtom;

  real_t rCut = sim.eam_pot.cutoff;
  real_t rCut2 = rCut*rCut;

  // init forces and energy
  real_t ifx = 0;
  real_t ify = 0;
  real_t ifz = 0;
  real_t ie = 0;
  real_t irho = 0;

  if (step == 3) {
    ifx = sim.atoms.f.x[iOff];
    ify = sim.atoms.f.y[iOff];
    ifz = sim.atoms.f.z[iOff];
  }

  // fetch position
  real_t irx = sim.atoms.r.x[iOff];
  real_t iry = sim.atoms.r.y[iOff];
  real_t irz = sim.atoms.r.z[iOff];
 
  // loop over my neighbor cells
  for (int j = 0; j < N_MAX_NEIGHBORS; j++) 
  { 
    int jBox = sim.neighbor_cells[iBox * N_MAX_NEIGHBORS + j];

    // loop over all atoms in the neighbor cell 
    for (int jAtom = 0; jAtom < sim.boxes.nAtoms[jBox]; jAtom++) 
    {  
      int jOff = jBox * MAXATOMS + jAtom; 

      real_t dx = irx - sim.atoms.r.x[jOff];
      real_t dy = iry - sim.atoms.r.y[jOff];
      real_t dz = irz - sim.atoms.r.z[jOff];

      // distance^2
      real_t r2 = dx*dx + dy*dy + dz*dz;

      // no divide by zero
      if (r2 <= rCut2 && r2 > 0.0) 
      {

        real_t phiTmp, dPhi, rhoTmp, dRho;
        if(!spline)
        {
            real_t r = sqrt(r2);

            if (step == 1) {
                interpolate(sim.eam_pot.phi, r, phiTmp, dPhi);
                interpolate(sim.eam_pot.rho, r, rhoTmp, dRho);
            }
            else {
                // step = 3
                interpolate(sim.eam_pot.rho, r, rhoTmp, dRho);
                dPhi = (sim.eam_pot.dfEmbed[iOff] + sim.eam_pot.dfEmbed[jOff]) * dRho;
            }

            dPhi /= r;
        }
        else
        {
            if(step == 1) {
                interpolateSpline(sim.eam_pot.phiS, r2, phiTmp, dPhi);
                interpolateSpline(sim.eam_pot.rhoS, r2, rhoTmp, dRho);
            }
            else
            {
                //step 3
                interpolateSpline(sim.eam_pot.rhoS, r2, rhoTmp,dRho);
                dPhi = (sim.eam_pot.dfEmbed[iOff] + sim.eam_pot.dfEmbed[jOff]) * dRho;
            }

        }
        // update forces
        ifx -= dPhi * dx;
        ify -= dPhi * dy;
        ifz -= dPhi * dz;

        // update energy & accumulate rhobar
        if (step == 1) {
          ie += phiTmp;
          irho += rhoTmp;
        }
      } 
    } // loop over all atoms
  } // loop over neighbor cells

  sim.atoms.f.x[iOff] = ifx;
  sim.atoms.f.y[iOff] = ify;
  sim.atoms.f.z[iOff] = ifz;

  if (step == 1) {
    sim.atoms.e[iOff] = 0.5 * ie;
    sim.eam_pot.rhobar[iOff] = irho;
  }
}


/// templated for the 1st and 3rd EAM passes using the neighborlist
template<int step, bool spline>
__global__
__launch_bounds__(THREAD_ATOM_CTA, THREAD_ATOM_ACTIVE_CTAS)
void EAM_Force_thread_atom_NL(SimGpu sim, AtomListGpu list)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x; 
  if (tid >= list.n) return;

  // compute box ID and local atom ID
  int iAtom = list.atoms[tid];
  int iBox = list.cells[tid]; 
  int iOff = iBox * MAXATOMS + iAtom;

  real_t rCut = sim.eam_pot.cutoff;
  real_t rCut2 = rCut*rCut;

  // init forces and energy
  real_t ifx = 0;
  real_t ify = 0;
  real_t ifz = 0;
  real_t ie = 0;
  real_t irho = 0;

  assert(iOff < sim.boxes.nLocalBoxes * MAXATOMS && iOff >=0 );

  if (step == 3) {
    ifx = sim.atoms.f.x[iOff];
    ify = sim.atoms.f.y[iOff];
    ifz = sim.atoms.f.z[iOff];
  }

  real_t *const __restrict__ rx = sim.atoms.r.x;
  real_t *const __restrict__ ry = sim.atoms.r.y;
  real_t *const __restrict__ rz = sim.atoms.r.z;

  // fetch position
  real_t irx = rx[iOff];
  real_t iry = ry[iOff];
  real_t irz = rz[iOff];
 
  int iLid = tid; 
  const int ldNeighborList = sim.atoms.neighborList.nMaxLocal; //leading dimension
  assert(iLid < ldNeighborList);
  int* neighborList = sim.atoms.neighborList.list; 
  int nNeighbors = sim.atoms.neighborList.nNeighbors[iLid];

  // loop over my neighboring particles within the neighbor-list
  for (int j = 0; j < nNeighbors; j++) 
  { 
      const int jLid = j * ldNeighborList + iLid;
      assert(jLid < ldNeighborList * sim.atoms.neighborList.nMaxNeighbors  );
      const int jOff = neighborList[jLid];
      assert(jOff < sim.boxes.nTotalBoxes * MAXATOMS  && jOff >=0 );

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
          real_t phiTmp, dPhi, rhoTmp, dRho;
          if(!spline)
          {
              real_t r = sqrt(r2);

              if (step == 1) {
                  interpolate(sim.eam_pot.phi, r, phiTmp, dPhi);
                  interpolate(sim.eam_pot.rho, r, rhoTmp, dRho);
              }
              else {
                  // step = 3
                  interpolate(sim.eam_pot.rho, r, rhoTmp, dRho);
                  dPhi = (sim.eam_pot.dfEmbed[iOff] + sim.eam_pot.dfEmbed[jOff]) * dRho;
              }

              dPhi /= r;
          }
          else
          {
              if(step == 1) {
                  interpolateSpline(sim.eam_pot.phiS, r2, phiTmp, dPhi);
                  interpolateSpline(sim.eam_pot.rhoS, r2, rhoTmp, dRho);
              }
              else
              {
                  //step 3
                  interpolateSpline(sim.eam_pot.rhoS, r2, rhoTmp,dRho);
                  dPhi = (sim.eam_pot.dfEmbed[iOff] + sim.eam_pot.dfEmbed[jOff]) * dRho;
              }

          }
          // update forces
          ifx -= dPhi * dx;
          ify -= dPhi * dy;
          ifz -= dPhi * dz;

          // update energy & accumulate rhobar
        if (step == 1) {
          ie += phiTmp;
          irho += rhoTmp;
        }
      } 
  } // loop over neighbor-list

  sim.atoms.f.x[iOff] = ifx;
  sim.atoms.f.y[iOff] = ify;
  sim.atoms.f.z[iOff] = ifz;

  if (step == 1) {
    sim.atoms.e[iOff] = 0.5 * ie;
    sim.eam_pot.rhobar[iOff] = irho;
  }
}

// compute embedding energy
__global__ 
void EAM_Force_thread_atom2(SimGpu sim, AtomListGpu list)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= list.n) return;

  // compute box ID and local atom ID
  int iAtom = list.atoms[tid];
  assert(iAtom < MAXATOMS);
  int iBox = list.cells[tid];
  assert(iBox < sim.boxes.nLocalBoxes);

  int iOff = iBox * MAXATOMS + iAtom;

  real_t fEmbed, dfEmbed;
  interpolate(sim.eam_pot.f, sim.eam_pot.rhobar[iOff], fEmbed, dfEmbed);
  sim.eam_pot.dfEmbed[iOff] = dfEmbed; // save derivative for halo exchange
  sim.atoms.e[iOff] += fEmbed;
}

