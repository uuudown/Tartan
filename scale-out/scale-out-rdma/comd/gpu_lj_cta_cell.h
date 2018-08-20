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
__launch_bounds__(CTA_CELL_CTA, CTA_CELL_ACTIVE_CTAS)
void LJ_Force_cta_cell(SimGpu sim, int * cells_list, real_t rCut2, real_t s6)
{
  __shared__ real_t otherX[SHARED_SIZE_CTA_CELL];
  __shared__ real_t otherY[SHARED_SIZE_CTA_CELL];
  __shared__ real_t otherZ[SHARED_SIZE_CTA_CELL];

  // compute box ID and local atom ID
  const int iBox = (cells_list == NULL)? blockIdx.x: cells_list[blockIdx.x]; 
  const int nAtoms = sim.boxes.nAtoms[iBox];
  
  // common constants for LJ potential
  const real_t epsilon = sim.lj_pot.epsilon;

  const real_t rCut6 = s6 / (rCut2*rCut2*rCut2);
  const real_t eShift = rCut6 * (rCut6 - 1.0f);

  for(int iAtom = threadIdx.x; iAtom < MAXATOMS; iAtom += blockDim.x)
  {

      // zero out forces and energy
      real_t ifx = 0.f;
      real_t ify = 0.f;
      real_t ifz = 0.f;
      real_t ie = 0.f;
      

      // fetch position
      const int iOff = iBox * MAXATOMS + iAtom;

      const real_t irx = sim.atoms.r.x[iOff];
      const real_t iry = sim.atoms.r.y[iOff];
      const real_t irz = sim.atoms.r.z[iOff];

      // loop over my neighbor cells
      for (int j = 0; j < N_MAX_NEIGHBORS; j++) 
      {
          const int jBox = sim.neighbor_cells[iBox * N_MAX_NEIGHBORS + j];
          for(int base = 0; base < MAXATOMS; base += SHARED_SIZE_CTA_CELL)
          {
              __syncthreads();
              //Fetch atom positions
#pragma unroll
              for(int i = 0; i < SHARED_SIZE_CTA_CELL; i += CTA_CELL_CTA)
              {
                  otherX[i+threadIdx.x] = sim.atoms.r.x[jBox*MAXATOMS + i + threadIdx.x + base];
                  otherY[i+threadIdx.x] = sim.atoms.r.y[jBox*MAXATOMS + i + threadIdx.x + base];
                  otherZ[i+threadIdx.x] = sim.atoms.r.z[jBox*MAXATOMS + i + threadIdx.x + base];
              }
              __syncthreads();
              if(iAtom >= nAtoms)
                  continue;
              int maxN = SHARED_SIZE_CTA_CELL + base < sim.boxes.nAtoms[jBox]?SHARED_SIZE_CTA_CELL:sim.boxes.nAtoms[jBox]-base;
              // loop over all atoms in the neighbor cell 
              for (int jAtom = 0; jAtom < maxN; jAtom++) 
              {  
                  real_t dx = irx - otherX[jAtom];
                  real_t dy = iry - otherY[jAtom];
                  real_t dz = irz - otherZ[jAtom];

                  // distance^2
                  real_t r2 = dx*dx + dy*dy + dz*dz;

                  // no divide by zero
                  if (r2 <= rCut2 && r2 != 0.0f)  
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
              } // loop over all atoms
          }
      } // loop over neighbor cells

      sim.atoms.f.x[iOff] = ifx * epsilon;
      sim.atoms.f.y[iOff] = ify * epsilon;
      sim.atoms.f.z[iOff] = ifz * epsilon;

      sim.atoms.e[iOff] = ie * 4.f * epsilon;

  }
}

template<bool genPairlist, int atomsPerInt>
__global__
__launch_bounds__(CTA_CELL_CTA, CTA_CELL_ACTIVE_CTAS)
void LJ_Force_cta_cell_pairlist(SimGpu sim, int * cells_list, real_t rCut2, real_t s6, real_t plcutoff)
{
    __shared__ volatile real_t otherX[CTA_CELL_CTA];
    __shared__ volatile real_t otherY[CTA_CELL_CTA];
    __shared__ volatile real_t otherZ[CTA_CELL_CTA];
    // compute box ID and local atom ID
    const int iBox = (cells_list == NULL)? blockIdx.x: cells_list[blockIdx.x]; 
    const int nAtoms = sim.boxes.nAtoms[iBox];

    // common constants for LJ potential
    const real_t epsilon = sim.lj_pot.epsilon;

    const real_t rCut6 = s6 / (rCut2*rCut2*rCut2);
    const real_t eShift = rCut6 * (rCut6 - 1.0f);

    const int laneid = get_lane_id();
    const int warp_start = threadIdx.x & 96;

    for(int iAtom = threadIdx.x; iAtom < MAXATOMS; iAtom += blockDim.x)
    {
        // zero out forces and energy
        real_t ifx = 0.f;
        real_t ify = 0.f;
        real_t ifz = 0.f;
        real_t ie = 0.f;

        const int  warpid = blockIdx.x * MAXATOMS/WARP_SIZE + (iAtom >> 5);

        // fetch position
        const int iOff = iBox * MAXATOMS + iAtom;

        const real_t irx = sim.atoms.r.x[iOff];
        const real_t iry = sim.atoms.r.y[iOff];
        const real_t irz = sim.atoms.r.z[iOff];
        if(genPairlist)
        {
            sim.atoms.neighborList.lastR.x[iOff] = irx;
            sim.atoms.neighborList.lastR.y[iOff] = iry;
            sim.atoms.neighborList.lastR.z[iOff] = irz;
        }

        // loop over my neighbor cells
        for (int j = 0; j < N_MAX_NEIGHBORS; j++) 
        {
            const int jBox = sim.neighbor_cells[iBox * N_MAX_NEIGHBORS + j];
            
            unsigned int flag = 0;
            unsigned int mask;
            int base_shared = warp_start;
            //MAXATOMS has to be multiple of PAIRLIST_STEP
            for(int base = 0; base < MAXATOMS; base += PAIRLIST_STEP)
            {
                //Fetch atom positions
                //Warps download the same atoms to avoid synchronisation
                //and false dependencies
                if(base % WARP_SIZE == 0)
                {
                    otherX[threadIdx.x] = sim.atoms.r.x[jBox*MAXATOMS + laneid + base];
                    otherY[threadIdx.x] = sim.atoms.r.y[jBox*MAXATOMS + laneid + base];
                    otherZ[threadIdx.x] = sim.atoms.r.z[jBox*MAXATOMS + laneid + base];
                    base_shared = warp_start;
                }


                if(genPairlist)
                {
                    if(base % atomsPerInt == 0)
                    {
                        if(base != 0)
                        {
                            if(threadIdx.x % 32 == 0)
                                sim.pairlist[N_MAX_NEIGHBORS * ((MAXATOMS+atomsPerInt-1)/atomsPerInt) * warpid + j * ((MAXATOMS+atomsPerInt-1)/atomsPerInt)+(base-PAIRLIST_STEP)/atomsPerInt] = flag;
                        }
                        flag = 0;
                        mask = 1;
                    }
                    else
                    {
                        mask <<= 1;
                    }
                }
                else
                {
                    flag >>= 1;
                    if(base % atomsPerInt == 0)
                    {
                        flag = sim.pairlist[N_MAX_NEIGHBORS * ((MAXATOMS+atomsPerInt-1)/atomsPerInt) * warpid + j * ((MAXATOMS+atomsPerInt-1)/atomsPerInt)+base/atomsPerInt];
                    }
                    if((flag & 1) == 0)
                    {
                        base_shared += PAIRLIST_STEP;
                        continue;
                    }
                }
                if(iAtom >= nAtoms)
                    continue;
                int maxN =  (PAIRLIST_STEP + base < sim.boxes.nAtoms[jBox]?PAIRLIST_STEP:sim.boxes.nAtoms[jBox]-base) + base_shared;
                // loop over all atoms stored in shared memory
                for (int jAtom = base_shared; jAtom < maxN; jAtom++) 
                {  
                    real_t dx = irx - otherX[jAtom];
                    real_t dy = iry - otherY[jAtom];
                    real_t dz = irz - otherZ[jAtom];

                    // distance^2
                    real_t r2 = dx*dx + dy*dy + dz*dz;

                    if(genPairlist && __any(r2 <= plcutoff * plcutoff))
                    {
                        flag |= mask;
                    
                    }
                    // no divide by zero
                    if (r2 <= rCut2 && r2 != 0.0f)  
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
                } // loop over all atoms
                base_shared += PAIRLIST_STEP;
            }
            if(genPairlist)
            {
                if(threadIdx.x % 32 == 0)
                    sim.pairlist[N_MAX_NEIGHBORS * ((MAXATOMS+atomsPerInt-1)/atomsPerInt) * warpid + j * ((MAXATOMS+atomsPerInt-1)/atomsPerInt)+(MAXATOMS-PAIRLIST_STEP)/atomsPerInt] = flag;
            }

        } // loop over neighbor cells

        sim.atoms.f.x[iOff] = ifx * epsilon;
        sim.atoms.f.y[iOff] = ify * epsilon;
        sim.atoms.f.z[iOff] = ifz * epsilon;
        
        sim.atoms.e[iOff] = ie * 4.f * epsilon;

    }
}

