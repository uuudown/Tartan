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

// templated for the 1st and 3rd EAM passes
// 1 cta is assigned to 1 cell
// all threads shared the same set of neighbors
// so we can store neighbor positions in smem
// 1 warp processes 1 atom
template<int step, bool spline>
__global__
__launch_bounds__(CTA_CELL_CTA, CTA_CELL_ACTIVE_CTAS)
void EAM_Force_cta_cell(SimGpu sim, int *cells_list)
{
  // warp & lane ids
  int warp_id = get_warp_id();
  int lane_id = get_lane_id();

  // cell id = CUDA block id
  int iBox = (cells_list == NULL) ? blockIdx.x : cells_list[blockIdx.x];

  // num of cell atoms & neighbor atoms
  int natoms = sim.boxes.nAtoms[iBox];
  int nneigh = sim.num_neigh_atoms[iBox];

  // distribute smem
  extern __shared__ real_t smem[];

  // neighbor positions
  volatile real_t *rx = smem;
  volatile real_t *ry = rx + CTA_CELL_CTA;
  volatile real_t *rz = rx + 2 * CTA_CELL_CTA;

  // neighbor embed force
  volatile real_t *fe = rx + 3 * CTA_CELL_CTA;

  // local positions
  volatile real_t *irx = smem + CTA_CELL_CTA * (3 + (step == 3));
  volatile real_t *iry = irx + sim.max_atoms_cell;
  volatile real_t *irz = iry + sim.max_atoms_cell;

  // local forces
  volatile real_t *ifx = irz + sim.max_atoms_cell;  
  volatile real_t *ify = ifx + sim.max_atoms_cell;
  volatile real_t *ifz = ify + sim.max_atoms_cell;
  volatile real_t *ie = ifz + sim.max_atoms_cell;
  volatile real_t *irho = ie + sim.max_atoms_cell;

  // per-warp neighbor offsets
  volatile char *nl_off = (char*)(((step == 1) ? irho : ifz) + sim.max_atoms_cell) + warp_id * 64;

  // compute squared cut-off
  real_t rCut2 = sim.eam_pot.cutoff * sim.eam_pot.cutoff;
 
  // save local atoms positions
  if (threadIdx.x < natoms) 
  {
    irx[threadIdx.x] = sim.atoms.r.x[iBox * MAXATOMS + threadIdx.x];
    iry[threadIdx.x] = sim.atoms.r.y[iBox * MAXATOMS + threadIdx.x];
    irz[threadIdx.x] = sim.atoms.r.z[iBox * MAXATOMS + threadIdx.x];

    ifx[threadIdx.x] = 0;
    ify[threadIdx.x] = 0;
    ifz[threadIdx.x] = 0;

    if (step == 1) {
      ie[threadIdx.x] = 0;
      irho[threadIdx.x] = 0;
    }
  }

  // process neighbors in chunks of CTA size to save on smem
  int global_base = 0;
  while (global_base < nneigh)
  {
    int global_neighbor = global_base + threadIdx.x;

    // last chunk might be incomplete
    int tail = 0;
    if (global_base + CTA_CELL_CTA > nneigh)
      tail = nneigh - global_base;

    // load up neighbor particles in smem: 1 thread per neighbor atom
    if (global_neighbor < nneigh)
    {
      int jOff = sim.neighbor_atoms[iBox * N_MAX_NEIGHBORS * MAXATOMS + global_neighbor];

      rx[threadIdx.x] = sim.atoms.r.x[jOff];
      ry[threadIdx.x] = sim.atoms.r.y[jOff];
      rz[threadIdx.x] = sim.atoms.r.z[jOff];

      if (step == 3)
	fe[threadIdx.x] = sim.eam_pot.dfEmbed[jOff];
    }
 
    // ensure data is loaded
    __syncthreads();

    // 1 warp is assigned to 1 atom
    int iatom_base = 0;

    // only process atoms inside current box
    while (iatom_base < natoms)
    {
      int iAtom = iatom_base + warp_id;
      if (iAtom < natoms)
      {
        // init forces and energy
        real_t reg_ifx = 0;
        real_t reg_ify = 0;
        real_t reg_ifz = 0;
        real_t reg_ie = 0;
        real_t reg_irho = 0;
	  
        real_t dx, dy, dz, r2;
		
        // create neighbor list
        int warpTotal = 0;
	for (int base = 0; base < CTA_CELL_CTA; base += WARP_SIZE) 
  	{
          int j = base + lane_id;

	  if (tail == 0 || j < tail) 
          {
	    dx = irx[iAtom] - rx[j];
	    dy = iry[iAtom] - ry[j];
	    dz = irz[iAtom] - rz[j];

            // distance^2
            r2 = dx*dx + dy*dy + dz*dz;
	  }

	  // aggregate neighbors that passes cut-off check
	  // warp-scan using ballot/popc 	
	  uint flag = (r2 <= rCut2 && r2 > 0 && (tail == 0 || j < tail));  // flag(lane id) 
	  uint bits = __ballot(flag);                           // 0 1 0 1  1 1 0 0 = flag(0) flag(1) .. flag(31)
	  uint mask = bfi(0, 0xffffffff, 0, lane_id);         // bits < lane id = 1, bits > lane id = 0
	  uint exc = __popc(mask & bits);                       // exclusive scan 

	  if (flag) 
	    nl_off[warpTotal + exc] = j; 	    		  // fill nl array - compacted

	  warpTotal += __popc(bits);                            // total 1s per warp
	} // compute neighbor lists

	for (int neighbor_id = lane_id; neighbor_id < warpTotal; neighbor_id += WARP_SIZE)
    	{
	  int j = nl_off[neighbor_id];

	  dx = irx[iAtom] - rx[j];
	  dy = iry[iAtom] - ry[j];
	  dz = irz[iAtom] - rz[j];

	  r2 = dx*dx + dy*dy + dz*dz;

      real_t phiTmp, dPhi, rhoTmp, dRho;
      if(!spline)
      {
          real_t r = sqrt_opt(r2);	

          if (step == 1) {
              interpolate(sim.eam_pot.phi, r, phiTmp, dPhi);
              interpolate(sim.eam_pot.rho, r, rhoTmp, dRho);
          }
          else {
              // step = 3
              // TODO: this is not optimal
              interpolate(sim.eam_pot.rho, r, rhoTmp, dRho);
              int iOff = iBox * MAXATOMS + iAtom;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
              dPhi = (__ldg(sim.eam_pot.dfEmbed + iOff) + fe[j]) * dRho;
#else
              dPhi = (sim.eam_pot.dfEmbed[iOff] + fe[j]) * dRho;
#endif
          }

          dPhi /= r;
      }
      else
      {
          if (step == 1) {
              interpolateSpline(sim.eam_pot.phiS, r2, phiTmp, dPhi);
              interpolateSpline(sim.eam_pot.rhoS, r2, rhoTmp, dRho);
          }
          else {
              // step = 3
              // TODO: this is not optimal
              interpolateSpline(sim.eam_pot.rhoS, r2, rhoTmp, dRho);
              int iOff = iBox * MAXATOMS + iAtom;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
              dPhi = (__ldg(sim.eam_pot.dfEmbed + iOff) + fe[j]) * dRho;
#else
              dPhi = (sim.eam_pot.dfEmbed[iOff] + fe[j]) * dRho;
#endif
          }

      }
	  // update forces
	  reg_ifx -= dPhi * dx;
	  reg_ify -= dPhi * dy;
	  reg_ifz -= dPhi * dz;
	
	  // update energy & accumulate rhobar
	  if (step == 1) {
	    reg_ie += phiTmp;
	    reg_irho += rhoTmp;
	  }	
	} // accumulate forces in regs

	warp_reduce<step>(reg_ifx, reg_ify, reg_ifz, reg_ie, reg_irho);

  	// single thread writes the final result
	if (lane_id == 0) 
        {
	  ifx[iAtom] += reg_ifx;
	  ify[iAtom] += reg_ify;
	  ifz[iAtom] += reg_ifz;

    	  if (step == 1) {
	    ie[iAtom] += reg_ie;
	    irho[iAtom] += reg_irho;
	  }
        }
      }  // check if iAtom < num atoms

      iatom_base += CTA_CELL_CTA / WARP_SIZE;
    }  // iterate on all atoms in cell

    __syncthreads();
 
    global_base += CTA_CELL_CTA;
  }  // iterate on all neighbors

  // single thread writes the final result for each atom
  if (threadIdx.x < natoms) 
  {
    int iAtom = threadIdx.x;
    int iOff = iBox * MAXATOMS + threadIdx.x;
    if (step == 1)
    {
      sim.atoms.f.x[iOff] = ifx[iAtom];
      sim.atoms.f.y[iOff] = ify[iAtom];
      sim.atoms.f.z[iOff] = ifz[iAtom];
      sim.atoms.e[iOff] = 0.5 * ie[iAtom];
      sim.eam_pot.rhobar[iOff] = irho[iAtom];
    }
    else {
      // step 3
      sim.atoms.f.x[iOff] += ifx[iAtom];
      sim.atoms.f.y[iOff] += ify[iAtom];
      sim.atoms.f.z[iOff] += ifz[iAtom];
    }
  }
}

__global__
void EAM_Force_cta_cell2(SimGpu sim, int *cell_list)
{
  // compute box ID and local atom ID
  int iAtom = threadIdx.x;
  int iBox = (cell_list == NULL) ? blockIdx.x : cell_list[blockIdx.x];

  if (iAtom < sim.boxes.nAtoms[iBox]) {
    int iOff = iBox * MAXATOMS + iAtom;

    real_t fEmbed, dfEmbed;
    interpolate(sim.eam_pot.f, sim.eam_pot.rhobar[iOff], fEmbed, dfEmbed);
    sim.eam_pot.dfEmbed[iOff] = dfEmbed; // save derivative for halo exchange
    sim.atoms.e[iOff] += fEmbed;
  }
}

