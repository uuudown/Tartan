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

/// \file
/// Functions to maintain neighbor list of each atom. 

#include "defines.h"
#include "CoMDTypes.h"
#include "gpu_neighborList.h"
#include "linkCells.h"
#include "initAtoms.h"
#include "memUtils.h"
#include "parallel.h"
#include "gpu_types.h"

#include <assert.h>
#include <cuda_runtime.h>
#include "gpu_kernels.h"

/// Initialize Neighborlist. Allocates all required data structures and initializes all
/// variables. Requires atoms to be initialized and nLocal needs to be set.
/// \param [in] nLocalBoxes  The index with box iBox of the atom to be moved.
/// \param [in] skinDistance Skin distance used by buildNeighborList.
void initNeighborListGpu(SimGpu * sim, NeighborListGpu* neighborList, const int nLocalBoxes, const real_t skinDistance)
{

   neighborList->nMaxLocal = MAXATOMS*nLocalBoxes; // make this list a little larger to make room for migrated particles
   neighborList->nMaxNeighbors = MAXNEIGHBORLISTSIZE;
   neighborList->skinDistance = skinDistance;
   neighborList->skinDistance2 = skinDistance*skinDistance;
   neighborList->skinDistanceHalf2 = (skinDistance/2.0)*(skinDistance/2.0);
   neighborList->nStepsSinceLastBuild = 0;
   neighborList->updateNeighborListRequired = 1;
   neighborList->updateLinkCellsRequired = 0;
   neighborList->forceRebuildFlag = 1; 

   cudaMalloc((void**)&(neighborList->list), neighborList->nMaxLocal * neighborList->nMaxNeighbors * sizeof(int));
   cudaMalloc((void**)&(neighborList->nNeighbors), neighborList->nMaxLocal * sizeof(int));

   cudaMalloc((void**)&(neighborList->lastR.x), neighborList->nMaxLocal * sizeof(real_t));
   cudaMalloc((void**)&(neighborList->lastR.y), neighborList->nMaxLocal * sizeof(real_t));
   cudaMalloc((void**)&(neighborList->lastR.z), neighborList->nMaxLocal * sizeof(real_t));  

   emptyNeighborListGpu(sim, BOTH);

} 

/// Free all the memory associated with Neighborlist
void destroyNeighborListGpu(NeighborListGpu** neighborList)
{
   if (! neighborList) return;
   if (! *neighborList) return;

   comdFree((*neighborList)->list);
   comdFree((*neighborList)->nNeighbors);
   cudaFree((*neighborList)->lastR.x);
   cudaFree((*neighborList)->lastR.y);
   cudaFree((*neighborList)->lastR.z);
   comdFree((*neighborList));
   *neighborList = NULL;

   return;
}

void neighborListForceRebuildGpu(struct NeighborListGpuSt* neighborList)
{
   neighborList->forceRebuildFlag = 1; 
}



