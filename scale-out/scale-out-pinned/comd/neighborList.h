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

#ifndef __NEIGHBOR_LIST_H_
#define __NEIGHBOR_LIST_H_

#include "mytype.h"

struct LinkCellSt;
struct SimFlatSt;
struct AtomsSt;

/// Neighbor List data
typedef struct NeighborListSt
{

   int maxNeighbors;    //!< Maximum number of neighbors each atom can store within the neighbor list. (This value may increase with time)
   int nMaxLocal;         //!< maximum number of particles that fit on this processor 
   real_t skinDistance; //!< skinDistance in Angstrom. Particle i is in the neighborlist of particle j if r_ij <= (cutoff + skinDistance).
   real_t skinDistance2; //!< skinDistance**2 in Angstrom. Particle i is in the neighborlist of particle j if r_ij <= (cutoff + skinDistance).
   real_t skinDistanceHalf2; //!< (skinDistance/2)**2 in Angstrom.
   int* nNeighbors;      //!< Number of neighbors within the neighbor list of each atom
   int* list;           //!< neighborlist of size maxNeighbors * nNeighborlists. 
   int nStepsSinceLastBuild; //!< #MD steps since last build
   int updateNeighborListRequired; //!< Flag that indicates if this step requires a neighborList build (-1:not set, 0:not required, 1:required)
   int updateLinkCellsRequired; //!< Flag that indicates if we need to rebuild the linkCells. This is required to assure the same order of the particles within halo and boundary cells
   vec_t lastR;        //!< positions at last buildNeighborlist() call. Used to identify dirty cells (i.e. cells which require a new neighborlist).
   int forceRebuildFlag;    //!< flag indicates if a neighborlist rebuild is forced to happen (this is different from updateNeighborListRequired) 

} NeighborList;

/// Initialized the NeighborList data stucture
NeighborList* initNeighborList(const int nLocalBoxes, const real_t skinDistance);

/// frees all data associated with *neighborList
void destroyNeighborList(NeighborList** neighborList);

/// Build the neighbor list for all boxes which are marked as dirty.
void buildNeighborList(struct SimFlatSt* s, int boundaryFlag);

/// Sets all neighbor counts to zero
void emptyNeighborList(NeighborList* neighborList);

/// returns 1 iff this step will require a rebuild of the neighborlist
int neighborListUpdateRequired(struct SimFlatSt* sim);

/// Forces a rebuild of the neighborlist (but does not rebuild the neighborlist yet, this requires a separate call).
void neighborListForceRebuild(NeighborList* neighborList);
#endif
