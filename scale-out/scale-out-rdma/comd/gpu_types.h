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

#pragma once

#ifndef __GPU_TYPES_H_
#define __GPU_TYPES_H_

#include "mytype.h"

// TODO: we can change that to 16 since # of max atoms is always less than 15
#define N_MAX_NEIGHBORS  27

typedef struct HashTableGpuSt
{
   int *offset;  //!< Stores the offsets for each received particles. (e.g. if the first particle will be stored to iOff=133, then offset[0] = 133)
   int nEntriesPut; //!< Number of stored particles in the offset array.
   int nEntriesGet; //!< Number of particles that have already been read from the offset array. (only used by hashtableGet)
   int nMaxEntries; //!< Size of the offset array
} HashTableGpu;


typedef struct InterpolationObjectGpuSt
{
   int n;           //!< the number of values in the table
   real_t x0;       //!< the starting ordinate range
   real_t xn;       //!< the ending ordinate range
   real_t invDx;    //!< the inverse of the table spacing
   real_t invDxHalf;//!< Half of the inverse of the table spacing
   real_t* values;  //!< the abscissa values
   real_t invDxXx0; //!< the starting ordinate range times the inverse of the table spacing

} InterpolationObjectGpu;

typedef struct InterpolationSplineObjectGpuSt
{
    int n;
    //We do not need precision here
    float x0;  
    float xn;
    float invDx;
    float invDxXx0;
    real_t * coefficients;
} InterpolationSplineObjectGpu;

typedef struct LjPotentialGpuSt
{
   real_t cutoff;          //!< potential cutoff distance in Angstroms
   real_t sigma;
   real_t epsilon;

   InterpolationObjectGpu lj_interpolation;

   real_t plcutoff;        //!< cutoff for pairlist construction

} LjPotentialGpu;



typedef struct EamPotentialGpuSt
{
   real_t cutoff;          //!< potential cutoff distance in Angstroms

   InterpolationObjectGpu phi;  //!< Pair energy
   InterpolationObjectGpu rho;  //!< Electron Density
   InterpolationObjectGpu f;    //!< Embedding Energy

   InterpolationSplineObjectGpu phiS;  //!< Pair energy
   InterpolationSplineObjectGpu rhoS;  //!< Electron Density
   InterpolationSplineObjectGpu fS;    //!< Embedding Energy

   real_t* rhobar;        //!< per atom storage for rhobar
   real_t* dfEmbed;       //!< per atom storage for derivative of Embedding

} EamPotentialGpu;

typedef struct LinkCellGpuSt
{
  // # of local/total boxes
  int nLocalBoxes;
  int nTotalBoxes;

  int3_t gridSize;

  real3_t localMin;
  real3_t localMax;
  real3_t invBoxSize;

   int *boxIDLookUp; //!< 3D array storing the box IDs 
   int3_t *boxIDLookUpReverse; //!< 1D array storing the tuple for a given box ID 

  int *nAtoms;		// number of atoms per cell
} LinkCellGpu;

// compacted list of atoms & corresponding cells
typedef struct AtomList
{
  int n;      //<! total number atoms within this list
  int *atoms; //<! particleID (non-global)
  int *cells; //<! cellID
} AtomListGpu;

/// Neighbor List data
typedef struct NeighborListGpuSt
{

   int nMaxNeighbors;    //!< Maximum number of neighbors each atom can store within the neighbor list. (This value may increase with time)
   int nMaxLocal;         //!< maximum number of particles that fit on this processor 
   real_t skinDistance; //!< skinDistance in Angstrom. Particle i is in the neighborlist of particle j if r_ij <= (cutoff + skinDistance).
   real_t skinDistance2; //!< skinDistance**2 in Angstrom. Particle i is in the neighborlist of particle j if r_ij <= (cutoff + skinDistance).
   real_t skinDistanceHalf2; //!< (skinDistance/2)**2 in Angstrom.
   int* nNeighbors;      //!< Number of neighbors within the neighbor list of each atom
   int* list;           //!< neighborlist of size maxNeighbors * nNeighborlists. Successive elements within a single neighborlist have a stride of nNeighborlists (due to coalessed memory accesses). 
   int nStepsSinceLastBuild; //!< #MD steps since last build
   int updateNeighborListRequired; //!< Flag that indicates if this step requires a neighborList build (-1:not set, 0:not required, 1:required)
   int updateLinkCellsRequired; //!< Flag that indicates if we need to rebuild the linkCells. This is required to assure the same order of the particles within halo and boundary cells
   vec_t lastR;        //!< Device pointer. positions at last buildNeighborlist() call. Used to identify dirty cells (i.e. cells which require a new neighborlist).
   int forceRebuildFlag;    //!< flag indicates if a neighborlist rebuild is forced to happen (this is different from updateNeighborListRequired) 

} NeighborListGpu;

typedef struct AtomsGpuSt{

  vec_t r;			// atoms positions
  vec_t p;			// atoms momentum
  vec_t f;			// atoms forces
  real_t *e;		// atoms energies
  int *iSpecies;  // atoms species id
  int *gid;			// atoms global id
  NeighborListGpu neighborList;

} AtomsGpu;

typedef struct SimGpuSt {
  int max_atoms_cell;		// max atoms per cell (usually < 32)

  AtomsGpu atoms;

  int *neighbor_cells;		// neighbor cells indices
  int *neighbor_atoms;		// neighbor atom offsets 
  int *num_neigh_atoms;		// number of neighbor atoms per cell

  // species data
  real_t *species_mass;		// masses of species

  LinkCellGpu boxes;

  HashTableGpu d_hashTable;
  int *d_updateLinkCellsRequired; //flag that indicates that another linkCell() call is required //TODO this should be haloExchangeRequired()
  int *cell_type;		// type of cell: 0 - interior, 1 - boundary

  AtomListGpu a_list;		// all local cells
  AtomListGpu i_list;		// interior cells
  AtomListGpu b_list;		// boundary cells

  // potentials
  LjPotentialGpu lj_pot;
  EamPotentialGpu eam_pot;

  int * pairlist;

  int genPairlist;
  int usePairlist;

} SimGpu;

#endif
