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
#include "neighborList.h"
#include "linkCells.h"
#include "initAtoms.h"
#include "memUtils.h"
#include "parallel.h"
#include "gpu_kernels.h"

#include <assert.h>

void buildNeighborListCpu(SimFlat* s);
int neighborListUpdateRequiredCpu(NeighborList* neighborList, LinkCell*const  boxes, Atoms* const atoms);

/// Initialize Neighborlist. Allocates all required data structures and initializes all
/// variables. Requires atoms to be initialized and nLocal needs to be set.
/// \param [in] nLocalBoxes  The index with box iBox of the atom to be moved.
/// \param [in] skinDistance Skin distance used by buildNeighborList.
NeighborList* initNeighborList(const int nLocalBoxes, const real_t skinDistance)
{

   NeighborList* neighborList = comdMalloc(sizeof(NeighborList)); 

   neighborList->nMaxLocal = MAXATOMS*nLocalBoxes; // make this list a little larger to make room for migrated particles
   neighborList->maxNeighbors = 1024;//TODO: choose this value dynamically
   neighborList->skinDistance = skinDistance;
   neighborList->skinDistance2 = skinDistance*skinDistance;
   neighborList->skinDistanceHalf2 = (skinDistance/2.0)*(skinDistance/2.0);
   neighborList->nStepsSinceLastBuild = 0;
   neighborList->updateNeighborListRequired = 1;
   neighborList->updateLinkCellsRequired = 0;
   neighborList->forceRebuildFlag = 1; 

   neighborList->list = comdMalloc(sizeof(int) * neighborList->nMaxLocal * neighborList->maxNeighbors); 
   neighborList->nNeighbors = comdMalloc(sizeof(int) * neighborList->nMaxLocal);
   malloc_vec(&(neighborList->lastR), neighborList->nMaxLocal);

   emptyNeighborList(neighborList);

   return neighborList;
} 

/// Free all the memory associated with Neighborlist
void destroyNeighborList(NeighborList** neighborList)
{
   if (! neighborList) return;
   if (! *neighborList) return;

   comdFree((*neighborList)->list);
   comdFree((*neighborList)->nNeighbors);
   free_vec(&(*neighborList)->lastR);
   comdFree((*neighborList));
   *neighborList = NULL;

   return;
}

/// @param boundaryFlag if  0:build boundary+interior; 1: build boundary; 2: build interior
void buildNeighborList(SimFlat* sim, int boundaryFlag)
{
        if(sim->method == THREAD_ATOM_NL || sim->method == WARP_ATOM_NL){
           buildNeighborListGpu(&(sim->gpu), sim->method, boundaryFlag);
        } else if(sim->method == CPU_NL)
           buildNeighborListCpu(sim);
        else
                printf("ERROR: Method not supported!\n");
}

/// Build the neighbor list for all boxes which are marked as dirty.
void buildNeighborListCpu(SimFlat* s)
{
   NeighborList* neighborList = s->atoms->neighborList;
   
   if(neighborList->updateNeighborListRequired == -1)
      neighborListUpdateRequired(s);

   if(neighborList->updateNeighborListRequired == 1){
           emptyNeighborList(neighborList);

           const real_t rCut2 = (s->pot->cutoff+neighborList->skinDistance) * (s->pot->cutoff+neighborList->skinDistance);

           int nbrBoxes[27];
           // loop over local boxes
           for (int iBox=0; iBox<s->boxes->nLocalBoxes; iBox++)
           {
                   int nIBox = s->boxes->nAtoms[iBox];
                   int nNbrBoxes = getNeighborBoxes(s->boxes, iBox, nbrBoxes);
                   // loop over neighbor boxes of iBox (some may be halo boxes)
                   for (int jTmp=0; jTmp<nNbrBoxes; jTmp++)
                   {
                           int jBox = nbrBoxes[jTmp];
#ifndef FULLLIST 
                           if (jBox < iBox ) continue;
#endif

                           int nJBox = s->boxes->nAtoms[jBox];
                           // loop over atoms in iBox
                           for (int iOff=MAXATOMS*iBox,ii=0; ii<nIBox; ii++,iOff++)
                           {
                                   int iLid = s->atoms->lid[iOff];
                                   assert(iLid < neighborList->nMaxLocal);
                                   int* iNeighborList = &(neighborList->list[neighborList->maxNeighbors * iLid]);
                                   int nNeighbors = neighborList->nNeighbors[iLid];
                                   neighborList->lastR.x[iLid] = s->atoms->r.x[iOff];
                                   neighborList->lastR.y[iLid] = s->atoms->r.y[iOff];
                                   neighborList->lastR.z[iLid] = s->atoms->r.z[iOff];

                                   // loop over atoms in jBox
                                   for (int jOff=MAXATOMS*jBox,ij=0; ij<nJBox; ij++,jOff++)
                                   {
#ifndef FULLLIST 
                                           if ( (iBox==jBox) && (ij <= ii) ) continue;
#else
                                           if ( (iBox==jBox) && ij == ii ) continue;
#endif

                                           real_t r2 = 0.0;
                                           real3_old dr;
                                           dr[0] = s->atoms->r.x[iOff] - s->atoms->r.x[jOff];
                                           dr[1] = s->atoms->r.y[iOff] - s->atoms->r.y[jOff];
                                           dr[2] = s->atoms->r.z[iOff] - s->atoms->r.z[jOff];
                                           r2 = dr[0]*dr[0] + dr[1]*dr[1] + dr[2]*dr[2];
                                           if(r2>rCut2) continue;

                                           //add particle j to i's neighbor list
                                           if(r2 <= rCut2) {
                                                   assert(nNeighbors < neighborList->maxNeighbors);
                                                   iNeighborList[nNeighbors++] = jOff;
                                           }
                                   }
                                   neighborList->nNeighbors[iLid] = nNeighbors;
                           }
                   }
                   //pad particles s.t. each neighbor-list is a multiple of the vector_width with particles positioned at infinity
                   for (int iOff=MAXATOMS*iBox,ii=0; ii<nIBox; ii++,iOff++)
                   {
                      int iLid = s->atoms->lid[iOff];
                      assert(iLid < neighborList->nMaxLocal);
                      int* iNeighborList = &(neighborList->list[neighborList->maxNeighbors * iLid]);
                      int nNeighbors = neighborList->nNeighbors[iLid];
                      int padding = VECTOR_WIDTH - (nNeighbors % VECTOR_WIDTH);
                      for( int i = 0; i<padding; ++i){
                         iNeighborList[nNeighbors++] = MAXATOMS*s->boxes->nTotalBoxes;
                      }
                      neighborList->nNeighbors[iLid] = nNeighbors;
//                      printf("%d ",nNeighbors);
                   }
           }
           neighborList->forceRebuildFlag = 0; 
           neighborList->nStepsSinceLastBuild = 1;
           neighborList->updateNeighborListRequired = 0;
   }else
           neighborList->nStepsSinceLastBuild++;

//   for(int i = 0 ; i < s->atoms->nLocal; ++i){
//           int iLid = s->atoms->lid[i];
//           printf("%d \n",neighborList->nNeighbors[iLid]);
//   }
//   exit(-1);
}

/// Sets all neighbor counts to zero
void emptyNeighborList(NeighborList* neighborList)
{
        for(int i = 0; i < neighborList->nMaxLocal; ++i)
           neighborList->nNeighbors[i] = 0;
}

int neighborListUpdateRequired(SimFlat* sim)
{
   if(sim->method == THREAD_ATOM_NL || sim->method == WARP_ATOM_NL || sim->usePairlist)
     return neighborListUpdateRequiredGpu(&(sim->gpu));
   else if(sim->method == CPU_NL)
     return neighborListUpdateRequiredCpu(sim->atoms->neighborList,sim->boxes,sim->atoms);
   else
     return -1;
}

/// \param [inout] neighborList NeighborList (the only value that might be changed is updateNeighborListRequired
/// \return 1 iff neighborlist update is required in this step
int neighborListUpdateRequiredCpu(NeighborList* neighborList, LinkCell*const  boxes, Atoms* const atoms)
{
        if(neighborList->forceRebuildFlag == 1)
                   neighborList->updateNeighborListRequired = 1; 
        else if(neighborList->updateNeighborListRequired == -1){
                //loop over local boxes
                for (int iBox=0; iBox<boxes->nLocalBoxes && neighborList->updateNeighborListRequired != 1; iBox++)
                {
                        for (int iOff=MAXATOMS*iBox,ii=0; ii<boxes->nAtoms[iBox] && neighborList->updateNeighborListRequired != 1; ii++,iOff++)
                        {
                                int iLid = atoms->lid[iOff];
                                real_t dx = neighborList->lastR.x[iLid] - atoms->r.x[iOff];
                                real_t dy = neighborList->lastR.y[iLid] - atoms->r.y[iOff];
                                real_t dz = neighborList->lastR.z[iLid] - atoms->r.z[iOff];

                                //check if a neighborlist build is required
                                if( (dx*dx + dy*dy + dz*dz) > neighborList->skinDistanceHalf2 ){
                                        neighborList->updateNeighborListRequired = 1;
                                        //setDirtyFlat(boxes, iBox); //TODO future version could just rebuild neighbor list for the required boxes.
                                }
                        }
                }

                //if one process has to rebuild the neighbor list, then all have to do so in order to avoid deadlocks
                //TODO: this behavior can be improved in future versions (if one node has to rebuild its NL, then all have to do so).
                int tmpUpdateNeighborListRequired;
                maxIntParallel(&(neighborList->updateNeighborListRequired), &tmpUpdateNeighborListRequired, 1);

                if(tmpUpdateNeighborListRequired > 0)
                   neighborList->updateNeighborListRequired = 1; 
                else
                   neighborList->updateNeighborListRequired = 0; 

        }
        return  neighborList->updateNeighborListRequired;
}

void neighborListForceRebuild(NeighborList* neighborList)
{
   neighborList->forceRebuildFlag = 1; 
}
