/// \file
/// Functions to maintain link cell structures for fast pair finding.

#include "linkCells.h"

#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <math.h>

#include "parallel.h"
#include "memUtils.h"
#include "decomposition.h"
#include "defines.h"
#include "hashTable.h"
#include "mytype.h"
#include "neighborList.h"
#include "performanceTimers.h"
#include "CoMDTypes.h"
#include "gpu_kernels.h"

#define   MIN(A,B) ((A) < (B) ? (A) : (B))
#define   MAX(A,B) ((A) > (B) ? (A) : (B))

static void copyAtom(LinkCell* boxes, Atoms* atoms, int iAtom, int iBox, int jAtom, int jBox);
static void getTuple(LinkCell* boxes, int iBox, int* ixp, int* iyp, int* izp);

int computeHilbertCode(int x, int y, int z) //TODO: This piece of code has been take and modified from: http://codereview.stackexchange.com/questions/26671/faster-3d-hilbert-curve-in-c (add citation)
{
  const int nbits = sizeof(int) > sizeof(int) ? 20 : 10; // nbits is the curve order.
  const int transform[8] = {0, 1, 7, 6, 3, 2, 4, 5};

  int s = 0;

  for( int i = nbits - 1; i >= 0; i-- ) {
    int xi = ((int)((x)>>i))&1;
    int yi = ((int)((y)>>i))&1;
    int zi = ((int)((z)>>i))&1;

    int index = (xi<<2) + (yi<<1) + zi;
    s = (s<<3) + transform[index];

    if( index==0 ) {
      int temp = z; z = y; y = temp; //swap y and z
    } else if( index==1 ) {
      int temp = x; x = y; y = temp; //swap x and y
    } else if( index==2 ) {
      int temp = (z)^(-1); z = (y)^(-1); y = temp; //swap and complement z and y
    } else if( index==3 ) {
      int temp = (x)^(-1); x = (y)^(-1); y = temp; //swap and complement x and y
    } else if( index==4 ) {
      x = (x)^(-1); z = (z)^(-1); // complement z and x
    } else if( index==5 ) {
      int temp = x; x = y; y = temp; //swap x and y
    } else if( index==6 ) {
      x = (x)^(-1); z = (z)^(-1); // complement z and x
    } else if( index==7 ) {
      int temp = (x)^(-1); x = (y)^(-1); y = temp; //swap and complement x and y
    }
  }

  return s;
}

/// In CoMD 1.1, atoms are stored in link cells.  Link cells are widely
/// used in classical MD to avoid an O(N^2) search for atoms that
/// interact.  Link cells are formed by subdividing the local spatial
/// domain with a Cartesian grid where the grid spacing in each
/// direction is at least as big as he potential's cutoff distance.
/// Because atoms don't interact beyond the potential cutoff, for an
/// atom iAtom in any given link cell, we can be certain that all atoms
/// that interact with iAtom are contained in the same link cell, or one
/// of the 26 neighboring link cells.
/// 
/// CoMD chooses the link cell size (boxSize) on each axis to be the
/// shortest possible distance, longer than cutoff, such that the local
/// domain size divided by boxSize is an integer.  I.e., the link cells
/// are commensurate with with the local domain size.  While this does
/// not result in the smallest possible link cells, it does allow us to
/// keep a strict separation between the link cells that are entirely
/// inside the local domain and those that represent halo regions.
///
/// The number of local link cells in each direction is stored in
/// gridSize.  Local link cells have 3D grid coordinates (ix, iy, iz)
/// where ix, iy, and iz can range from 0 to gridSize[iAxis]-1,
/// whiere iAxis is 0 for x, 1 for y and 2 for the z direction.  The
/// number of local link cells is thus nLocalBoxes =
/// gridSize[0]*gridSize[1]*gridSize[2].
///
/// The local link cells are surrounded by one complete shell of halo
/// link cells.  The halo cells provide temporary storage for halo or
/// "ghost" atoms that belong to other tasks, but whose coordinates are
/// needed locally to complete the force calculation.  Halo link cells
/// have at least one coordinate with a value of either -1 or
/// gridSize[iAxis].
///
/// Because CoMD stores data in ordinary 1D C arrays, a mapping is
/// needed from the 3D grid coords to a 1D array index.  For the local
/// cells we use the conventional mapping ix + iy*nx + iz*nx*ny.  This
/// keeps all of the local cells in a contiguous region of memory
/// starting from the beginning of any relevant array and makes it easy
/// to iterate the local cells in a single loop.  Halo cells are mapped
/// differently.  After the local cells, the two planes of link cells
/// that are face neighbors with local cells across the -x or +x axis
/// are next.  These are followed by face neighbors across the -y and +y
/// axis (including cells that are y-face neighbors with an x-plane of
/// halo cells), followed by all remaining cells in the -z and +z planes
/// of halo cells.  The total number of link cells (on each rank) is
/// nTotalBoxes.
///
/// Data storage arrays that are used in association with link cells
/// should be allocated to store nTotalBoxes*MAXATOMS items.  Data for
/// the first atom in linkCell iBox is stored at index iBox*MAXATOMS.
/// Data for subsequent atoms in the same link cell are stored
/// sequentially, and the number of atoms in link cell iBox is
/// nAtoms[iBox].
///
/// \see getBoxFromTuple is the 3D->1D mapping for link cell indices.
/// \see getTuple is the 1D->3D mapping
///
/// \param [in] cutoff The cutoff distance of the potential.
LinkCell* initLinkCells(const Domain* domain, real_t cutoff, int useHilbert)
{
   assert(domain);
   LinkCell* ll = (LinkCell*)comdMalloc(sizeof(LinkCell));

   for (int i = 0; i < 3; i++)
   {
      ll->localMin[i] = domain->localMin[i];
      ll->localMax[i] = domain->localMax[i];
      ll->gridSize[i] = domain->localExtent[i] / cutoff; // local number of boxes
      ll->boxSize[i] = domain->localExtent[i] / ((real_t) ll->gridSize[i]);
      ll->invBoxSize[i] = 1.0/ll->boxSize[i];
   }

   ll->nLocalBoxes = ll->gridSize[0] * ll->gridSize[1] * ll->gridSize[2];
   
   ll->nHaloBoxes = 2 * ((ll->gridSize[0] + 2) *
                         (ll->gridSize[1] + ll->gridSize[2] + 2) +
                         (ll->gridSize[1] * ll->gridSize[2]));

   ll->nTotalBoxes = ll->nLocalBoxes + ll->nHaloBoxes;

   if(useHilbert && !(ISPOWER2(ll->gridSize[0]) && ISPOWER2(ll->gridSize[1]) && ISPOWER2(ll->gridSize[2]))){
      useHilbert = 0;
      printf("WARNING: We are NOT using space-filling curves because the grid-dimension is not a power of two. (this feature needs to be implemented)\n");
   }

   ll->boxIDLookUpReverse = (int3_t*) comdMalloc(ll->nLocalBoxes * sizeof(int3_t)); //!< 1D array storing the tuple for a given box ID 
   ll->boxIDLookUp = (int*) comdMalloc(ll->gridSize[0] * ll->gridSize[1] * ll->gridSize[2] * sizeof(int));
   for(int i = 0; i < ll->gridSize[2]; ++i){
      for(int j = 0; j < ll->gridSize[1]; ++j){
         for(int k = 0; k < ll->gridSize[0]; ++k){
            int idx = IDX3D(k,j,i,ll->gridSize[0],ll->gridSize[1]);
            int boxID;
            if(useHilbert == 1)
              boxID = computeHilbertCode(k,j,i);
            else
              boxID = idx;
            assert(boxID < ll->nLocalBoxes);
            ll->boxIDLookUp[idx] = boxID; 
            ll->boxIDLookUpReverse[boxID].x = k;
            ll->boxIDLookUpReverse[boxID].y = j;
            ll->boxIDLookUpReverse[boxID].z = i;
         }
      }
   }
   
   ll->nAtoms = (int*)comdMalloc(ll->nTotalBoxes*sizeof(int));
   for (int iBox=0; iBox<ll->nTotalBoxes; ++iBox)
      ll->nAtoms[iBox] = 0;

   assert ( (ll->gridSize[0] >= 2) && (ll->gridSize[1] >= 2) && (ll->gridSize[2] >= 2) );
   return ll;
}

void destroyLinkCells(LinkCell** boxes)
{
   if (! boxes) return;
   if (! *boxes) return;

   comdFree((*boxes)->nAtoms);
   comdFree(*boxes);
   *boxes = NULL;

   return;
}

/// \details
/// Populates the nbrBoxes array with the 27 boxes that are adjacent to
/// iBox.  The count is 27 instead of 26 because iBox is included in the
/// list (as neighbor 13).  Caller is responsible to alloc and free
/// nbrBoxes.
/// \return The number of nbr boxes (always 27 in this implementation).
int getNeighborBoxes(LinkCell* boxes, int iBox, int* nbrBoxes)
{
   int ix, iy, iz;
   getTuple(boxes, iBox, &ix, &iy, &iz);
   
   int count = 0;
   for (int i=ix-1; i<=ix+1; i++)
      for (int j=iy-1; j<=iy+1; j++)
         for (int k=iz-1; k<=iz+1; k++)
            nbrBoxes[count++] = getBoxFromTuple(boxes,i,j,k);
   
   return count;
}

/// \details
/// Finds the appropriate link cell for an atom based on the spatial
/// coordinates and stores data in that link cell.
/// \param [in] gid   The global of the atom.
/// \param [in] iType The species index of the atom.
/// \param [in] x     The x-coordinate of the atom.
/// \param [in] y     The y-coordinate of the atom.
/// \param [in] z     The z-coordinate of the atom.
/// \param [in] px    The x-component of the atom's momentum.
/// \param [in] py    The y-component of the atom's momentum.
/// \param [in] pz    The z-component of the atom's momentum.
/// \return offset of this particle within the atoms structure
int putAtomInBox(LinkCell* boxes, Atoms* atoms,
                  const int gid, const int iType,
                  const real_t x,  const real_t y,  const real_t z,
                  const real_t px, const real_t py, const real_t pz)
{
   real_t xyz[3] = {x,y,z};
   
   // Find correct box.
   int iBox = getBoxFromCoord(boxes, xyz);
   int iOff = iBox*MAXATOMS;
   iOff += boxes->nAtoms[iBox];
   
   // assign values to array elements
   if (iBox < boxes->nLocalBoxes)
      atoms->nLocal++;
   boxes->nAtoms[iBox]++;
   atoms->gid[iOff] = gid;
   atoms->iSpecies[iOff] = iType;
   
   atoms->r.x[iOff] = x;
   atoms->r.y[iOff] = y;
   atoms->r.z[iOff] = z;
   
   atoms->p.x[iOff] = px;
   atoms->p.y[iOff] = py;
   atoms->p.z[iOff] = pz;

   return iOff;
}

/// \details
/// Updates the partile at position iOff within the atoms structure.
/// \param [in] gid   The global of the atom.
/// \param [in] iType The species index of the atom.
/// \param [in] x     The x-coordinate of the atom.
/// \param [in] y     The y-coordinate of the atom.
/// \param [in] z     The z-coordinate of the atom.
/// \param [in] px    The x-component of the atom's momentum.
/// \param [in] py    The y-component of the atom's momentum.
/// \param [in] pz    The z-component of the atom's momentum.
/// \param [in] iOff  The position within the atoms structure where this particle should be stored.
void updateAtomInBoxAt(LinkCell* boxes, Atoms* atoms,
                  const int gid, const int iType,
                  const real_t x,  const real_t y,  const real_t z,
                  const real_t px, const real_t py, const real_t pz,
                  const int iOff)
{
   
   
   int iBox = iOff/MAXATOMS;
   boxes->nAtoms[iBox]++;

   atoms->gid[iOff] = gid;
   atoms->iSpecies[iOff] = iType;


   atoms->r.x[iOff] = x;
   atoms->r.y[iOff] = y;
   atoms->r.z[iOff] = z;
   
   atoms->p.x[iOff] = px;
   atoms->p.y[iOff] = py;
   atoms->p.z[iOff] = pz;
}

/// Calculates the link cell index from the grid coords.  The valid
/// coordinate range in direction ii is [-1, gridSize[ii]].  Any
/// coordinate that involves a -1 or gridSize[ii] is a halo link cell.
/// Because of the order in which the local and halo link cells are
/// stored the indices of the halo cells are special cases.
/// \see initLinkCells for an explanation of storage order.
int getBoxFromTuple(LinkCell* boxes, int ix, int iy, int iz)
{
   int iBox = 0;
   const int* gridSize = boxes->gridSize; // alias
   
   // Halo in Z+
   if (iz == gridSize[2])
   {
      iBox = boxes->nLocalBoxes + 2*gridSize[2]*gridSize[1] + 2*gridSize[2]*(gridSize[0]+2) +
         (gridSize[0]+2)*(gridSize[1]+2) + (gridSize[0]+2)*(iy+1) + (ix+1);
   }
   // Halo in Z-
   else if (iz == -1)
   {
      iBox = boxes->nLocalBoxes + 2*gridSize[2]*gridSize[1] + 2*gridSize[2]*(gridSize[0]+2) +
         (gridSize[0]+2)*(iy+1) + (ix+1);
   }
   // Halo in Y+
   else if (iy == gridSize[1])
   {
      iBox = boxes->nLocalBoxes + 2*gridSize[2]*gridSize[1] + gridSize[2]*(gridSize[0]+2) +
         (gridSize[0]+2)*iz + (ix+1);
   }
   // Halo in Y-
   else if (iy == -1)
   {
      iBox = boxes->nLocalBoxes + 2*gridSize[2]*gridSize[1] + iz*(gridSize[0]+2) + (ix+1);
   }
   // Halo in X+
   else if (ix == gridSize[0])
   {
      iBox = boxes->nLocalBoxes + gridSize[1]*gridSize[2] + iz*gridSize[1] + iy;
   }
   // Halo in X-
   else if (ix == -1)
   {
      iBox = boxes->nLocalBoxes + iz*gridSize[1] + iy;
   }
   // local link celll.
   else
   {
      iBox = boxes->boxIDLookUp[IDX3D(ix,iy,iz, gridSize[0],gridSize[1])];
   }
   assert(iBox >= 0);
   assert(iBox < boxes->nTotalBoxes);

   return iBox;
}

/// Move an atom from one link cell to another.
/// \param iId [in]  The index with box iBox of the atom to be moved.
/// \param iBox [in] The index of the link cell the particle is moving from.
/// \param jBox [in] The index of the link cell the particle is moving to.
void moveAtom(LinkCell* boxes, Atoms* atoms, int iId, int iBox, int jBox)
{
   int nj = boxes->nAtoms[jBox];
   copyAtom(boxes, atoms, iId, iBox, nj, jBox);
   boxes->nAtoms[jBox]++;

   assert(boxes->nAtoms[jBox] < MAXATOMS);

   boxes->nAtoms[iBox]--;
   int ni = boxes->nAtoms[iBox];
   if (ni) copyAtom(boxes, atoms, ni, iBox, iId, iBox);

   if (jBox > boxes->nLocalBoxes)
      --atoms->nLocal;
   
   return;
}

/// \details
/// This is the first step in returning data structures to a consistent
/// state after the atoms move each time step.  First we discard all
/// atoms in the halo link cells.  These are all atoms that are
/// currently stored on other ranks and so any information we have about
/// them is stale.  Next, we move any atoms that have crossed link cell
/// boundaries into their new link cells.  It is likely that some atoms
/// will be moved into halo link cells.  Since we have deleted halo
/// atoms from other tasks, it is clear that any atoms that are in halo
/// cells at the end of this routine have just transitioned from local
/// to halo atoms.  Such atom must be sent to other tasks by a halo
/// exchange to avoid being lost.
/// \see redistributeAtoms
void updateLinkCellsCpu(LinkCell* boxes, Atoms* atoms)
{
   emptyHaloCells(boxes);
   
   for (int iBox=0; iBox<boxes->nLocalBoxes; ++iBox)
   {
      int iOff = iBox*MAXATOMS;
      int ii=0;
      while (ii < boxes->nAtoms[iBox])
      {
         real3_old rr;
         rr[0] = atoms->r.x[iOff+ii];
         rr[1] = atoms->r.y[iOff+ii];
         rr[2] = atoms->r.z[iOff+ii];
         int jBox = getBoxFromCoord(boxes, rr);

         if (jBox != iBox)
            moveAtom(boxes, atoms, ii, iBox, jBox);
         else
            ++ii;
      }
   }
}

/// \return The largest number of atoms in any link cell.
int maxOccupancy(LinkCell* boxes)
{
   int localMax = 0;
   for (int ii=0; ii<boxes->nLocalBoxes; ++ii)
      localMax = MAX(localMax, boxes->nAtoms[ii]);

   int globalMax;

   startTimer(commReduceTimer);
   maxIntParallel(&localMax, &globalMax, 1);
   stopTimer(commReduceTimer);

   return globalMax;
}

/// Copy atom iAtom in link cell iBox to atom jAtom in link cell jBox.
/// Any data at jAtom, jBox is overwritten.  This routine can be used to
/// re-order atoms within a link cell.
void copyAtom(LinkCell* boxes, Atoms* atoms, int iAtom, int iBox, int jAtom, int jBox)
{
   const int iOff = MAXATOMS*iBox+iAtom;
   const int jOff = MAXATOMS*jBox+jAtom;
   atoms->gid[jOff] = atoms->gid[iOff];
   atoms->iSpecies[jOff] = atoms->iSpecies[iOff];
   copyVec(&(atoms->r), iOff, jOff);
   copyVec(&(atoms->p), iOff, jOff);
   copyVec(&(atoms->f), iOff, jOff); //TODO is this required?
   memcpy(atoms->U+jOff,  atoms->U+iOff,  sizeof(real_t));
}

/// Get the index of the link cell that contains the specified
/// coordinate.  This can be either a halo or a local link cell.
///
/// Because the rank ownership of an atom is strictly determined by the
/// atom's position, we need to take care that all ranks will agree which
/// rank owns an atom.  The conditionals at the end of this function are
/// special care to ensure that all ranks make compatible link cell
/// assignments for atoms that are near a link cell boundaries.  If no
/// ranks claim an atom in a local cell it will be lost.  If multiple
/// ranks claim an atom it will be duplicated.
int getBoxFromCoord(LinkCell* boxes, real_t rr[3])
{
   const real_t* localMin = boxes->localMin; // alias
   const real_t* localMax = boxes->localMax; // alias
   const int*    gridSize = boxes->gridSize; // alias
   int ix = (int)(floor((rr[0] - localMin[0])*boxes->invBoxSize[0]));
   int iy = (int)(floor((rr[1] - localMin[1])*boxes->invBoxSize[1]));
   int iz = (int)(floor((rr[2] - localMin[2])*boxes->invBoxSize[2]));


   // For each axis, if we are inside the local domain, make sure we get
   // a local link cell.  Otherwise, make sure we get a halo link cell.
   if (rr[0] < localMax[0]) 
   {
      if (ix == gridSize[0]) ix = gridSize[0] - 1;
   }
   else
      ix = gridSize[0]; // assign to halo cell
   if (rr[1] < localMax[1])
   {
      if (iy == gridSize[1]) iy = gridSize[1] - 1;
   }
   else
      iy = gridSize[1];
   if (rr[2] < localMax[2])
   {
      if (iz == gridSize[2]) iz = gridSize[2] - 1;
   }
   else
      iz = gridSize[2];
   
   return getBoxFromTuple(boxes, ix, iy, iz);
}

/// Set the number of atoms to zero in all halo link cells.
void emptyHaloCells(LinkCell* boxes)
{
   for (int ii=boxes->nLocalBoxes; ii<boxes->nTotalBoxes; ++ii)
      boxes->nAtoms[ii] = 0;
}

/// Get the grid coordinates of the link cell with index iBox.  Local
/// cells are easy as they use a standard 1D->3D mapping.  Halo cell are
/// special cases.
/// \see initLinkCells for information on link cell order.
/// \param [in]  iBox Index to link cell for which tuple is needed.
/// \param [out] ixp  x grid coord of link cell.
/// \param [out] iyp  y grid coord of link cell.
/// \param [out] izp  z grid coord of link cell.
void getTuple(LinkCell* boxes, int iBox, int* ixp, int* iyp, int* izp)
{
   int ix, iy, iz;
   const int* gridSize = boxes->gridSize; // alias
   
   // If a local box
   if( iBox < boxes->nLocalBoxes)
   {
      ix = boxes->boxIDLookUpReverse[iBox].x;
      iy = boxes->boxIDLookUpReverse[iBox].y;
      iz = boxes->boxIDLookUpReverse[iBox].z;
   }
   // It's a halo box
   else 
   {
      int ink;
      ink = iBox - boxes->nLocalBoxes;
      if (ink < 2*gridSize[1]*gridSize[2])
      {
         if (ink < gridSize[1]*gridSize[2]) 
         {
            ix = 0;
         }
         else 
         {
            ink -= gridSize[1]*gridSize[2];
            ix = gridSize[0] + 1;
         }
         iy = 1 + ink % gridSize[1];
         iz = 1 + ink / gridSize[1];
      }
      else if (ink < (2 * gridSize[2] * (gridSize[1] + gridSize[0] + 2))) 
      {
         ink -= 2 * gridSize[2] * gridSize[1];
         if (ink < ((gridSize[0] + 2) *gridSize[2])) 
         {
            iy = 0;
         }
         else 
         {
            ink -= (gridSize[0] + 2) * gridSize[2];
            iy = gridSize[1] + 1;
         }
         ix = ink % (gridSize[0] + 2);
         iz = 1 + ink / (gridSize[0] + 2);
      }
      else 
      {
         ink -= 2 * gridSize[2] * (gridSize[1] + gridSize[0] + 2);
         if (ink < ((gridSize[0] + 2) * (gridSize[1] + 2))) 
         {
            iz = 0;
         }
         else 
         {
            ink -= (gridSize[0] + 2) * (gridSize[1] + 2);
            iz = gridSize[2] + 1;
         }
         ix = ink % (gridSize[0] + 2);
         iy = ink / (gridSize[0] + 2);
      }
      
      // Calculated as off by 1
      ix--;
      iy--;
      iz--;
   }
   
   *ixp = ix;
   *iyp = iy;
   *izp = iz;
}

void emptyBoundaryCells(LinkCell* boxes, int *boundaryCellList, int nBoundaryCells)
{
   for (int i=0; i<nBoundaryCells; ++i)
           boxes->nAtoms[boundaryCellList[i]] = 0;
}

void updateGpuBoundaryCells(SimFlat *sim)
{
      for(int i = 0; i < sim->n_boundary1_cells; ++i)
      {
              int iBox = sim->boundary1_cells_h[i];
              int f_size = sim->boxes->nAtoms[iBox] * sizeof(real_t);
              int i_size = sim->boxes->nAtoms[iBox] * sizeof(int);
              for(int ii=0; ii<sim->boxes->nAtoms[iBox];++ii)
              {
                      int iOff = iBox*MAXATOMS + ii;
                      sim->host.atoms.r.x[iOff] = sim->atoms->r.x[iOff];
                      sim->host.atoms.r.y[iOff] = sim->atoms->r.y[iOff];
                      sim->host.atoms.r.z[iOff] = sim->atoms->r.z[iOff];

                      sim->host.atoms.p.x[iOff] = sim->atoms->p.x[iOff];
                      sim->host.atoms.p.y[iOff] = sim->atoms->p.y[iOff];
                      sim->host.atoms.p.z[iOff] = sim->atoms->p.z[iOff];

              }
              cudaMemcpy((sim->gpu.atoms.r.x)+(iBox*MAXATOMS), (sim->host.atoms.r.x)+(iBox*MAXATOMS), f_size, cudaMemcpyHostToDevice);
              cudaMemcpy((sim->gpu.atoms.r.y)+(iBox*MAXATOMS), (sim->host.atoms.r.y)+(iBox*MAXATOMS), f_size, cudaMemcpyHostToDevice);
              cudaMemcpy((sim->gpu.atoms.r.z)+(iBox*MAXATOMS), (sim->host.atoms.r.z)+(iBox*MAXATOMS), f_size, cudaMemcpyHostToDevice);
              cudaMemcpy((sim->gpu.atoms.p.x)+(iBox*MAXATOMS), (sim->host.atoms.p.x)+(iBox*MAXATOMS), f_size, cudaMemcpyHostToDevice);
              cudaMemcpy((sim->gpu.atoms.p.y)+(iBox*MAXATOMS), (sim->host.atoms.p.y)+(iBox*MAXATOMS), f_size, cudaMemcpyHostToDevice);
              cudaMemcpy((sim->gpu.atoms.p.z)+(iBox*MAXATOMS), (sim->host.atoms.p.z)+(iBox*MAXATOMS), f_size, cudaMemcpyHostToDevice);

              cudaMemcpy((sim->gpu.atoms.gid)+(iBox*MAXATOMS), (sim->atoms->gid)+(iBox*MAXATOMS), i_size, cudaMemcpyHostToDevice);
              cudaMemcpy((sim->gpu.atoms.iSpecies)+(iBox*MAXATOMS), (sim->atoms->iSpecies)+(iBox*MAXATOMS), i_size, cudaMemcpyHostToDevice);
      }
}
