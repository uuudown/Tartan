/// \file
/// Functions to maintain link cell structures for fast pair finding.

#ifndef __LINK_CELLS_H_
#define __LINK_CELLS_H_

#include "mytype.h"

struct DomainSt;
struct AtomsSt;
struct SimFlatSt;

/// Link cell data.  For convenience, we keep a copy of the localMin and
/// localMax coordinates that are also found in the DomainsSt.
typedef struct LinkCellSt
{
   int gridSize[3];     //!< number of boxes in each dimension on processor
   int nLocalBoxes;     //!< total number of local boxes on processor
   int nHaloBoxes;      //!< total number of remote halo/ghost boxes on processor
   int nTotalBoxes;     //!< total number of boxes on processor
                        //!< nLocalBoxes + nHaloBoxes
   real3_old localMin;      //!< minimum local bounds on processor
   real3_old localMax;      //!< maximum local bounds on processor
   real3_old boxSize;       //!< size of box in each dimension
   real3_old invBoxSize;    //!< inverse size of box in each dimension

   int *boxIDLookUp; //!< 3D array storing the box IDs 
   int3_t *boxIDLookUpReverse; //!< 1D array storing the tuple for a given box ID 

   int* nAtoms;         //!< total number of atoms in each box
} LinkCell;

LinkCell* initLinkCells(const struct DomainSt* domain, real_t cutoff, int useHilbert);
void destroyLinkCells(LinkCell** boxes);

int getNeighborBoxes(LinkCell* boxes, int iBox, int* nbrBoxes);
int putAtomInBox(LinkCell* boxes, struct AtomsSt* atoms,
                  const int gid, const int iType,
                  const real_t x,  const real_t y,  const real_t z,
                  const real_t px, const real_t py, const real_t pz);
void updateAtomInBoxAt(LinkCell* boxes, struct AtomsSt* atoms,
                  const int gid, const int iType,
                  const real_t x,  const real_t y,  const real_t z,
                  const real_t px, const real_t py, const real_t pz, const int iOff);
int getBoxFromTuple(LinkCell* boxes, int x, int y, int z);

void moveAtom(LinkCell* boxes, struct AtomsSt* atoms, int iId, int iBox, int jBox);

/// Update link cell data structures when the atoms have moved.
void updateLinkCellsCpu(LinkCell* boxes, struct AtomsSt* atoms);

int maxOccupancy(LinkCell* boxes);

int getBoxFromCoord(LinkCell* boxes, real_t rr[3]);

void emptyHaloCells(LinkCell* boxes);

/// updates the particles within the boundary cells. Requires that the particles are already present in the hashtable
void updateGpuBoundaryCells(struct SimFlatSt* sim);
#endif
