/// \file
/// Initialize the atom configuration.

#ifndef __INIT_ATOMS_H
#define __INIT_ATOMS_H

#include "mytype.h"

struct SimFlatSt;
struct LinkCellSt;
struct NeighborListSt;

/// Atom data
typedef struct AtomsSt
{
   // atom-specific data
   int nLocal;    //!< total number of atoms on this processor
   int* lid;      //!< A locally unique id for each atom (used for the neighborlist)
   int nGlobal;   //!< total number of atoms in simulation

   int* gid;      //!< A globally unique id for each atom
   int* iSpecies; //!< the species index of the atom

   struct NeighborListSt* neighborList;

   vec_t r;     //!< positions
   vec_t p;     //!< momenta of atoms
   vec_t f;     //!< forces 
   real_t* U;     //!< potential energy per atom
} Atoms;


/// Allocates memory to store atom data.
Atoms* initAtoms(struct LinkCellSt* boxes, const real_t skinDistance);
void destroyAtoms(struct AtomsSt* atoms);

void createFccLattice(int nx, int ny, int nz, real_t lat, struct SimFlatSt* s);

void setVcm(struct SimFlatSt* s, real_t vcm[3]);
void setTemperature(struct SimFlatSt* s, real_t temperature);
void randomDisplacements(struct SimFlatSt* s, real_t delta);

/// Update the local id of each local particle. This is used by the neighbor-list
void atomsUpdateLocalId(struct LinkCellSt* boxes, Atoms* atoms);
#endif
