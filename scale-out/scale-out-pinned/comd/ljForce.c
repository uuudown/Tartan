/// \file
/// Computes forces for the 12-6 Lennard Jones (LJ) potential.
///
/// The Lennard-Jones model is not a good representation for the
/// bonding in copper, its use has been limited to constant volume
/// simulations where the embedding energy contribution to the cohesive
/// energy is not included in the two-body potential
///
/// The parameters here are taken from Wolf and Phillpot and fit to the
/// room temperature lattice constant and the bulk melt temperature
/// Ref: D. Wolf and S.Yip eds. Materials Interfaces (Chapman & Hall
///      1992) Page 230.
///
/// Notes on LJ:
///
/// http://en.wikipedia.org/wiki/Lennard_Jones_potential
///
/// The total inter-atomic potential energy in the LJ model is:
///
/// \f[
///   E_{tot} = \sum_{ij} U_{LJ}(r_{ij})
/// \f]
/// \f[
///   U_{LJ}(r_{ij}) = 4 \epsilon
///           \left\{ \left(\frac{\sigma}{r_{ij}}\right)^{12}
///           - \left(\frac{\sigma}{r_{ij}}\right)^6 \right\}
/// \f]
///
/// where \f$\epsilon\f$ and \f$\sigma\f$ are the material parameters in the potential.
///    - \f$\epsilon\f$ = well depth
///    - \f$\sigma\f$   = hard sphere diameter
///
///  To limit the interation range, the LJ potential is typically
///  truncated to zero at some cutoff distance. A common choice for the
///  cutoff distance is 2.5 * \f$\sigma\f$.
///  This implementation can optionally shift the potential slightly
///  upward so the value of the potential is zero at the cuotff
///  distance.  This shift has no effect on the particle dynamics.
///
///
/// The force on atom i is given by
///
/// \f[
///   F_i = -\nabla_i \sum_{jk} U_{LJ}(r_{jk})
/// \f]
///
/// where the subsrcipt i on the gradient operator indicates that the
/// derivatives are taken with respect to the coordinates of atom i.
/// Liberal use of the chain rule leads to the expression
///
/// \f{eqnarray*}{
///   F_i &=& - \sum_j U'_{LJ}(r_{ij})\hat{r}_{ij}\\
///       &=& \sum_j 24 \frac{\epsilon}{r_{ij}} \left\{ 2 \left(\frac{\sigma}{r_{ij}}\right)^{12}
///               - \left(\frac{\sigma}{r_{ij}}\right)^6 \right\} \hat{r}_{ij}
/// \f}
///
/// where \f$\hat{r}_{ij}\f$ is a unit vector in the direction from atom
/// i to atom j.
/// 
///

#include "defines.h"
#include "neighborList.h"
#include "mytype.h"
#include "ljForce.h"

#include <float.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

#include "constants.h"
#include "mytype.h"
#include "parallel.h"
#include "linkCells.h"
#include "memUtils.h"
#include "CoMDTypes.h"

#include "gpu_types.h"
#include "gpu_utility.h"
#include "gpu_kernels.h"

#define POT_SHIFT 1.0


static int ljForce(SimFlat* s);
int ljForceCpuNL(SimFlat* sim);
static void ljPrint(FILE* file, BasePotential* pot);

void ljDestroy(BasePotential** inppot)
{
   if ( ! inppot ) return;
   LjPotential* pot = (LjPotential*)(*inppot);
   if ( ! pot ) return;
   comdFree(pot);
   *inppot = NULL;

   return;
}

/// Initialize an Lennard Jones potential for Copper.
BasePotential* initLjPot(void)
{
   LjPotential *pot = (LjPotential*)comdMalloc(sizeof(LjPotential));
   pot->force = ljForce;
   pot->print = ljPrint;
   pot->destroy = ljDestroy;
   pot->sigma = 2.315;	                  // Angstrom
   pot->epsilon = 0.167;                  // eV
   pot->mass = 63.55 * amuToInternalMass; // Atomic Mass Units (amu)

   pot->lat = 3.615;                      // Equilibrium lattice const in Angs
   strcpy(pot->latticeType, "FCC");       // lattice type, i.e. FCC, BCC, etc.
   pot->cutoff = 5*pot->sigma;          // Potential cutoff in Angs

   strcpy(pot->name, "Cu");
   pot->atomicNo = 29;

   return (BasePotential*) pot;
}

void ljPrint(FILE* file, BasePotential* pot)
{
   LjPotential* ljPot = (LjPotential*) pot;
   fprintf(file, "  Potential type   : Lennard-Jones\n");
   fprintf(file, "  Species name     : %s\n", ljPot->name);
   fprintf(file, "  Atomic number    : %d\n", ljPot->atomicNo);
   fprintf(file, "  Mass             : "FMT1" amu\n", ljPot->mass / amuToInternalMass); // print in amu
   fprintf(file, "  Lattice Type     : %s\n", ljPot->latticeType);
   fprintf(file, "  Lattice spacing  : "FMT1" Angstroms\n", ljPot->lat);
   fprintf(file, "  Cutoff           : "FMT1" Angstroms\n", ljPot->cutoff);
   fprintf(file, "  Epsilon          : "FMT1" eV\n", ljPot->epsilon);
   fprintf(file, "  Sigma            : "FMT1" Angstroms\n", ljPot->sigma);
}

int ljForce(SimFlat* sim)
{
   if(sim->method == CPU_NL)
           ljForceCpuNL(sim);
   else
           ljForceGpu(&(sim->gpu), sim->ljInterpolation, sim->gpu.boxes.nLocalBoxes, NULL, sim->pot->cutoff + sim->skinDistance, sim->method);

   return 0;
}

int ljForceCpuNL(SimFlat* sim)
{
   LjPotential* pot = (LjPotential *) sim->pot;
   real_t sigma = pot->sigma;
   real_t epsilon = pot->epsilon;
   real_t rCut = pot->cutoff;
   real_t rCut2 = rCut*rCut;

   // zero forces and energy
   real_t ePot = 0.0;
   sim->ePotential = 0.0;
   int fSize = sim->boxes->nTotalBoxes*MAXATOMS;
   for (int ii=0; ii<fSize; ++ii)
   {
      sim->atoms->U[ii] = 0.;
   }
   zeroVecAll(&(sim->atoms->f), fSize);
   
   real_t s6 = sigma*sigma*sigma*sigma*sigma*sigma;

   real_t rCut6 = s6 / (rCut2*rCut2*rCut2);
   real_t eShift = POT_SHIFT * rCut6 * (rCut6 - 1.0);

   NeighborList* neighborList = sim->atoms->neighborList;
   real_t *rx = sim->atoms->r.x;
   real_t *ry = sim->atoms->r.y;
   real_t *rz = sim->atoms->r.z;

   real_t *fx = sim->atoms->f.x;
   real_t *fy = sim->atoms->f.y;
   real_t *fz = sim->atoms->f.z;

   real_t *U = sim->atoms->U;

   const int nLocalBoxes = sim->boxes->nLocalBoxes;
   const int nMaxLocalParticles = MAXATOMS*nLocalBoxes;
   int nbrBoxes[27];
   // loop over local boxes
   for (int iBox=0; iBox<nLocalBoxes; iBox++)
   {
           int nIBox = sim->boxes->nAtoms[iBox];
           // loop over atoms in iBox
           for (int iOff=MAXATOMS*iBox,ii=0; ii<nIBox; ii++,iOff++)
           {

                   int iLid = sim->atoms->lid[iOff];
                   real_t irx = rx[iOff];
                   real_t iry = ry[iOff];
                   real_t irz = rz[iOff];
                   real_t ifx = 0.0;
                   real_t ify = 0.0;
                   real_t ifz = 0.0;
                   real_t iU = 0.0;
                   assert(iLid < neighborList->nMaxLocal);
                   int* iNeighborList = &(neighborList->list[neighborList->maxNeighbors * iLid]);
                   const int nNeighbors = neighborList->nNeighbors[iLid];
                   // loop over atoms in neighborlist
                   for (int ij=0; ij<nNeighbors; ij++)
                   {
                           int jOff = iNeighborList[ij];

                           real_t drx = irx - rx[jOff];
                           real_t dry = iry - ry[jOff];
                           real_t drz = irz - rz[jOff];
                           real_t r2 = drx*drx + dry*dry + drz*drz;
#ifdef FULLLIST
                           if(jOff == iOff) r2 = FLT_MAX;
#endif


                           real_t fr = 0.0;
                           real_t eLocal = 0.0;
                           // Important note:
                           // from this point on r actually refers to 1.0/r
                           if ( r2 <= rCut2) 
                           {
                             r2 = 1.0/r2;
                             real_t r6 = (s6*r2) * (r2*r2);
                             eLocal = r6 * (r6 - 1.0) - eShift;
                             iU += eLocal;


                             // different formulation to avoid sqrt computation
                             fr = - 4.0*epsilon*r6*r2*(12.0*r6 - 6.0);

                             ifx -= drx*fr;
                             ify -= dry*fr;
                             ifz -= drz*fr;
                           }
#ifndef FULLLIST
                           U[jOff] += eLocal;
                           fx[jOff] += drx*fr;
                           fy[jOff] += dry*fr;
                           fz[jOff] += drz*fr;
#endif
                   } // loop over atoms in neighborlist
                   U[iOff] += iU;
                   fx[iOff] += ifx;
                   fy[iOff] += ify;
                   fz[iOff] += ifz;
           } // // loop over atoms in iBox
   } // loop over local boxes in system

   // loop over local boxes
   for (int iBox=0; iBox<nLocalBoxes; iBox++)
   {
           int nIBox = sim->boxes->nAtoms[iBox];
           // loop over atoms in iBox
           for (int iOff=MAXATOMS*iBox,ii=0; ii<nIBox; ii++,iOff++)
           {
              U[iOff] *= 0.5;
              // calculate energy contribution 
              ePot += U[iOff];
           }
   }
   ePot = ePot*4.0*epsilon;
   sim->ePotential = ePot;

   return 0;
}
//{
//   LjPotential* pot = (LjPotential *) sim->pot;
//   real_t sigma = pot->sigma;
//   real_t epsilon = pot->epsilon;
//   real_t rCut = pot->cutoff;
//   real_t rCut2 = rCut*rCut;
//
//   // zero forces and energy
//   real_t ePot = 0.0;
//   sim->ePotential = 0.0;
//   int fSize = sim->boxes->nTotalBoxes*MAXATOMS;
//   for (int ii=0; ii<fSize; ++ii)
//   {
//      sim->atoms->U[ii] = 0.;
//   }
//   zeroVecAll(&(sim->atoms->f), fSize);
//   
//   real_t s6 = sigma*sigma*sigma*sigma*sigma*sigma;
//
//   real_t rCut6 = s6 / (rCut2*rCut2*rCut2);
//   real_t eShift = POT_SHIFT * rCut6 * (rCut6 - 1.0);
//
//   real_t *rx = sim->atoms->r.x;
//   real_t *ry = sim->atoms->r.y;
//   real_t *rz = sim->atoms->r.z;
//   real_t *fx = sim->atoms->f.x;
//   real_t *fy = sim->atoms->f.y;
//   real_t *fz = sim->atoms->f.z;
//
//   NeighborList* neighborList = sim->atoms->neighborList;
//
//   int nbrBoxes[27];
//   // loop over local boxes
//   for (int iBox=0; iBox<sim->boxes->nLocalBoxes; iBox++)
//   {
//           int nIBox = sim->boxes->nAtoms[iBox];
//           // loop over atoms in iBox
//           for (int iOff=MAXATOMS*iBox,ii=0; ii<nIBox; ii++,iOff++)
//           {
//
//                   int iLid = sim->atoms->lid[iOff];
//                   assert(iLid < neighborList->nMaxLocal);
//                   int* iNeighborList = &(neighborList->list[neighborList->maxNeighbors * iLid]);
//                   const int nNeighbors = neighborList->nNeighbors[iLid];
//                   // loop over atoms in neighborlist
//                   for (int ij=0; ij<nNeighbors; ij++)
//                   {
//                           int jOff = iNeighborList[ij];
//
//                           real_t r2 = 0.0;
//                           real3_old dr;
//                           dr[0] = rx[iOff] - rx[jOff];
//                           r2+=dr[0]*dr[0];
//                           dr[1] = ry[iOff] - ry[jOff];
//                           r2+=dr[1]*dr[1];
//                           dr[2] = rz[iOff] - rz[jOff];
//                           r2+=dr[2]*dr[2];
//
//                           if ( r2 <= rCut2){ 
//
//                              // Important note:
//                              // from this point on r actually refers to 1.0/r
//                              r2 = 1.0/r2;
//                              real_t r6 = s6 * (r2*r2*r2);
//                              real_t eLocal = r6 * (r6 - 1.0) - eShift;
//                              sim->atoms->U[iOff] += 0.5*eLocal;
//                              sim->atoms->U[jOff] += 0.5*eLocal;
//
//                              // different formulation to avoid sqrt computation
//                              real_t fr = - 4.0*epsilon*r6*r2*(12.0*r6 - 6.0);
//
//                              fx[iOff] -= dr[0]*fr;
//                              fy[iOff] -= dr[1]*fr;
//                              fz[iOff] -= dr[2]*fr;
//                              fx[jOff] += dr[0]*fr;
//                              fy[jOff] += dr[1]*fr;
//                              fz[jOff] += dr[2]*fr;
//                           }
//                   } // loop over atoms in neighborlist
//           } // // loop over atoms in iBox
//   } // loop over local boxes in system
//
//   // loop over local boxes
//   for (int iBox=0; iBox<sim->boxes->nLocalBoxes; iBox++)
//   {
//           int nIBox = sim->boxes->nAtoms[iBox];
//           // loop over atoms in iBox
//           for (int iOff=MAXATOMS*iBox,ii=0; ii<nIBox; ii++,iOff++)
//              ePot += sim->atoms->U[iOff];
//   }
//
//   ePot = ePot*4.0*epsilon;
//   sim->ePotential = ePot;
//
//   return 0;
//}
