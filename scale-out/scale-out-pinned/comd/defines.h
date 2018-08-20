
#ifndef __DEFINES_H_
#define __DEFINES_H_

#define HASHTABLE_FREE -1
#define BOUNDARY 1
#define INTERIOR 2
#define BOTH 0

//methods
#define THREAD_ATOM 0
#define THREAD_ATOM_NL 1
#define WARP_ATOM 2
#define WARP_ATOM_NL 3
#define CTA_CELL 4
//CPU method has to be the last
#define CPU_NL 5


#define ISPOWER2(v) ((v) && !((v) & ((v) - 1)))
            
#define IDX3D(x,y,z,X,Y) ((z)*((Y)*(X)) + ((y)*(X)) + (x))

/// The maximum number of atoms that can be stored in a link cell.
//Moved to the Makefile
//#define MAXATOMS 256 

#define WARP_SIZE		32

#define THREAD_ATOM_CTA         128
#define WARP_ATOM_CTA		128
#define CTA_CELL_CTA		128

// NOTE: the following is tuned for GK110
#ifdef COMD_DOUBLE
#define THREAD_ATOM_ACTIVE_CTAS 	10	// 62%
#define WARP_ATOM_ACTIVE_CTAS 		12	// 75%
#define CTA_CELL_ACTIVE_CTAS 		10	// 62%
#define WARP_ATOM_NL_CTAS            9  // 56%
#else
// 100% occupancy for SP
#define THREAD_ATOM_ACTIVE_CTAS 	16
#define WARP_ATOM_ACTIVE_CTAS 		16
#define CTA_CELL_ACTIVE_CTAS 		16
#define WARP_ATOM_NL_CTAS           16
#endif

//log_2(x)
#define LOG(X) _LOG( X )
#define _LOG(X) _LOG_ ## X

#define _LOG_32 5
#define _LOG_16 4
#define _LOG_8  3
#define _LOG_4  2
#define _LOG_2  1
#define _LOG_1  0

//Number of threads collaborating to make neighbor list for a single atom
#define NEIGHLIST_PACKSIZE 8
#define NEIGHLIST_PACKSIZE_LOG LOG(NEIGHLIST_PACKSIZE)
//Number of threads to compute forces of single atom in warp_atom_nl method
#define KERNEL_PACKSIZE 4

//Maximum size of neighbor list for a single atom
#define MAXNEIGHBORLISTSIZE 64

#define VECTOR_WIDTH 4

//size of shared memory used in cta_cell kernel for Lennard-Jones
//it can't be less than CTA_CELL_CTA
#define SHARED_SIZE_CTA_CELL 128 

//Number of atoms covered by a single entry of pairlist
//Cannot be bigger than 1024 (resulting in 32x32 blocks)
#define PAIRLIST_ATOMS_PER_INT 1024

#define PAIRLIST_STEP (PAIRLIST_ATOMS_PER_INT/32)

#endif
