/// \file
/// Communicate halo data such as "ghost" atoms with neighboring tasks.

#ifndef __HALO_EXCHANGE_
#define __HALO_EXCHANGE_

#include "mytype.h"
#include <cuda_runtime.h>

struct AtomsSt;
struct LinkCellSt;
struct DomainSt;
struct HashTableSt;

/// Extra data members that are needed for the exchange of atom data.
/// For an atom exchange, the HaloExchangeSt::parms will point to a
/// structure of this type.
typedef struct AtomExchangeParmsSt
{
   int nCells[6];        //!< Number of cells in cellList for each face.
   int* cellList[6];     //!< List of link cells from which to load data for each face.
   int* cellListGpu[6];  //!< GPU list of link cells from which to load data for each face.
   int* d_natoms_buf;    // temp buf for scan result
   int* h_natoms_buf;    // temp buf for scan result
   int* d_partial_sums;  //!< device-pointer. partial sums for scan
   real_t* pbcFactor[6]; //!< Whether this face is a periodic boundary.
} AtomExchangeParms;

/// A structure to package data for a single atom to pack into a
/// send/recv buffer.  Also used for sorting atoms within link cells.
typedef struct AtomMsgSt
{
   int gid;
   int type;
   real_t rx, ry, rz;
   real_t px, py, pz;
} AtomMsg;

/// Same as AtomMsgSt but transposed (i.e. using structure of arrays)
typedef struct AtomMsgSoASt
{
   int* gid;
   int* type;
   real_t *rx, *ry, *rz;
   real_t *px, *py, *pz;
} AtomMsgSoA;

/// A polymorphic structure to store information about a halo exchange.
/// This structure can be thought of as an abstract base class that
/// specifies the interface and implements the communication patterns of
/// a halo exchange.  Concrete sub-classes supply actual implementations
/// of the loadBuffer, unloadBuffer, and destroy functions, that are
/// specific to the actual data being exchanged.  If the subclass needs
/// additional data members, these can be stored in a structure that is
/// pointed to by parms.
///
/// Designing the structure this way allows us to re-use the
/// communication code for both atom data and partial force data.
///
/// \see eamForce
/// \see redistributeAtoms
typedef struct HaloExchangeSt
{
   /// The MPI ranks of the six face neighbors of the local domain.
   /// Ranks are stored in the order specified in HaloFaceOrder.
   int nbrRank[6];
   /// The maximum send/recv buffer size (in bytes) that will be needed
   /// for this halo exchange.
   int bufCapacity;
   /// Pointer to a sub-class specific function to load the send buffer.
   /// \param [in] parms The parms member of the structure.  This is a
   ///                   pointer to a sub-class specific structure that can
   ///                   be used by the load and unload functions to store
   ///                   sub-class specific data.
   /// \param [in] data  A pointer to a structure that the contains the data
   ///                   that is needed by the loadBuffer function.  The
   ///                   loadBuffer function will cast the pointer to a
   ///                   concrete type that is appropriate for the data
   ///                   being exchanged.
   /// \param [in] face  Specifies the face across which data is being sent.
   /// \param [in] buf   The send buffer to be loaded
   /// \return The number of bytes loaded into the send buffer.
   int  (*loadBuffer)(void* parms, void* data, int face, char* buf);
   /// Pointer to a sub-class specific function to unload the recv buffer.
   /// \param [in] parms The parms member of the structure.  This is a
   ///                   pointer to a sub-class specific structure that can
   ///                   be used by the load and unload functions to store
   ///                   sub-class specific data.
   /// \param [out] data A pointer to a structure that the contains the data
   ///                   that is needed by the unloadBuffer function.  The
   ///                   unloadBuffer function will cast the pointer to a
   ///                   concrete type that is appropriate for the data
   ///                   being exchanged.
   /// \param [in] face  Specifies the face across which data is being sent.
   /// \param [in] bufSize The number of bytes in the recv buffer.
   /// \param [in] buf   The recv buffer to be unloaded.
   void (*unloadBuffer)(void* parms, void* data, int face, int bufSize, char* buf);
   /// Pointer to a function to deallocate any memory used by the
   /// sub-class parms.  Essentially this is a virtual destructor.
   void (*destroy)(void* parms);
   /// A pointer to a sub-class specific structure that contains
   /// additional data members needed by the sub-class.
   void* parms;
   /// Hash-table used to lookup the Offset of particle gid within the atoms structure. The hashTable is only used by the atom exchange.
   struct HashTableSt* hashTable;

   /// Type can be 0 for atom-exchange or 1 for force-exchange
   int type;

   // MPI buffers
   char *sendBufM;
   char *sendBufP;
   char *recvBufP;
   char *recvBufM;
} HaloExchange;

/// Create a HaloExchange for atom data.
HaloExchange* initAtomHaloExchange(struct DomainSt* domain, struct LinkCellSt* boxes);

/// Create a HaloExchange for force data.
HaloExchange* initForceHaloExchange(struct DomainSt* domain, struct LinkCellSt* boxes, int useCPU);

/// HaloExchange destructor.
void destroyHaloExchange(HaloExchange** haloExchange);

/// Execute a halo exchange.
void haloExchange(HaloExchange* haloExchange, void* data);

/// Sort the atoms by gid in the specified link cell.
void sortAtomsInCell(struct AtomsSt* atoms, struct LinkCellSt* boxes, int iBox);



/// Package data for the force exchange.
typedef struct ForceMsgSt
{
   real_t dfEmbed;
}
ForceMsg;

/// Extra data members that are needed for the exchange of force data.
/// For an force exchange, the HaloExchangeSt::parms will point to a
/// structure of this type.
typedef struct ForceExchangeParmsSt
{
   int nCells[6];     //!< Number of cells to send/recv for each face.
   int* sendCells[6]; //!< List of link cells to send for each face.
   int* sendCellsGpu[6]; //!< GPU List of link cells to send for each face.
   int* recvCells[6]; //!< List of link cells to recv for each face.
   int* recvCellsGpu[6]; //!< GPU List of link cells to recv for each face.
   int* natoms_buf[6];    // temp buf for scan result
   int* partial_sums[6];  // partial sums for scan
}
ForceExchangeParms;

#endif
