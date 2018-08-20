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

#include "hashTable.h"
#include "memUtils.h"

#include <stdio.h>
#include <assert.h>

HashTable* initHashTable(int nMaxEntries)
{

   HashTable *hashTable = (HashTable *) comdMalloc(sizeof(HashTable));

   hashTable->nMaxEntries = nMaxEntries; 
   hashTable->nEntriesPut = 0; //allocates a 5MB hashtable. This number is prime.
   hashTable->nEntriesGet = 0; //allocates a 5MB hashtable. This number is prime.

   hashTable->offset = (int*) comdMalloc(sizeof(int) * hashTable->nMaxEntries);

   emptyHashTable(hashTable);

   return hashTable;
}

void destroyHashTable(HashTable** hashTable)
{
   if (! hashTable) return;
   if (! *hashTable) return;

   comdFree((*hashTable)->offset);
   comdFree(*hashTable);
   *hashTable = NULL;
}

/// \param [inout] hashTable HashTable were the (key,value) pair will be inserted
void hashTablePut(HashTable* hashTable, int iOff)
{
        assert(hashTable->nEntriesPut < hashTable->nMaxEntries);
        hashTable->offset[hashTable->nEntriesPut] = iOff;
        hashTable->nEntriesPut++;
}

/// \details
/// This function finds the offset of the particle with gid = key. However, since one particle
/// with the same gid can show up in the haloCells multiple times (due to PBC), we have to find the correct particle.
/// Therefor, we require its old position to be present in the r array. We can be sure that we found the correct
/// particle if its old position and the updated position only differ by less than the skinDistance**2 / 4.0 (one particle is not allowed to move that far within one iteration).
/// 
/// \param [in] hashTable hashTable 
/// \param [in] key gid of the particle we are looking for
/// \param [in] rx x-coordinate of the updated position of the particle we are looking for
/// \param [in] ry y-coordinate of the updated position of the particle we are looking for
/// \param [in] rz z-coordinate of the updated position of the particle we are looking for
/// \param [in] r pointer to the position array within the atoms structure (contains old positions for the haloCells) 
/// \param [in] skinDistance2 Squared skin-distance
/// \return returns offset within the atoms structure for particle with gid==key
int hashTableGet(HashTable* hashTable)
{
        int iOff = hashTable->offset[hashTable->nEntriesGet];
        hashTable->nEntriesGet++;

        return iOff;
}

void emptyHashTable(HashTable* hashTable)
{
   hashTable->nEntriesPut = 0;
}

void hashTableEnlarge(HashTable *hashTable)
{
        printf("hashTableEnlarge: TODO\n");
        exit(-1);
}

