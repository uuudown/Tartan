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
/// HashTable implementation used for haloExchange.

#ifndef __HASH_TABLE_H
#define __HASH_TABLE_H

#include "mytype.h"


typedef struct HashTableSt
{
   int *offset;  //!< Stores the offsets for each received particles. (e.g. if the first particle will be stored to iOff=133, then offset[0] = 133)
   int nEntriesPut; //!< Number of stored particles in the offset array.
   int nEntriesGet; //!< Number of particles that have already been read from the offset array. (only used by hashtableGet)
   int nMaxEntries; //!< Size of the offset array
} HashTable;

/// allocates and initializes the hashTable
HashTable* initHashTable(int nMaxEntries);

/// frees all data associated with *hashTable
void destroyHashTable(HashTable** hashTable);

/// inserts (key,value) pair into the hashTable
void hashTablePut(HashTable* hashTable, int iOff);

/// get the value of the hash table associated with the key
int hashTableGet(HashTable* hashTable);

/// clears all entries in the hashTable
void emptyHashTable(HashTable* hashTable);

/// resizes the hashtable to a larger value
void hashTableEnlarge(HashTable *hashTable);
         
#endif
