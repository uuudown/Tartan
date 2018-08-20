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

#ifndef _GPU_NEIGHBOR_LIST_H_
#define _GPU_NEIGHBOR_LIST_H_

#include "mytype.h"

struct NeighborListGpuSt;
struct LinkCellSt;
struct SimFlatSt;
struct AtomsSt;

/// Initialized the NeighborList data stucture
void initNeighborListGpu(struct SimGpuSt * sim, struct NeighborListGpuSt* neighborList, const int nLocalBoxes, const real_t skinDistance);

/// frees all data associated with *neighborList
void destroyNeighborListGpu(struct NeighborListGpuSt** neighborList);

///// Sets all neighbor counts to zero
//void emptyNeighborListGpu(NeighborListGpu* neighborList);
//
///// returns 1 iff this step will require a rebuild of the neighborlist
//int neighborListUpdateRequiredGpu(struct SimGpuSt* sim);

/// Forces a rebuild of the neighborlist (but does not rebuild the neighborlist yet, this requires a separate call).
void neighborListForceRebuildGpu(struct NeighborListGpuSt* neighborList);
#endif
