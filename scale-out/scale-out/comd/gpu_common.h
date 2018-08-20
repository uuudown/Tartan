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

#ifndef __GPU_COMMON_H_
#define __GPU_COMMON_H_

#include "defines.h"
#include <cuda.h>

__device__ __forceinline__ float sqrt_approx(float f)
{
    float ret;
    asm ("sqrt.approx.ftz.f32 %0, %1;" : "=f"(ret) : "f"(f));
    return ret;
}

/// Interpolate a table to determine f(r) and its derivative f'(r).
///
/// \param [in] table Interpolation table.
/// \param [in] r Point where function value is needed.
/// \param [out] f The interpolated value of f(r).
/// \param [out] df The interpolatedi value of df(r)/dr.
__inline__ __device__
void interpolate(InterpolationObjectGpu table, real_t r, real_t &f, real_t &df)
{
   const real_t* tt = table.values; // alias

   // check boundaries
   r = max(r, table.x0);
   r = min(r, table.xn);
    
   // compute index
   r = r * table.invDx - table.invDxXx0;
   
   real_t ri = floor(r);
   
   int ii = (int)ri;
   assert(ii < table.n );

   // reset r to fractional distance
   r = r - ri;
   
    // using LDG on Kepler only
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350  
    real_t v0 = __ldg(tt + ii);
    real_t v1 = __ldg(tt + ii + 1);
    real_t v2 = __ldg(tt + ii + 2);
    real_t v3 = __ldg(tt + ii + 3);
#else
    real_t v0 = tt[ii];
    real_t v1 = tt[ii + 1];
    real_t v2 = tt[ii + 2];
    real_t v3 = tt[ii + 3];
#endif
   
   real_t g1 = v2 - v0;
   real_t g2 = v3 - v1;

     f = v1 + 0.5 * r * (g1 + r * (v2 + v0 - 2.0 * v1));
     df = (g1 + r * (g2 - g1)) * table.invDxHalf;
}

/// Interpolate using spline coefficients table 
/// to determine f(r) and its derivative f'(r).
///
/// \param [in] table Table with spline coefficients.
/// \param [in] r2 Square of point where function value is needed.
/// \param [out] f The interpolated value of f(r).
/// \param [out] df The interpolated value of 1/r*df(r)/dr.
__inline__ __device__
void interpolateSpline(InterpolationSplineObjectGpu table, real_t r2, real_t &f, real_t &df)
{
   const real_t* tt = table.coefficients; // alias

   float r = sqrt_approx(r2);

   // check boundaries
   r = max(r, table.x0);
   r = min(r, table.xn);
    
   // compute index
   r = r * table.invDx - table.invDxXx0;
   
   real_t ri = floor(r);
   
   int ii = 4*(int)ri;

    // using LDG on Kepler only
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350  
    real_t a = __ldg(tt + ii);
    real_t b = __ldg(tt + ii + 1);
    real_t c = __ldg(tt + ii + 2);
    real_t d = __ldg(tt + ii + 3);
#else
    real_t a = tt[ii];
    real_t b = tt[ii + 1];
    real_t c = tt[ii + 2];
    real_t d = tt[ii + 3];
#endif
   
     real_t tmp = a*r2+b;
     f = (tmp*r2+c)*r2+d;
     df =2*((3*tmp-b)*r2+c);
}


#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 300
// emulate shuffles through shared memory for old devices (SLOW)
__device__ __forceinline__
double __shfl_xor(double var, int laneMask, double *smem)
{
  smem[threadIdx.x] = var;
  const int warp_id = threadIdx.x / WARP_SIZE;
  const int lane_id = threadIdx.x % WARP_SIZE;
  return laneMask >= 0 && laneMask < WARP_SIZE ? smem[WARP_SIZE * warp_id + (lane_id ^ laneMask)] : var;
}

__device__ __forceinline__
float __shfl_xor(float var, int laneMask, float *smem)
{
  smem[threadIdx.x] = var;
  const int warp_id = threadIdx.x / WARP_SIZE;
  const int lane_id = threadIdx.x % WARP_SIZE;
  return laneMask >= 0 && laneMask < WARP_SIZE ? smem[WARP_SIZE * warp_id + (lane_id ^ laneMask)] : var;
}

__device__ __forceinline__
int __shfl_up(int var, unsigned int delta, int width, int *smem)
{
  smem[threadIdx.x] = var;
  const int warp_id = threadIdx.x / width;
  const int lane_id = threadIdx.x % width;
  return lane_id >= delta ? smem[width * warp_id + (lane_id - delta)] : var;
}

__device__ __forceinline__
double __shfl(double var, int laneMask, volatile double *smem)
{
  smem[threadIdx.x] = var;
  const int warp_id = threadIdx.x / WARP_SIZE;
  return laneMask >= 0 && laneMask < WARP_SIZE ? smem[WARP_SIZE * warp_id + laneMask] : var;
}

__device__ __forceinline__
float __shfl(float var, int laneMask, volatile float *smem)
{
  smem[threadIdx.x] = var;
  const int warp_id = threadIdx.x / WARP_SIZE;
  return laneMask >= 0 && laneMask < WARP_SIZE ? smem[WARP_SIZE * warp_id + laneMask] : var;
}

__device__ __forceinline__
int __shfl(int var, int laneMask, volatile int *smem)
{
  smem[threadIdx.x] = var;
  const int warp_id = threadIdx.x / WARP_SIZE;
  return laneMask >= 0 && laneMask < WARP_SIZE ? smem[WARP_SIZE * warp_id + laneMask] : var;
}

__device__ __forceinline__
int __shfl_down(real_t var, unsigned int delta, int width, real_t *smem)
{
  smem[threadIdx.x] = var;
  const int warp_id = threadIdx.x / width;
  const int lane_id = threadIdx.x % width;
  return lane_id < width - delta ? smem[width * warp_id + (lane_id + delta)] : var;
}


#else	// >= SM 3.0

#if (CUDA_VERSION < 6050)
__device__ __forceinline__
double __shfl_xor(double var, int laneMask)
{
  int lo = __shfl_xor( __double2loint(var), laneMask );
  int hi = __shfl_xor( __double2hiint(var), laneMask );
  return __hiloint2double( hi, lo );
}

__device__ __forceinline__
double __shfl(double var, int laneMask)
{
  int lo = __shfl( __double2loint(var), laneMask );
  int hi = __shfl( __double2hiint(var), laneMask );
  return __hiloint2double( hi, lo );
}

__device__ __forceinline__
double __shfl_down(double var, unsigned int delta, int width = 32)
{
    int lo, hi;
    asm volatile("mov.b64 {%0,%1}, %2;":"=r"(lo), "=r"(hi):"d"(var));
    lo = __shfl_down(lo, delta, width);
    hi = __shfl_down(hi, delta, width);
    return __hiloint2double(hi, lo);
}
#endif
#endif

#ifndef uint
typedef unsigned int uint;
#endif

// insert the first numBits of y into x starting at bit
__device__ uint bfi(uint x, uint y, uint bit, uint numBits) {
  uint ret;
  asm("bfi.b32 %0, %1, %2, %3, %4;" :
      "=r"(ret) : "r"(y), "r"(x), "r"(bit), "r"(numBits));
  return ret;
}

__device__ __forceinline__
void warp_reduce(real_t &x, real_t *smem)
{
  int lane_id = threadIdx.x % WARP_SIZE;
  smem[threadIdx.x] = x;
  // technically we also need warp sync here
  if (lane_id < 16) smem[threadIdx.x] += smem[threadIdx.x + 16];
  if (lane_id < 8) smem[threadIdx.x] += smem[threadIdx.x + 8];
  if (lane_id < 4) smem[threadIdx.x] += smem[threadIdx.x + 4];
  if (lane_id < 2) smem[threadIdx.x] += smem[threadIdx.x + 2];
  if (lane_id < 1) smem[threadIdx.x] += smem[threadIdx.x + 1];
  x = smem[threadIdx.x];
}

template<int step>
__device__ __forceinline__
void warp_reduce(real_t &ifx, real_t &ify, real_t &ifz, real_t &ie, real_t &irho)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
  // warp reduction
  for (int i = WARP_SIZE / 2; i > 0; i /= 2) {
    ifx += __shfl_xor(ifx, i);
    ify += __shfl_xor(ify, i);
    ifz += __shfl_xor(ifz, i);
    if (step == 1) {
      ie += __shfl_xor(ie, i);
      irho += __shfl_xor(irho, i);
    }
  }
#else
  // reduction using shared memory
  __shared__ real_t smem[WARP_ATOM_CTA];
  warp_reduce(ifx, smem);
  warp_reduce(ify, smem);
  warp_reduce(ifz, smem);
  if (step == 1) {
    warp_reduce(ie, smem);
    warp_reduce(irho, smem);
  }
#endif
}

// emulate atomic add for doubles
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
__device__ inline void atomicAdd(double *address, double value)
{
  unsigned long long oldval, newval, readback;
  oldval = __double_as_longlong(*address);
  newval = __double_as_longlong(__longlong_as_double(oldval) + value);
  while ((readback = atomicCAS((unsigned long long*)address, oldval, newval)) != oldval)
  {
    oldval = readback;
    newval = __double_as_longlong(__longlong_as_double(oldval) + value);
  }
}
#endif

static __device__ __forceinline__ int get_warp_id()
{
  return threadIdx.x >> 5;
}

static __device__ __forceinline__ int get_lane_id()
{
  int id;
  asm( "mov.u32 %0, %%laneid;" : "=r"(id) );
  return id;
}

// optimized version of DP rsqrt(a) provided by Norbert Juffa
__device__
double fast_rsqrt(double a)
{
  double x, e, t;
  float f;
  asm ("cvt.rn.f32.f64       %0, %1;" : "=f"(f) : "d"(a));
  asm ("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(f) : "f"(f));
  asm ("cvt.f64.f32          %0, %1;" : "=d"(x) : "f"(f));
  t = __dmul_rn (x, x);
  e = __fma_rn (a, -t, 1.0);
  t = __fma_rn (0.375, e, 0.5);
  e = __dmul_rn (e, x);
  x = __fma_rn (t, e, x);
  return x;
}

__device__
float fast_rsqrt(float a)
{
  return rsqrtf(a);
}

// optimized version of sqrt(a)
template<class real>
__device__
real sqrt_opt(real a)
{
#if 1
  return a * fast_rsqrt(a);
#else
  return sqrt(a);
#endif
}

#endif
