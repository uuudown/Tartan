/*
 * PCG Random Number Generation for C.
 *
 * Copyright 2014 Melissa O'Neill <oneill@pcg-random.org>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * For additional information about the PCG random number generation scheme,
 * including its license and other licensing options, visit
 *
 *       http://www.pcg-random.org
 */

/*
 * This code is derived from the full C implementation, which is in turn
 * derived from the canonical C++ PCG implementation. The C++ version
 * has many additional features and is preferable if you can use C++ in
 * your project.
 */

/* simple GPU-based PCG by Cristobal A. Navarro, Feb 9, 2016 */
#ifndef CPCG_H
#define CPCG_H

#define INV_UINT_MAX 2.3283064e-10f
#define PCG_DEFAULT_MULTIPLIER_64  6364136223846793005ULL

#include <limits.h>
#include <inttypes.h>

__host__ __device__ inline uint32_t gpu_pcg32_random_r(uint64_t *state, uint64_t *inc);

__host__ __device__ inline void gpu_pcg32_srandom_r(uint64_t *state, uint64_t *inc, uint64_t initstate, uint64_t initseq){
    *state = 0U;
    *inc = (initseq << 1u) | 1u;
    gpu_pcg32_random_r(state, inc);
    *state += initstate;
    gpu_pcg32_random_r(state, inc);
}


__host__ __device__ inline uint32_t gpu_pcg32_random_r(uint64_t *state, uint64_t *inc){
    uint64_t oldstate = *state;
    *state = oldstate * 6364136223846793005ULL + *inc;
    uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
    uint32_t rot = oldstate >> 59u;
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
    //return ((xorshifted >> rot) | (xorshifted << ((-rot) & 31)))*INV_UINT_MAX;
}

__host__ __device__ inline float gpu_rand01(uint64_t *state, uint64_t *inc){
    //return (float) gpu_pcg32_random_r(state, inc) / (float)(UINT_MAX);
    return (float) gpu_pcg32_random_r(state, inc) * INV_UINT_MAX;
}

__host__ __device__ uint64_t pcg_advance(uint64_t state, uint64_t delta, uint64_t cur_mult,
                            uint64_t cur_plus)
{
    uint64_t acc_mult = 1u;
    uint64_t acc_plus = 0u;
    while (delta > 0) {
        if (delta & 1) {
            acc_mult *= cur_mult;
            acc_plus = acc_plus * cur_mult + cur_plus;
        }
        cur_plus = (cur_mult + 1) * cur_plus;
        cur_mult *= cur_mult;
        delta /= 2;
    }
    return acc_mult * state + acc_plus;
}

__host__ __device__ void pcg_skip_ahead(uint64_t *state, uint64_t *inc, uint64_t delta){
    *state = pcg_advance(*state, delta, PCG_DEFAULT_MULTIPLIER_64, *inc);
}

#endif

