//////////////////////////////////////////////////////////////////////////////////
//                                                                              //
//  trueke                                                                      //
//  A multi-GPU implementation of the exchange Monte Carlo method.              //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
//                                                                              //
//  Copyright © 2015 Cristobal A. Navarro, Wei Huang.                           //
//                                                                              //
//  This file is part of trueke.                                                //
//  trueke is free software: you can redistribute it and/or modify              //
//  it under the terms of the GNU General Public License as published by        //
//  the Free Software Foundation, either version 3 of the License, or           //
//  (at your option) any later version.                                         //
//                                                                              //
//  trueke is distributed in the hope that it will be useful,                   //
//  but WITHOUT ANY WARRANTY; without even the implied warranty of              //
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the               //
//  GNU General Public License for more details.                                //
//                                                                              //
//  You should have received a copy of the GNU General Public License           //
//  along with trueke.  If not, see <http://www.gnu.org/licenses/>.             //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
#ifndef _KERNEL_PRNG_SETUP_
#define _KERNEL_PRNG_SETUP_

#include <stdio.h>

// Murmur hash 64-bit
__device__ uint64_t mmhash64( const void * key, int len, unsigned int seed ){
	const uint64_t m = 0xc6a4a7935bd1e995;
	const int r = 47;

	uint64_t h = seed ^ (len * m);

	const uint64_t * data = (const uint64_t *)key;
	const uint64_t * end = data + (len/8);

	while(data != end){
		uint64_t k = *data++;

		k *= m; 
		k ^= k >> r; 
		k *= m; 

		h ^= k;
		h *= m; 
	}
	const unsigned char * data2 = (const unsigned char*)data;
	switch(len & 7)
	{
		case 7: h ^= uint64_t(data2[6]) << 48;
		case 6: h ^= uint64_t(data2[5]) << 40;
		case 5: h ^= uint64_t(data2[4]) << 32;
		case 4: h ^= uint64_t(data2[3]) << 24;
		case 3: h ^= uint64_t(data2[2]) << 16;
		case 2: h ^= uint64_t(data2[1]) << 8;
		case 1: h ^= uint64_t(data2[0]);
		h *= m;
	};

	h ^= h >> r;
	h *= m;
	h ^= h >> r;
	return h;
} 

__global__ void kernel_prng_setup(curandState *state, int N, unsigned long long seed, unsigned long long seq){
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	/* Each thread gets same seed, a different sequence number, no offset */
	if( x < N ){
		/* medium quality, faster */
		curand_init(seed + (unsigned long long)x, seq, 0ULL, &state[x]);
		//curand_init(0, 0, 0, &state[x]);
		/* high quality, slower */
		//curand_init(seed, x, 0, &state[x]);
	}
}

__global__ void kernel_gpupcg_setup(uint64_t *state, uint64_t *inc, int N, unsigned long long seed, unsigned long long seq){
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	if( x < N ){
        // exclusive seeds, per replica sequences 
		unsigned long long tseed = x + seed;
		unsigned long long hseed = mmhash64(&tseed, sizeof(unsigned long long), 17);
		unsigned long long hseq = mmhash64(&seq, sizeof(unsigned long long), 47);
		//unsigned long long hseq = seq;
        gpu_pcg32_srandom_r(&state[x], &inc[x], hseed, hseq);

        // exclusive seeds common seq
        //gpu_pcg32_srandom_r(&state[x], &inc[x], x + seed + seq, 1);

        // vary seeds accross lattice positions
        //gpu_pcg32_srandom_r(&(state[x]), &(inc[x]), x + seed, seq);

        // one unique sequence for each thread in the multi-GPU system
        //gpu_pcg32_srandom_r(&(state[x]), &(inc[x]), seed, seq + x);

	}
}
#endif
