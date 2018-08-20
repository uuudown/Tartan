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
#ifndef _REDUCTION_H_
#define _REDUCTION_H_

/* warp reduction with shfl function */
template < typename T >
__inline__ __device__ float warp_reduce(T val){
	for (int offset = WARPSIZE >> 1; offset > 0; offset >>= 1)
		val += __shfl_down(val, offset);
	return val;
}

/* block reduction with warp reduction */
template < typename T >
__inline__ __device__ float block_reduce(T val){
	static __shared__ T shared[WARPSIZE];
	int tid = threadIdx.z * BY * BX + threadIdx.y * BX + threadIdx.x;
	int lane = tid & (WARPSIZE-1);
	int wid = tid/WARPSIZE;
	val = warp_reduce<T>(val);
	
	if(lane == 0)
		shared[wid] = val;

	__syncthreads();

	val = (tid < (blockDim.x * blockDim.y * blockDim.z)/WARPSIZE) ? shared[lane] : 0;
	if(wid == 0){
		val = warp_reduce<T>(val);
	}
	return val;
}

/* magnetization reduction using block reduction */
template <typename T>
__global__ void kernel_redmagnetization(int *s, int L, T *out){
	// offsets
	int x = blockIdx.x *blockDim.x + threadIdx.x;
	int y = blockIdx.y *blockDim.y + threadIdx.y;
	int z = blockIdx.z *blockDim.z + threadIdx.z;
	int tid = threadIdx.z * BY * BX + threadIdx.y * BX + threadIdx.x;
	int id = C(x,y,z,L);
	int sum = s[id];
	sum = block_reduce<T>(sum); 
	if(tid == 0){
		atomicAdd(out, sum);
	}
}


/* energy reduction using block reduction */
template <typename T>
__global__ void kernel_redenergy(int *s, int L, T *out, int *H, float h){
	// offsets
	int x = blockIdx.x *blockDim.x + threadIdx.x;
	int y = blockIdx.y *blockDim.y + threadIdx.y;
	int z = blockIdx.z *blockDim.z + threadIdx.z;
	int tid = threadIdx.z * BY * BX + threadIdx.y * BX + threadIdx.x;
	int id = C(x,y,z,L);
    // this optimization only works for L being a power of 2
	//float sum = -(float)(s[id] * ((float)(s[C((x+1) & (L-1), y, z, L)] + s[C(x, (y+1) & (L-1), z, L)] + s[C(x, y, (z+1) & (L-1), L)]) + h*H[id]));

    // this line works always
	float sum = -(float)(s[id] * ((float)(s[C((x+1) >=  L? 0: x+1, y, z, L)] + s[C(x, (y+1) >= L? 0 : y+1, z, L)] + s[C(x, y, (z+1) >= L? 0 : z+1, L)]) + h*H[id]));
	sum = block_reduce<T>(sum); 
	if(tid == 0){
		atomicAdd(out, sum);
	}
}


/* vector3 warp reduction with shfl function */
template < typename T >
__inline__ __device__ float3 warp_reduce3(T val){
	for (int offset = WARPSIZE >> 1; offset > 0; offset >>= 1){
		val.x += __shfl_down(val.x, offset);
		val.y += __shfl_down(val.y, offset);
		val.z += __shfl_down(val.z, offset);
	};
	return val;
}

/* vector3 block reduction with warp reduction */
template < typename T >
__inline__ __device__ T block_reduce3(T val, int tid){
	static __shared__ T shared[WARPSIZE];
	int lane = tid & (WARPSIZE-1);
	int wid = tid/WARPSIZE;
	val = warp_reduce3<T>(val);

	/* save the partial reduction if it thread is in lane 0 */
	if(lane == 0)
		shared[wid] = val;

	/* sync threads in block */
	__syncthreads();

	/* save the partial reductions on the first |B| / WARPSIZE threads */
	val = (tid < (blockDim.x * blockDim.y * blockDim.z)/WARPSIZE) ? shared[lane] : make_float3(0.0f,0.0f,0.0f);
	if(wid == 0){
		/* reduction of partial reductions */
		val = warp_reduce3<T>(val);
	}
	return val;
}

__global__ void kernel_redcorrlen(int *s, int L, float3 *F1, float3 *F2){
	/* 3D coords */
	int x = blockIdx.x *blockDim.x + threadIdx.x;
	int y = blockIdx.y *blockDim.y + threadIdx.y;
	int z = blockIdx.z *blockDim.z + threadIdx.z;

	/* local thread id */
	int tid = threadIdx.z*BY*BX + threadIdx.y*BX + threadIdx.x;

	/* global thread id */
	int id = C(x,y,z,L);

	/* k value */
	float k = 2.0f*PI/(float)L;

	/* q value */
	float q = (float)s[id];

	/* vectorized reduction element */
	float3 ver1 = make_float3(q * __cosf(k * (float)x), q * __cosf(k * (float)y), q * __cosf(k * (float)z) );
	float3 ver2 = make_float3(q * __sinf(k * (float)x), q * __sinf(k * (float)y), q * __sinf(k * (float)z) );

	/* block reduction */
	ver1 = block_reduce3<float3>(ver1, tid); 
	ver2 = block_reduce3<float3>(ver2, tid); 
	if(tid == 0){
		/* reduce the x parts */
		atomicAdd(&(F1->x), ver1.x);
		atomicAdd(&(F2->x), ver2.x);

		/* reduce the y parts */
		atomicAdd(&(F1->y), ver1.y);
		atomicAdd(&(F2->y), ver2.y);

		/* reduce the z parts */
		atomicAdd(&(F1->z), ver1.z);
		atomicAdd(&(F2->z), ver2.z);
	}
}

/* host side redenergy call */
void redenergy(setup_t *s, int tid, int a, int b, int k){
	/* define space of computation */
	dim3 block(BX, BY, BZ);
	dim3 grid((s->L + BX - 1)/BX, (s->L + BY - 1)/BY,  (s->L + BZ - 1)/BZ);
	/* launch reduction kernel */
	kernel_redenergy<float><<<grid, block, 0, s->rstream[k]>>>(s->dlat[k], s->L, s->dE[tid] + k - a, s->dH[tid], s->h);
	cudaDeviceSynchronize();
	cudaCheckErrors("redenergy");
}

/* host side redenergy call */
void adapt_redenergy(setup_t *s, int tid, int k){
	/* define space of computation */
	dim3 block(BX, BY, BZ);
	dim3 grid((s->L + BX - 1)/BX, (s->L + BY - 1)/BY,  (s->L + BZ - 1)/BZ);
	/* launch reduction kernel */
	kernel_redenergy<float><<<grid, block, 0, s->arstream[tid][k]>>>(s->mdlat[tid][k], s->L, s->dE[tid] + k, s->dH[tid], s->h);
	cudaDeviceSynchronize();
	cudaCheckErrors("adapt_redenergy");

}

/* host side redmagnetization call */
void redmagnetization(setup_t *s, int tid, int a, int b, int k){
	/* define space of computation */
	dim3 block(BX, BY, BZ);
	dim3 grid((s->L + BX - 1)/BX, (s->L + BY - 1)/BY,  (s->L + BZ - 1)/BZ);
	/* launch reduction kernel */
	kernel_redmagnetization<int><<<grid, block, 0, s->rstream[k]>>>(s->dlat[k], s->L, s->dM[tid] + k - a);
	cudaDeviceSynchronize();
	cudaCheckErrors("redmagnetization");
}

/* host side redcorrlen call */
void redcorrlen(setup_t *s, int tid, int a, int b, int k){
	/* define space of computation */
	dim3 block(BX, BY, BZ);
	dim3 grid((s->L + BX - 1)/BX, (s->L + BY - 1)/BY,  (s->L + BZ - 1)/BZ);
	/* launch reduction kernel */
	kernel_redcorrlen<<<grid, block, 0, s->rstream[k]>>>(s->dlat[k], s->L, s->dF1[tid] + k - a, s->dF2[tid] + k - a);
	cudaDeviceSynchronize();
	cudaCheckErrors("redcorrlen");
}
#endif
