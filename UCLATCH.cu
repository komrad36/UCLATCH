/*******************************************************************
*   UCLATCH.cu
*   UCLATCH
*
*	Author: Kareem Omar
*	kareem.omar@uah.edu
*	https://github.com/komrad36
*
*	Last updated Sep 12, 2016
*******************************************************************/
//
// Fastest implementation of an UPRIGHT (no rotation)
// LATCH 512-bit binary feature descriptor
// as described in the 2015 paper by
// Levi and Hassner:
//
// "LATCH: Learned Arrangements of Three Patch Codes"
// http://arxiv.org/abs/1501.03719
//
// See also the ECCV 2016 Descriptor Workshop paper, of which I am a coauthor:
//
// "The CUDA LATCH Binary Descriptor"
// http://arxiv.org/abs/1609.03986
//
// And the original LATCH project's website:
// http://www.openu.ac.il/home/hassner/projects/LATCH/
//
// Note once again that this is an UPRIGHT CUDA LATCH, a.k.a. UCLATCH.
// A fast rotation- and scale-invariant version is
// also available on my GitHub.
//
// This implementation is insanely fast, matching or beating
// the much simpler ORB descriptor despite outputting twice
// as many bits AND being a superior descriptor.
//
// A key insight responsible for much of the performance of
// this laboriously crafted CUDA kernel is due to
// Christopher Parker (https://github.com/csp256) to whom
// I am extremely grateful.
//
// CUDA CC 3.0 or higher is required.
//
// All functionality is contained in the files UCLATCH.h
// and UCLATCH.cu. 'main.cpp' is simply a sample test harness
// with example usage and performance testing.
//

#include "UCLATCH.h"

__global__ void
#ifndef __INTELLISENSE__
__launch_bounds__(512, 4)
#endif
UCLATCH_kernel(const cudaTextureObject_t d_img_tex, const cudaTextureObject_t d_triplets, const cudaTextureObject_t d_kps, uint32_t* const __restrict__ d_desc) {
	volatile __shared__ uint8_t s_ROI[4608];
	const uint2 pt = tex1D<uint2>(d_kps, blockIdx.x);
	for (uint32_t i = 0; i <= 48; i += 16) for (uint32_t k = 0; k <= 32; k += 32) s_ROI[(threadIdx.y + i) * 72 + threadIdx.x + k] = tex2D<uint8_t>(d_img_tex, pt.x + threadIdx.x + k - 32, pt.y + threadIdx.y - 32 + i);
	const uint32_t ROI_base = 144 * (threadIdx.x & 3) + (threadIdx.x >> 2);
  uint32_t triplet_base = threadIdx.y << 5, desc = 0;
	__syncthreads();
	for (int32_t i = 0; i < 4; ++i, triplet_base += 8) {
		int32_t accum[8];
		for (uint32_t j = 0; j < 8; ++j) {
			const ushort4 t = tex1D<ushort4>(d_triplets, triplet_base + j);
			const int32_t b1 = s_ROI[ROI_base + t.y]     , b2 = s_ROI[ROI_base + t.y + 72]     ;
			const int32_t a1 = s_ROI[ROI_base + t.x] - b1, a2 = s_ROI[ROI_base + t.x + 72] - b2;
			const int32_t c1 = s_ROI[ROI_base + t.z] - b1, c2 = s_ROI[ROI_base + t.z + 72] - b2;
			accum[j] = a1 * a1 - c1 * c1 + a2 * a2 - c2 * c2;
		}
		for (int32_t k = 1; k <= 4; k <<= 1) {
			for (int32_t s = 0; s < 8; s += k) accum[s] += __shfl_xor(accum[s], k);
			if (threadIdx.x & k) for (int32_t s = 0; s < 8; s += k << 1) accum[s] = accum[s + k];
		}		
		accum[0] += __shfl_xor(accum[0], 8);
		desc |= (accum[0] + __shfl_xor(accum[0], 16) < 0) << ((i << 3) + (threadIdx.x & 7));
	}
	for (int32_t s = 1; s <= 4; s <<= 1) desc |= __shfl_xor(desc, s);
	if (threadIdx.x == 0) d_desc[(blockIdx.x << 4) + threadIdx.y] = desc;
}

void UCLATCH(const cudaTextureObject_t d_img_tex, const cudaTextureObject_t d_triplets, const cudaTextureObject_t d_kps, const int num_kps, uint64_t* const __restrict d_desc) {
	UCLATCH_kernel<<<num_kps, { 32, 16 }>>>(d_img_tex, d_triplets, d_kps, reinterpret_cast<uint32_t*>(d_desc));
	cudaDeviceSynchronize();
}
