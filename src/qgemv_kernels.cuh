// Quantized GEMV kernels for the batch-1 LLM decode path.
//
// One operation: y = W * x, where W is a quantized [M, K] weight matrix and x is
// an fp32 activation vector of length K. This is the operation that dominates
// batch-1 decode, and it is memory bound on every GPU ever built.
//
// The block layouts are byte-identical to the ggml block_q8_0 and block_q4_0
// structs, so a row of W here is a row of W in a GGUF file. That is deliberate:
// the point of the AoS-vs-SoA comparison below is to measure what the on-disk
// layout costs at the kernel level.

#pragma once
#include <cuda_fp16.h>
#include <cstdint>

#define QK 32  // ggml quantization block size, both formats

// ---------------------------------------------------------------- layouts

// 34 bytes. Note it is not a multiple of 4, let alone 16.
struct block_q8_0 { __half d; int8_t qs[QK]; };
// 18 bytes. Same problem.
struct block_q4_0 { __half d; uint8_t qs[QK / 2]; };

static_assert(sizeof(block_q8_0) == 34, "block_q8_0 must match ggml");
static_assert(sizeof(block_q4_0) == 18, "block_q4_0 must match ggml");

__device__ __forceinline__ float warp_reduce(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
  return v;
}

// ================================================================ Q8_0

// v0: one thread per row, ggml AoS layout, scalar loop.
// Adjacent threads are (K/32)*34 bytes apart, so every load is a separate
// memory transaction. This is the baseline, and it is meant to be bad.
__global__ void qgemv_q8_v0_naive(const block_q8_0* __restrict__ W,
                                  const float* __restrict__ x,
                                  float* __restrict__ y, int M, int K) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= M) return;
  const int nb = K / QK;
  const block_q8_0* w = W + (size_t)row * nb;
  float acc = 0.f;
  for (int b = 0; b < nb; ++b) {
    float s = 0.f;
    for (int j = 0; j < QK; ++j) s += (float)w[b].qs[j] * x[b * QK + j];
    acc += __half2float(w[b].d) * s;
  }
  y[row] = acc;
}

// v1: one warp per row, ggml AoS layout. Lane l takes blocks l, l+32, l+64...
// so in each iteration the warp reads 32*34 = 1088 contiguous bytes. The
// accesses coalesce in aggregate even though no single lane load is wide.
__global__ void qgemv_q8_v1_warp(const block_q8_0* __restrict__ W,
                                 const float* __restrict__ x,
                                 float* __restrict__ y, int M, int K) {
  const int warps = blockDim.x / 32;
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.x * warps + (threadIdx.x >> 5);
  if (row >= M) return;
  const int nb = K / QK;
  const block_q8_0* w = W + (size_t)row * nb;
  float acc = 0.f;
  for (int b = lane; b < nb; b += 32) {
    const block_q8_0 blk = w[b];
    float s = 0.f;
#pragma unroll
    for (int j = 0; j < QK; ++j) s += (float)blk.qs[j] * x[b * QK + j];
    acc += __half2float(blk.d) * s;
  }
  acc = warp_reduce(acc);
  if (lane == 0) y[row] = acc;
}

// v2: as v1, but the activation vector is staged in shared memory once per
// block and reused by every row the block owns. Weight traffic is unchanged;
// this only tests whether x traffic was costing anything.
__global__ void qgemv_q8_v2_smem_x(const block_q8_0* __restrict__ W,
                                   const float* __restrict__ x,
                                   float* __restrict__ y, int M, int K) {
  extern __shared__ float sx[];
  for (int i = threadIdx.x; i < K; i += blockDim.x) sx[i] = x[i];
  __syncthreads();
  const int warps = blockDim.x / 32;
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.x * warps + (threadIdx.x >> 5);
  if (row >= M) return;
  const int nb = K / QK;
  const block_q8_0* w = W + (size_t)row * nb;
  float acc = 0.f;
  for (int b = lane; b < nb; b += 32) {
    const block_q8_0 blk = w[b];
    float s = 0.f;
#pragma unroll
    for (int j = 0; j < QK; ++j) s += (float)blk.qs[j] * sx[b * QK + j];
    acc += __half2float(blk.d) * s;
  }
  acc = warp_reduce(acc);
  if (lane == 0) y[row] = acc;
}

// v3: SoA. Quants live in one contiguous int8 plane, scales in another.
// Each lane now issues a 16-byte load, so the warp reads 512 aligned
// contiguous bytes per instruction. This is a layout change, not a scheduling
// change: v1 and v3 do the same arithmetic in the same order.
__global__ void qgemv_q8_v3_soa_vec(const int8_t* __restrict__ Q,
                                    const __half* __restrict__ D,
                                    const float* __restrict__ x,
                                    float* __restrict__ y, int M, int K) {
  const int warps = blockDim.x / 32;
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.x * warps + (threadIdx.x >> 5);
  if (row >= M) return;
  const int nb = K / QK;
  const int4* q4 = reinterpret_cast<const int4*>(Q + (size_t)row * K);
  const __half* d = D + (size_t)row * nb;
  const int nvec = K / 16;  // 16 int8 per lane load
  float acc = 0.f;
  for (int v = lane; v < nvec; v += 32) {
    int4 raw = q4[v];
    const int8_t* q = reinterpret_cast<const int8_t*>(&raw);
    const int base = v * 16;
    float s = 0.f;
#pragma unroll
    for (int j = 0; j < 16; ++j) s += (float)q[j] * x[base + j];
    acc += __half2float(d[base / QK]) * s;
  }
  acc = warp_reduce(acc);
  if (lane == 0) y[row] = acc;
}

// v4: SoA with WPR warps cooperating on one row, so that a small M still fills
// the machine. Partial sums are reduced through shared memory.
template <int WPR>
__global__ void qgemv_q8_v4_split(const int8_t* __restrict__ Q,
                                  const __half* __restrict__ D,
                                  const float* __restrict__ x,
                                  float* __restrict__ y, int M, int K) {
  extern __shared__ float part[];
  const int lane = threadIdx.x & 31;
  const int wid = threadIdx.x >> 5;  // warp id inside the block
  const int rows_per_block = blockDim.x / (32 * WPR);
  const int row_local = wid / WPR;
  const int sub = wid % WPR;  // which warp of this row
  const int row = blockIdx.x * rows_per_block + row_local;
  const int nb = K / QK;
  const int nvec = K / 16;
  float acc = 0.f;
  if (row < M) {
    const int4* q4 = reinterpret_cast<const int4*>(Q + (size_t)row * K);
    const __half* d = D + (size_t)row * nb;
    for (int v = lane + sub * 32; v < nvec; v += 32 * WPR) {
      int4 raw = q4[v];
      const int8_t* q = reinterpret_cast<const int8_t*>(&raw);
      const int base = v * 16;
      float s = 0.f;
#pragma unroll
      for (int j = 0; j < 16; ++j) s += (float)q[j] * x[base + j];
      acc += __half2float(d[base / QK]) * s;
    }
  }
  acc = warp_reduce(acc);
  if (lane == 0) part[wid] = acc;
  __syncthreads();
  if (row < M && sub == 0 && lane == 0) {
    float t = 0.f;
#pragma unroll
    for (int i = 0; i < WPR; ++i) t += part[row_local * WPR + i];
    y[row] = t;
  }
}

// ================================================================ Q4_0

// v0: one thread per row, ggml AoS layout. Same shape as the q8 baseline.
__global__ void qgemv_q4_v0_naive(const block_q4_0* __restrict__ W,
                                  const float* __restrict__ x,
                                  float* __restrict__ y, int M, int K) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= M) return;
  const int nb = K / QK;
  const block_q4_0* w = W + (size_t)row * nb;
  float acc = 0.f;
  for (int b = 0; b < nb; ++b) {
    float s = 0.f;
    for (int j = 0; j < QK / 2; ++j) {
      const uint8_t p = w[b].qs[j];
      s += (float)((int)(p & 0x0F) - 8) * x[b * QK + j];
      s += (float)((int)(p >> 4) - 8) * x[b * QK + j + 16];
    }
    acc += __half2float(w[b].d) * s;
  }
  y[row] = acc;
}

// v1: one warp per row, ggml AoS layout.
__global__ void qgemv_q4_v1_warp(const block_q4_0* __restrict__ W,
                                 const float* __restrict__ x,
                                 float* __restrict__ y, int M, int K) {
  const int warps = blockDim.x / 32;
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.x * warps + (threadIdx.x >> 5);
  if (row >= M) return;
  const int nb = K / QK;
  const block_q4_0* w = W + (size_t)row * nb;
  float acc = 0.f;
  for (int b = lane; b < nb; b += 32) {
    const block_q4_0 blk = w[b];
    float s = 0.f;
#pragma unroll
    for (int j = 0; j < QK / 2; ++j) {
      const uint8_t p = blk.qs[j];
      s += (float)((int)(p & 0x0F) - 8) * x[b * QK + j];
      s += (float)((int)(p >> 4) - 8) * x[b * QK + j + 16];
    }
    acc += __half2float(blk.d) * s;
  }
  acc = warp_reduce(acc);
  if (lane == 0) y[row] = acc;
}

// v3: SoA. One 16-byte lane load is exactly one q4_0 block of quants, which is
// the tidiest the two formats get.
__global__ void qgemv_q4_v3_soa_vec(const uint8_t* __restrict__ Q,
                                    const __half* __restrict__ D,
                                    const float* __restrict__ x,
                                    float* __restrict__ y, int M, int K) {
  const int warps = blockDim.x / 32;
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.x * warps + (threadIdx.x >> 5);
  if (row >= M) return;
  const int nb = K / QK;
  const int4* q4 = reinterpret_cast<const int4*>(Q + (size_t)row * (K / 2));
  const __half* d = D + (size_t)row * nb;
  float acc = 0.f;
  for (int b = lane; b < nb; b += 32) {
    int4 raw = q4[b];
    const uint8_t* p = reinterpret_cast<const uint8_t*>(&raw);
    const int base = b * QK;
    float s = 0.f;
#pragma unroll
    for (int j = 0; j < QK / 2; ++j) {
      s += (float)((int)(p[j] & 0x0F) - 8) * x[base + j];
      s += (float)((int)(p[j] >> 4) - 8) * x[base + j + 16];
    }
    acc += __half2float(d[b]) * s;
  }
  acc = warp_reduce(acc);
  if (lane == 0) y[row] = acc;
}

// ================================================ the missing factorial cells
//
// v1/v3 vary the weight layout while every lane gathers its own slice of x.
// v2 varies where x lives while the weight layout stays AoS. Neither isolates
// the two effects, so the grid below is completed: {AoS, SoA} x {x global,
// x shared}, for both formats. Every cell computes the same numbers.

// q8_0, SoA weights, x staged in shared memory.
__global__ void qgemv_q8_v5_soa_smem(const int8_t* __restrict__ Q,
                                     const __half* __restrict__ D,
                                     const float* __restrict__ x,
                                     float* __restrict__ y, int M, int K) {
  extern __shared__ float sx[];
  for (int i = threadIdx.x; i < K; i += blockDim.x) sx[i] = x[i];
  __syncthreads();
  const int warps = blockDim.x / 32;
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.x * warps + (threadIdx.x >> 5);
  if (row >= M) return;
  const int nb = K / QK;
  const int4* q4 = reinterpret_cast<const int4*>(Q + (size_t)row * K);
  const __half* d = D + (size_t)row * nb;
  const int nvec = K / 16;
  float acc = 0.f;
  for (int v = lane; v < nvec; v += 32) {
    int4 raw = q4[v];
    const int8_t* q = reinterpret_cast<const int8_t*>(&raw);
    const int base = v * 16;
    float s = 0.f;
#pragma unroll
    for (int j = 0; j < 16; ++j) s += (float)q[j] * sx[base + j];
    acc += __half2float(d[base / QK]) * s;
  }
  acc = warp_reduce(acc);
  if (lane == 0) y[row] = acc;
}

// q4_0, ggml AoS weights, x staged in shared memory.
__global__ void qgemv_q4_v2_smem_x(const block_q4_0* __restrict__ W,
                                   const float* __restrict__ x,
                                   float* __restrict__ y, int M, int K) {
  extern __shared__ float sx[];
  for (int i = threadIdx.x; i < K; i += blockDim.x) sx[i] = x[i];
  __syncthreads();
  const int warps = blockDim.x / 32;
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.x * warps + (threadIdx.x >> 5);
  if (row >= M) return;
  const int nb = K / QK;
  const block_q4_0* w = W + (size_t)row * nb;
  float acc = 0.f;
  for (int b = lane; b < nb; b += 32) {
    const block_q4_0 blk = w[b];
    float s = 0.f;
#pragma unroll
    for (int j = 0; j < QK / 2; ++j) {
      const uint8_t p = blk.qs[j];
      s += (float)((int)(p & 0x0F) - 8) * sx[b * QK + j];
      s += (float)((int)(p >> 4) - 8) * sx[b * QK + j + 16];
    }
    acc += __half2float(blk.d) * s;
  }
  acc = warp_reduce(acc);
  if (lane == 0) y[row] = acc;
}

// q4_0, SoA weights, x staged in shared memory.
__global__ void qgemv_q4_v5_soa_smem(const uint8_t* __restrict__ Q,
                                     const __half* __restrict__ D,
                                     const float* __restrict__ x,
                                     float* __restrict__ y, int M, int K) {
  extern __shared__ float sx[];
  for (int i = threadIdx.x; i < K; i += blockDim.x) sx[i] = x[i];
  __syncthreads();
  const int warps = blockDim.x / 32;
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.x * warps + (threadIdx.x >> 5);
  if (row >= M) return;
  const int nb = K / QK;
  const int4* q4 = reinterpret_cast<const int4*>(Q + (size_t)row * (K / 2));
  const __half* d = D + (size_t)row * nb;
  float acc = 0.f;
  for (int b = lane; b < nb; b += 32) {
    int4 raw = q4[b];
    const uint8_t* p = reinterpret_cast<const uint8_t*>(&raw);
    const int base = b * QK;
    float s = 0.f;
#pragma unroll
    for (int j = 0; j < QK / 2; ++j) {
      s += (float)((int)(p[j] & 0x0F) - 8) * sx[base + j];
      s += (float)((int)(p[j] >> 4) - 8) * sx[base + j + 16];
    }
    acc += __half2float(d[b]) * s;
  }
  acc = warp_reduce(acc);
  if (lane == 0) y[row] = acc;
}

// ============================================== attacking the q4_0 unpack cost
//
// Finding 3 says q4_0 loses to q8_0 because unpacking costs more than the bytes
// save. If that is right, cutting the unpack arithmetic should narrow the gap,
// and if it does not, the claim is wrong. v6 is that test.
//
// Two changes, both exact rearrangements rather than approximations:
//
//   1. The bias comes out of the inner loop by algebra. The block sum is
//      sum((q - 8) * x) = sum(q * x) - 8 * sum(x), and sum(x) over a block does
//      not depend on the row, so it is computed once per block and reused by
//      every row. That removes one integer subtract per weight.
//   2. Nibbles are extracted four bytes at a time with a 32-bit mask instead of
//      one byte at a time, which removes three quarters of the mask and shift
//      operations.
//
// Numerics: sum(q*x) with q in [0,15] against 8*sum(x) are comparable in
// magnitude, so the cancellation is mild. The test suite checks it rather than
// assuming it.
__global__ void qgemv_q4_v6_fastunpack(const uint8_t* __restrict__ Q,
                                       const __half* __restrict__ D,
                                       const float* __restrict__ x,
                                       const float* __restrict__ xsum,
                                       float* __restrict__ y, int M, int K) {
  extern __shared__ float smem[];
  const int nb = K / QK;
  float* sx = smem;              // K floats: the activation vector
  float* sxs = smem + K;         // nb floats: per-block sums of x
  for (int i = threadIdx.x; i < K; i += blockDim.x) sx[i] = x[i];
  for (int i = threadIdx.x; i < nb; i += blockDim.x) sxs[i] = xsum[i];
  __syncthreads();

  const int warps = blockDim.x / 32;
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.x * warps + (threadIdx.x >> 5);
  if (row >= M) return;
  const int4* q4 = reinterpret_cast<const int4*>(Q + (size_t)row * (K / 2));
  const __half* d = D + (size_t)row * nb;
  float acc = 0.f;
  for (int b = lane; b < nb; b += 32) {
    int4 raw = q4[b];
    const uint32_t* w = reinterpret_cast<const uint32_t*>(&raw);
    const int base = b * QK;
    float s = 0.f;
#pragma unroll
    for (int g = 0; g < 4; ++g) {          // 4 groups of 4 bytes
      const uint32_t v = w[g];
      const uint32_t lo = v & 0x0F0F0F0Fu;  // four low nibbles at once
      const uint32_t hi = (v >> 4) & 0x0F0F0F0Fu;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int k = g * 4 + j;
        s += (float)((lo >> (8 * j)) & 0xFFu) * sx[base + k];
        s += (float)((hi >> (8 * j)) & 0xFFu) * sx[base + k + 16];
      }
    }
    acc += __half2float(d[b]) * (s - 8.0f * sxs[b]);
  }
  acc = warp_reduce(acc);
  if (lane == 0) y[row] = acc;
}
