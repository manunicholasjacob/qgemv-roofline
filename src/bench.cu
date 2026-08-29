// Harness: correctness first, then timing, then achieved bandwidth.
//
// Nothing here reports a number that was not measured on the device it names.
// Timing uses CUDA events around a warmed-up loop and reports a distribution,
// because a single sample of a GPU kernel is not a measurement.

#include "qgemv_kernels.cuh"

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#define CUDA_OK(call)                                                          \
  do {                                                                         \
    cudaError_t e_ = (call);                                                   \
    if (e_ != cudaSuccess) {                                                   \
      fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_),      \
              __FILE__, __LINE__);                                             \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

// ---------------------------------------------------------------- quantize
// Both routines follow ggml semantics so the bytes are the bytes a GGUF holds.

static void quantize_q8_0(const float* src, int n, std::vector<block_q8_0>& out) {
  const int nb = n / QK;
  out.resize(nb);
  for (int b = 0; b < nb; ++b) {
    const float* v = src + b * QK;
    float amax = 0.f;
    for (int j = 0; j < QK; ++j) amax = std::max(amax, std::fabs(v[j]));
    const float d = amax / 127.f;
    const float id = d ? 1.f / d : 0.f;
    out[b].d = __float2half(d);
    for (int j = 0; j < QK; ++j) {
      int q = (int)lrintf(v[j] * id);
      out[b].qs[j] = (int8_t)std::max(-127, std::min(127, q));
    }
  }
}

static void quantize_q4_0(const float* src, int n, std::vector<block_q4_0>& out) {
  const int nb = n / QK;
  out.resize(nb);
  for (int b = 0; b < nb; ++b) {
    const float* v = src + b * QK;
    float amax = 0.f, mx = 0.f;
    for (int j = 0; j < QK; ++j) {
      if (std::fabs(v[j]) > amax) { amax = std::fabs(v[j]); mx = v[j]; }
    }
    const float d = mx / -8.f;
    const float id = d ? 1.f / d : 0.f;
    out[b].d = __float2half(d);
    for (int j = 0; j < QK / 2; ++j) {
      int q0 = (int)(v[j] * id + 8.5f);
      int q1 = (int)(v[j + 16] * id + 8.5f);
      q0 = std::max(0, std::min(15, q0));
      q1 = std::max(0, std::min(15, q1));
      out[b].qs[j] = (uint8_t)(q0 | (q1 << 4));
    }
  }
}

// ---------------------------------------------------------------- reference
// Double-precision CPU reference over the dequantized values. This is the
// ground truth for every kernel, and it is deliberately the slow obvious code.

static void ref_q8(const std::vector<block_q8_0>& W, const std::vector<float>& x,
                   int M, int K, std::vector<double>& y) {
  const int nb = K / QK;
  y.assign(M, 0.0);
  for (int r = 0; r < M; ++r) {
    double acc = 0.0;
    const block_q8_0* w = W.data() + (size_t)r * nb;
    for (int b = 0; b < nb; ++b) {
      const double d = (double)__half2float(w[b].d);
      double s = 0.0;
      for (int j = 0; j < QK; ++j) s += (double)w[b].qs[j] * (double)x[b * QK + j];
      acc += d * s;
    }
    y[r] = acc;
  }
}

static void ref_q4(const std::vector<block_q4_0>& W, const std::vector<float>& x,
                   int M, int K, std::vector<double>& y) {
  const int nb = K / QK;
  y.assign(M, 0.0);
  for (int r = 0; r < M; ++r) {
    double acc = 0.0;
    const block_q4_0* w = W.data() + (size_t)r * nb;
    for (int b = 0; b < nb; ++b) {
      const double d = (double)__half2float(w[b].d);
      double s = 0.0;
      for (int j = 0; j < QK / 2; ++j) {
        const uint8_t p = w[b].qs[j];
        s += (double)((int)(p & 0x0F) - 8) * (double)x[b * QK + j];
        s += (double)((int)(p >> 4) - 8) * (double)x[b * QK + j + 16];
      }
      acc += d * s;
    }
    y[r] = acc;
  }
}

// ---------------------------------------------------------------- timing

struct Stats { double median, mn, mx, mean, sd, p95; };

static Stats summarize(std::vector<float> ms) {
  std::sort(ms.begin(), ms.end());
  Stats s{};
  const size_t n = ms.size();
  s.median = ms[n / 2];
  s.mn = ms.front();
  s.mx = ms.back();
  s.p95 = ms[(size_t)(0.95 * (n - 1))];
  double sum = 0.0;
  for (float v : ms) sum += v;
  s.mean = sum / n;
  double var = 0.0;
  for (float v : ms) var += (v - s.mean) * (v - s.mean);
  s.sd = std::sqrt(var / n);
  return s;
}

template <typename F>
static Stats time_kernel(F launch, int warmup, int reps) {
  cudaEvent_t a, b;
  CUDA_OK(cudaEventCreate(&a));
  CUDA_OK(cudaEventCreate(&b));
  for (int i = 0; i < warmup; ++i) launch();
  CUDA_OK(cudaDeviceSynchronize());
  std::vector<float> ms;
  ms.reserve(reps);
  for (int i = 0; i < reps; ++i) {
    CUDA_OK(cudaEventRecord(a));
    launch();
    CUDA_OK(cudaEventRecord(b));
    CUDA_OK(cudaEventSynchronize(b));
    float t = 0.f;
    CUDA_OK(cudaEventElapsedTime(&t, a, b));
    ms.push_back(t);
  }
  CUDA_OK(cudaGetLastError());
  cudaEventDestroy(a);
  cudaEventDestroy(b);
  return summarize(ms);
}

// ---------------------------------------------------------------- ceiling
// A pure streaming read. This is the measured upper bound that every achieved
// figure below is a fraction of. The datasheet number is not that bound.

__global__ void stream_read(const int4* __restrict__ src, size_t n4,
                            float* __restrict__ sink) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  int4 acc = make_int4(0, 0, 0, 0);
  for (; i < n4; i += stride) {
    int4 v = src[i];
    acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w;
  }
  if ((acc.x | acc.y | acc.z | acc.w) == 0x7fffffff) sink[0] = 1.f;  // never taken
}

static void run_ceiling(size_t bytes, int reps) {
  const size_t n4 = bytes / sizeof(int4);
  int4* d = nullptr;
  float* sink = nullptr;
  CUDA_OK(cudaMalloc(&d, n4 * sizeof(int4)));
  CUDA_OK(cudaMalloc(&sink, sizeof(float)));
  CUDA_OK(cudaMemset(d, 1, n4 * sizeof(int4)));
  int dev = 0;
  cudaDeviceProp p{};
  CUDA_OK(cudaGetDeviceProperties(&p, dev));
  const int threads = 256;
  const int blocks = p.multiProcessorCount * 16;
  Stats s = time_kernel([&] { stream_read<<<blocks, threads>>>(d, n4, sink); }, 10, reps);
  const double gb = (double)(n4 * sizeof(int4)) / 1e9;
  printf("{\"kernel\":\"stream_read\",\"bytes\":%zu,\"ms_median\":%.6f,"
         "\"ms_min\":%.6f,\"ms_sd\":%.6f,\"gbps_median\":%.3f,\"gbps_max\":%.3f}\n",
         n4 * sizeof(int4), s.median, s.mn, s.sd, gb / (s.median / 1e3),
         gb / (s.mn / 1e3));

  // D2D copy, a second independent read of the same ceiling.
  int4* d2 = nullptr;
  CUDA_OK(cudaMalloc(&d2, n4 * sizeof(int4)));
  Stats c = time_kernel(
      [&] { cudaMemcpyAsync(d2, d, n4 * sizeof(int4), cudaMemcpyDeviceToDevice); }, 10,
      reps);
  const double gb2 = 2.0 * gb;  // read + write
  printf("{\"kernel\":\"memcpy_d2d\",\"bytes\":%zu,\"ms_median\":%.6f,"
         "\"gbps_median\":%.3f}\n",
         2 * n4 * sizeof(int4), c.median, gb2 / (c.median / 1e3));
  cudaFree(d);
  cudaFree(d2);
  cudaFree(sink);
}

// ---------------------------------------------------------------- main

struct Result {
  const char* name;
  const char* fmt;
  const char* layout;
  Stats s;
  double gbps;
  double max_abs_err;
  double max_rel_err;
};

static void emit(const Result& r, size_t wbytes, int M, int K) {
  printf("{\"kernel\":\"%s\",\"format\":\"%s\",\"layout\":\"%s\",\"M\":%d,\"K\":%d,"
         "\"weight_bytes\":%zu,\"ms_median\":%.6f,\"ms_min\":%.6f,\"ms_p95\":%.6f,"
         "\"ms_sd\":%.6f,\"gbps\":%.3f,\"max_abs_err\":%.6g,\"max_rel_err\":%.6g}\n",
         r.name, r.fmt, r.layout, M, K, wbytes, r.s.median, r.s.mn, r.s.p95, r.s.sd,
         r.gbps, r.max_abs_err, r.max_rel_err);
  fflush(stdout);
}

static void check(const std::vector<float>& got, const std::vector<double>& want,
                  double& max_abs, double& max_rel) {
  max_abs = 0.0;
  max_rel = 0.0;
  double scale = 0.0;
  for (double v : want) scale = std::max(scale, std::fabs(v));
  for (size_t i = 0; i < want.size(); ++i) {
    const double e = std::fabs((double)got[i] - want[i]);
    max_abs = std::max(max_abs, e);
    if (scale > 0) max_rel = std::max(max_rel, e / scale);
  }
}

int main(int argc, char** argv) {
  int M = 4096, K = 4096, reps = 200, warmup = 20;
  std::string mode = "bench";
  double sustain_s = 0.0;
  size_t ceil_bytes = 512u * 1024u * 1024u;
  std::string only;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() { return std::string(argv[++i]); };
    if (a == "--M") M = std::stoi(next());
    else if (a == "--K") K = std::stoi(next());
    else if (a == "--reps") reps = std::stoi(next());
    else if (a == "--mode") mode = next();
    else if (a == "--sustain") sustain_s = std::stod(next());
    else if (a == "--only") only = next();
    else if (a == "--bytes") ceil_bytes = (size_t)std::stoull(next());
  }
  if (K % 512 != 0) { fprintf(stderr, "K must be a multiple of 512\n"); return 1; }

  cudaDeviceProp p{};
  CUDA_OK(cudaGetDeviceProperties(&p, 0));
  int memclk_khz = 0, buswidth = 0, smclk_khz = 0;
  cudaDeviceGetAttribute(&memclk_khz, cudaDevAttrMemoryClockRate, 0);
  cudaDeviceGetAttribute(&buswidth, cudaDevAttrGlobalMemoryBusWidth, 0);
  cudaDeviceGetAttribute(&smclk_khz, cudaDevAttrClockRate, 0);
  printf("{\"device\":\"%s\",\"cc\":\"sm_%d%d\",\"sms\":%d,\"l2_bytes\":%d,"
         "\"mem_clock_khz\":%d,\"bus_bits\":%d,\"sm_clock_khz\":%d,"
         "\"datasheet_gbps_x2\":%.2f}\n",
         p.name, p.major, p.minor, p.multiProcessorCount, p.l2CacheSize, memclk_khz,
         buswidth, smclk_khz, 2.0 * memclk_khz * 1e3 * (buswidth / 8.0) / 1e9);
  fflush(stdout);

  if (mode == "ceiling") {
    run_ceiling(ceil_bytes, reps);
    return 0;
  }

  const int nb = K / QK;
  const size_t nw = (size_t)M * K;

  // Host data. Weights are drawn once and quantized once; every kernel below
  // sees the same numbers, so any output difference is a kernel bug.
  std::vector<float> hw(nw), hx(K);
  srand(1234);
  for (size_t i = 0; i < nw; ++i) hw[i] = (float)((rand() / (double)RAND_MAX) * 2.0 - 1.0);
  for (int i = 0; i < K; ++i) hx[i] = (float)((rand() / (double)RAND_MAX) * 2.0 - 1.0);

  std::vector<block_q8_0> q8(( size_t)M * nb);
  std::vector<block_q4_0> q4((size_t)M * nb);
  for (int r = 0; r < M; ++r) {
    std::vector<block_q8_0> t8;
    std::vector<block_q4_0> t4;
    quantize_q8_0(hw.data() + (size_t)r * K, K, t8);
    quantize_q4_0(hw.data() + (size_t)r * K, K, t4);
    memcpy(&q8[(size_t)r * nb], t8.data(), nb * sizeof(block_q8_0));
    memcpy(&q4[(size_t)r * nb], t4.data(), nb * sizeof(block_q4_0));
  }

  // SoA planes carry the identical quantized values, only rearranged.
  std::vector<int8_t> soa8_q(nw);
  std::vector<__half> soa8_d((size_t)M * nb);
  std::vector<uint8_t> soa4_q(nw / 2);
  std::vector<__half> soa4_d((size_t)M * nb);
  for (int r = 0; r < M; ++r) {
    for (int b = 0; b < nb; ++b) {
      const block_q8_0& B8 = q8[(size_t)r * nb + b];
      soa8_d[(size_t)r * nb + b] = B8.d;
      memcpy(&soa8_q[(size_t)r * K + (size_t)b * QK], B8.qs, QK);
      const block_q4_0& B4 = q4[(size_t)r * nb + b];
      soa4_d[(size_t)r * nb + b] = B4.d;
      memcpy(&soa4_q[(size_t)r * (K / 2) + (size_t)b * (QK / 2)], B4.qs, QK / 2);
    }
  }

  std::vector<double> ry8, ry4;
  ref_q8(q8, hx, M, K, ry8);
  ref_q4(q4, hx, M, K, ry4);

  // Device
  block_q8_0* dq8 = nullptr; block_q4_0* dq4 = nullptr;
  int8_t* ds8q = nullptr; uint8_t* ds4q = nullptr;
  __half *ds8d = nullptr, *ds4d = nullptr;
  float *dx = nullptr, *dy = nullptr, *dxsum = nullptr;
  CUDA_OK(cudaMalloc(&dq8, q8.size() * sizeof(block_q8_0)));
  CUDA_OK(cudaMalloc(&dq4, q4.size() * sizeof(block_q4_0)));
  CUDA_OK(cudaMalloc(&ds8q, soa8_q.size()));
  CUDA_OK(cudaMalloc(&ds4q, soa4_q.size()));
  CUDA_OK(cudaMalloc(&ds8d, soa8_d.size() * sizeof(__half)));
  CUDA_OK(cudaMalloc(&ds4d, soa4_d.size() * sizeof(__half)));
  CUDA_OK(cudaMalloc(&dx, K * sizeof(float)));
  CUDA_OK(cudaMalloc(&dy, M * sizeof(float)));
  CUDA_OK(cudaMalloc(&dxsum, nb * sizeof(float)));
  CUDA_OK(cudaMemcpy(dq8, q8.data(), q8.size() * sizeof(block_q8_0), cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(dq4, q4.data(), q4.size() * sizeof(block_q4_0), cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(ds8q, soa8_q.data(), soa8_q.size(), cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(ds4q, soa4_q.data(), soa4_q.size(), cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(ds8d, soa8_d.data(), soa8_d.size() * sizeof(__half), cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(ds4d, soa4_d.data(), soa4_d.size() * sizeof(__half), cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(dx, hx.data(), K * sizeof(float), cudaMemcpyHostToDevice));
  // Per-block sums of x, used by q4_v6 to lift the -8 bias out of the inner
  // loop. Computed once on the host; it is nb floats and row independent.
  std::vector<float> hxsum(nb, 0.f);
  for (int b = 0; b < nb; ++b) {
    float t = 0.f;
    for (int j = 0; j < QK; ++j) t += hx[b * QK + j];
    hxsum[b] = t;
  }
  CUDA_OK(cudaMemcpy(dxsum, hxsum.data(), nb * sizeof(float), cudaMemcpyHostToDevice));

  const size_t wb8 = q8.size() * sizeof(block_q8_0);
  const size_t wb4 = q4.size() * sizeof(block_q4_0);
  std::vector<float> hy(M);

  auto run_one = [&](const char* name, const char* fmt, const char* layout,
                     size_t wbytes, const std::vector<double>& ref,
                     std::function<void()> launch) {
    if (!only.empty() && only != name) return;
    CUDA_OK(cudaMemset(dy, 0, M * sizeof(float)));
    launch();
    CUDA_OK(cudaDeviceSynchronize());
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaMemcpy(hy.data(), dy, M * sizeof(float), cudaMemcpyDeviceToHost));
    Result r{name, fmt, layout, {}, 0, 0, 0};
    check(hy, ref, r.max_abs_err, r.max_rel_err);
    if (sustain_s > 0.0) {
      // Sustained mode: run flat out for a wall-clock window so an external
      // energy counter has something to integrate over.
      double elapsed = 0.0;
      cudaEvent_t a, b;
      cudaEventCreate(&a); cudaEventCreate(&b);
      cudaEventRecord(a);
      long long iters = 0;
      while (elapsed < sustain_s * 1000.0) {
        for (int i = 0; i < 200; ++i) { launch(); ++iters; }
        CUDA_OK(cudaDeviceSynchronize());
        cudaEventRecord(b); cudaEventSynchronize(b);
        float t; cudaEventElapsedTime(&t, a, b); elapsed = t;
      }
      printf("{\"sustain\":\"%s\",\"iters\":%lld,\"ms\":%.3f,\"bytes_total\":%.6g}\n",
             name, iters, elapsed, (double)wbytes * (double)iters);
      fflush(stdout);
      return;
    }
    r.s = time_kernel(launch, warmup, reps);
    r.gbps = ((double)wbytes / 1e9) / (r.s.median / 1e3);
    emit(r, wbytes, M, K);
  };

  const int TPB = 256;
  const int warps_per_block = TPB / 32;

  run_one("q8_v0_naive", "q8_0", "aos", wb8, ry8, [&] {
    qgemv_q8_v0_naive<<<(M + TPB - 1) / TPB, TPB>>>(dq8, dx, dy, M, K);
  });
  run_one("q8_v1_warp", "q8_0", "aos", wb8, ry8, [&] {
    qgemv_q8_v1_warp<<<(M + warps_per_block - 1) / warps_per_block, TPB>>>(dq8, dx, dy, M, K);
  });
  run_one("q8_v2_smem_x", "q8_0", "aos", wb8, ry8, [&] {
    qgemv_q8_v2_smem_x<<<(M + warps_per_block - 1) / warps_per_block, TPB,
                         K * sizeof(float)>>>(dq8, dx, dy, M, K);
  });
  run_one("q8_v3_soa_vec", "q8_0", "soa", wb8, ry8, [&] {
    qgemv_q8_v3_soa_vec<<<(M + warps_per_block - 1) / warps_per_block, TPB>>>(ds8q, ds8d, dx, dy, M, K);
  });
  run_one("q8_v4_split2", "q8_0", "soa", wb8, ry8, [&] {
    const int rpb = TPB / (32 * 2);
    qgemv_q8_v4_split<2><<<(M + rpb - 1) / rpb, TPB, warps_per_block * sizeof(float)>>>(
        ds8q, ds8d, dx, dy, M, K);
  });
  run_one("q8_v4_split4", "q8_0", "soa", wb8, ry8, [&] {
    const int rpb = TPB / (32 * 4);
    qgemv_q8_v4_split<4><<<(M + rpb - 1) / rpb, TPB, warps_per_block * sizeof(float)>>>(
        ds8q, ds8d, dx, dy, M, K);
  });
  run_one("q8_v5_soa_smem", "q8_0", "soa", wb8, ry8, [&] {
    qgemv_q8_v5_soa_smem<<<(M + warps_per_block - 1) / warps_per_block, TPB,
                           K * sizeof(float)>>>(ds8q, ds8d, dx, dy, M, K);
  });
  run_one("q4_v0_naive", "q4_0", "aos", wb4, ry4, [&] {
    qgemv_q4_v0_naive<<<(M + TPB - 1) / TPB, TPB>>>(dq4, dx, dy, M, K);
  });
  run_one("q4_v1_warp", "q4_0", "aos", wb4, ry4, [&] {
    qgemv_q4_v1_warp<<<(M + warps_per_block - 1) / warps_per_block, TPB>>>(dq4, dx, dy, M, K);
  });
  run_one("q4_v2_smem_x", "q4_0", "aos", wb4, ry4, [&] {
    qgemv_q4_v2_smem_x<<<(M + warps_per_block - 1) / warps_per_block, TPB,
                         K * sizeof(float)>>>(dq4, dx, dy, M, K);
  });
  run_one("q4_v3_soa_vec", "q4_0", "soa", wb4, ry4, [&] {
    qgemv_q4_v3_soa_vec<<<(M + warps_per_block - 1) / warps_per_block, TPB>>>(ds4q, ds4d, dx, dy, M, K);
  });
  run_one("q4_v5_soa_smem", "q4_0", "soa", wb4, ry4, [&] {
    qgemv_q4_v5_soa_smem<<<(M + warps_per_block - 1) / warps_per_block, TPB,
                           K * sizeof(float)>>>(ds4q, ds4d, dx, dy, M, K);
  });
  run_one("q4_v6_fastunpack", "q4_0", "soa", wb4, ry4, [&] {
    qgemv_q4_v6_fastunpack<<<(M + warps_per_block - 1) / warps_per_block, TPB,
                             (K + nb) * sizeof(float)>>>(ds4q, ds4d, dx, dxsum, dy, M, K);
  });

  cudaFree(dq8); cudaFree(dq4); cudaFree(ds8q); cudaFree(ds4q);
  cudaFree(ds8d); cudaFree(ds4d); cudaFree(dx); cudaFree(dy); cudaFree(dxsum);
  return 0;
}
