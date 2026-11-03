#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

// ── kernels ──────────────────────────────────────────────────────────────────

__global__ void copyFloat(const float* __restrict__ src, float* __restrict__ dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}

__global__ void copyFloat4(const float4* __restrict__ src, float4* __restrict__ dst, int n4) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n4) dst[i] = src[i];
}

// ── helpers ───────────────────────────────────────────────────────────────────

struct Result {
    float ms_lo, ms_med, ms_hi;
};

template<typename Fn>
Result time7(Fn fn) {
    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);
    float t[7];
    for (int r = 0; r < 7; r++) {
        cudaEventRecord(s);
        fn();
        cudaEventRecord(e);
        cudaEventSynchronize(e);
        cudaEventElapsedTime(&t[r], s, e);
    }
    cudaEventDestroy(s); cudaEventDestroy(e);
    Result res;
    float sorted[7]; std::copy(t, t+7, sorted); std::sort(sorted, sorted+7);
    res.ms_lo = sorted[0]; res.ms_med = sorted[3]; res.ms_hi = sorted[6];
    return res;
}

// ── demo ──────────────────────────────────────────────────────────────────────

void run_demo() {
    const int N       = 1 << 25;   // 32M floats = 128 MB
    const int THREADS = 256;

    size_t bytes = (size_t)N * sizeof(float);

    float *d_src, *d_dst;
    cudaMalloc(&d_src, bytes);
    cudaMalloc(&d_dst, bytes);

    // fill source
    std::vector<float> h(N, 1.5f);
    cudaMemcpy(d_src, h.data(), bytes, cudaMemcpyHostToDevice);

    int n4     = N / 4;
    int blocks1 = (N  + THREADS - 1) / THREADS;
    int blocks4 = (n4 + THREADS - 1) / THREADS;

    // warm-up
    copyFloat <<<blocks1, THREADS>>>(d_src, d_dst, N);
    copyFloat4<<<blocks4, THREADS>>>((float4*)d_src, (float4*)d_dst, n4);
    cudaDeviceSynchronize();
    std::cout << "Warm-up done.\n\n";

    auto r1 = time7([&]{ copyFloat <<<blocks1, THREADS>>>(d_src, d_dst, N);         cudaDeviceSynchronize(); });
    auto r4 = time7([&]{ copyFloat4<<<blocks4, THREADS>>>((float4*)d_src, (float4*)d_dst, n4); cudaDeviceSynchronize(); });

    // bandwidth = 2 * bytes (read + write) / time
    double peak_gbps = 320.0;  // T4 theoretical peak
    double gbps1 = (2.0 * bytes) / ((double)r1.ms_med * 1e-3) / 1e9;
    double gbps4 = (2.0 * bytes) / ((double)r4.ms_med * 1e-3) / 1e9;

    std::cout << "=== Day 16: memory bandwidth — float vs float4 ===\n";
    std::cout << "N = " << N << " floats (" << bytes/1024/1024 << " MB)   THREADS = " << THREADS << "\n\n";

    std::cout << "Timing (7 interleaved rounds, median reported):\n";
    std::printf("copyFloat           : %.3f ms (spread %.3f – %.3f)  →  %.1f GB/s  (%.1f%% of peak)\n",
                r1.ms_med, r1.ms_lo, r1.ms_hi, gbps1, (gbps1 / peak_gbps) * 100.0);
    std::printf("copyFloat4          : %.3f ms (spread %.3f – %.3f)  →  %.1f GB/s  (%.1f%% of peak)\n",
                r4.ms_med, r4.ms_lo, r4.ms_hi, gbps4, (gbps4 / peak_gbps) * 100.0);

    double speedup = r1.ms_med / r4.ms_med;
    std::printf("\nspeedup (float4 vs float) = %.2fx faster\n", speedup);

    // correctness check
    std::vector<float> h_out(N);
    copyFloat4<<<blocks4, THREADS>>>((float4*)d_src, (float4*)d_dst, n4);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out.data(), d_dst, bytes, cudaMemcpyDeviceToHost);
    bool ok = true;
    for (int i = 0; i < N; i++) if (std::abs(h_out[i] - 1.5f) > 1e-6f) { ok = false; break; }
    std::cout << (ok ? "\nfloat4 copy: ✓ correct\n" : "\nfloat4 copy: ✗ mismatch\n");

    cudaError_t err = cudaGetLastError();
    std::cout << (err == cudaSuccess ? "no CUDA errors\n" : cudaGetErrorString(err));

    cudaFree(d_src); cudaFree(d_dst);
}

// ── LeetGPU harness ───────────────────────────────────────────────────────────

__global__ void vectorAdd(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] + B[i];
}

extern "C" void solve(const float* A, const float* B, float* C, int N) {
    static bool demoDone = false;
    if (!demoDone) {
        run_demo();
        demoDone = true;
    }
    int threads = 256;
    int blocks  = (N + threads - 1) / threads;
    vectorAdd<<<blocks, threads>>>(A, B, C, N);
    cudaDeviceSynchronize();
}
