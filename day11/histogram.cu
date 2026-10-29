// Day 11: atomics, contention, and privatization (histogram)
// Run in LeetGPU Vector Addition challenge.

#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <functional>
#include <algorithm>
#include <cstdint>

// Baseline: direct global memory atomics (high contention)
__global__ void histogramGlobal(const uint8_t* in, unsigned int* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < n; i += stride) {
        atomicAdd(&out[in[i]], 1);
    }
}

// Privatized: local histogram in shared memory, commit once per block
__global__ void histogramPrivatized(const uint8_t* in, unsigned int* out, int n) {
    __shared__ unsigned int s_bins[256];
    int tid = threadIdx.x;

    if (tid < 256) s_bins[tid] = 0;
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + tid;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < n; i += stride) {
        atomicAdd(&s_bins[in[i]], 1);
    }
    __syncthreads();

    if (tid < 256) {
        atomicAdd(&out[tid], s_bins[tid]);
    }
}

// Privatized + 4-byte vectorized loads (coalesced 32-bit reads)
__global__ void histogramPrivatized4(const uint32_t* in, unsigned int* out, int n_u32) {
    __shared__ unsigned int s_bins[256];
    int tid = threadIdx.x;

    if (tid < 256) s_bins[tid] = 0;
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + tid;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < n_u32; i += stride) {
        uint32_t packed = in[i];
        atomicAdd(&s_bins[(packed      ) & 0xFF], 1);
        atomicAdd(&s_bins[(packed >>  8) & 0xFF], 1);
        atomicAdd(&s_bins[(packed >> 16) & 0xFF], 1);
        atomicAdd(&s_bins[(packed >> 24) & 0xFF], 1);
    }
    __syncthreads();

    if (tid < 256) {
        atomicAdd(&out[tid], s_bins[tid]);
    }
}

// ---- timing helpers ----
template <typename Launch>
float timeMs(Launch launch, int runs = 3) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    launch();
    cudaEventRecord(start);
    for (int r = 0; r < runs; r++) launch();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return ms / runs;
}

struct Bench {
    std::string name;
    std::function<void()> launch;
    std::vector<float> times;
    float median() const {
        std::vector<float> t = times;
        std::sort(t.begin(), t.end());
        return t[t.size() / 2];
    }
    float lo() const { return *std::min_element(times.begin(), times.end()); }
    float hi() const { return *std::max_element(times.begin(), times.end()); }
};

void runRounds(std::vector<Bench>& benches, int rounds = 7) {
    for (int r = 0; r < rounds; r++)
        for (auto& b : benches)
            b.times.push_back(timeMs(b.launch));
    for (const auto& b : benches)
        std::cout << std::left << std::setw(24) << b.name
                  << ": " << std::fixed << std::setprecision(3) << b.median()
                  << " ms (spread " << b.lo() << " – " << b.hi() << ")\n";
}

template <typename Launch>
void warmUp(Launch launch, float targetMs = 500.0f) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    float elapsed = 0.0f;
    while (elapsed < targetMs) {
        for (int k = 0; k < 10; k++) launch();
        cudaEventRecord(stop); cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
    }
    cudaEventDestroy(start); cudaEventDestroy(stop);
}

bool checkBins(const std::vector<unsigned int>& ref, const unsigned int* d_out, const std::string& label) {
    std::vector<unsigned int> h_out(256);
    cudaMemcpy(h_out.data(), d_out, 256 * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    bool ok = true;
    uint64_t total = 0;
    for (int i = 0; i < 256; i++) {
        total += h_out[i];
        if (h_out[i] != ref[i]) ok = false;
    }
    std::cout << label << ": " << (ok ? "✓ all 256 bins match" : "✗ MISMATCH")
              << " (total counted = " << total << ")\n";
    return ok;
}

void run_demo() {
    const int N       = 1 << 24;   // 16M bytes
    const int THREADS = 256;
    const int BLOCKS  = 64;

    std::cout << "=== Day 11: atomics, contention & privatization ===\n";
    std::cout << "N = " << N << " bytes (16 MB)   THREADS = " << THREADS
              << "   BLOCKS = " << BLOCKS << "\n";
    std::cout << "256 bins (0..255)\n\n";

    std::vector<uint8_t> h_in(N);
    std::vector<unsigned int> ref_bins(256, 0);
    uint32_t seed = 1337;
    for (int i = 0; i < N; i++) {
        seed = seed * 1664525u + 1013904223u;
        uint8_t val = static_cast<uint8_t>(seed >> 24);
        h_in[i] = val;
        ref_bins[val]++;
    }

    uint8_t* d_in;
    unsigned int* d_out;
    cudaMalloc(&d_in,  N * sizeof(uint8_t));
    cudaMalloc(&d_out, 256 * sizeof(unsigned int));
    cudaMemcpy(d_in, h_in.data(), N * sizeof(uint8_t), cudaMemcpyHostToDevice);

    auto clearBins = [&] {
        cudaMemset(d_out, 0, 256 * sizeof(unsigned int));
    };

    auto launchGlobal = [&] {
        clearBins();
        histogramGlobal<<<BLOCKS, THREADS>>>(d_in, d_out, N);
    };

    auto launchPrivatized = [&] {
        clearBins();
        histogramPrivatized<<<BLOCKS, THREADS>>>(d_in, d_out, N);
    };

    int n_u32 = N / 4;
    const uint32_t* d_in32 = reinterpret_cast<const uint32_t*>(d_in);
    auto launchPrivatized4 = [&] {
        clearBins();
        histogramPrivatized4<<<BLOCKS, THREADS>>>(d_in32, d_out, n_u32);
    };

    std::cout << "Spinning up T4...\n";
    warmUp(launchPrivatized);
    std::cout << "Warm-up done.\n\n";

    launchGlobal(); cudaDeviceSynchronize();
    checkBins(ref_bins, d_out, "histogram_global     ");

    launchPrivatized(); cudaDeviceSynchronize();
    checkBins(ref_bins, d_out, "histogram_privatized ");

    launchPrivatized4(); cudaDeviceSynchronize();
    checkBins(ref_bins, d_out, "histogram_vectorized4");
    std::cout << "\n";

    std::vector<Bench> benches = {
        {"global_atomic",      launchGlobal,      {}},
        {"shared_privatized",  launchPrivatized,  {}},
        {"shared_vectorized4", launchPrivatized4, {}},
    };

    std::cout << "Timing (7 interleaved rounds, median reported):\n";
    runRounds(benches);

    float tGlobal = benches[0].median();
    float tPriv   = benches[1].median();
    float tVec    = benches[2].median();

    std::cout << "\nspeedup (shared_privatized vs global)  = "
              << std::fixed << std::setprecision(2) << tGlobal / tPriv << "x faster\n";
    std::cout << "speedup (shared_vectorized4 vs global) = "
              << std::fixed << std::setprecision(2) << tGlobal / tVec << "x faster\n";

    cudaError_t err = cudaGetLastError();
    std::cout << (err == cudaSuccess ? "no CUDA errors\n" : cudaGetErrorString(err));

    cudaFree(d_in);
    cudaFree(d_out);
}

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
