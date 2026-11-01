// Day 14: bitonic sort & sorting networks
// Run in LeetGPU Vector Addition challenge.

#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <functional>
#include <algorithm>
#include <cmath>

// Bitonic sort: O(log^2 N) depth sorting network in shared memory
// For 1024 elements: 55 parallel comparison steps and 55 barriers
__global__ void bitonicSortShared(const float* in, float* out, int n) {
    __shared__ float sdata[1024];
    int tid = threadIdx.x; // 512 threads per block
    int blockOffset = blockIdx.x * 1024;

    sdata[2 * tid]     = in[blockOffset + 2 * tid];
    sdata[2 * tid + 1] = in[blockOffset + 2 * tid + 1];
    __syncthreads();

    // 10 outer stages (s = 2, 4, 8 ... 1024)
    for (int s = 2; s <= 1024; s *= 2) {
        // inner steps: stride d halves each round
        for (int d = s / 2; d > 0; d /= 2) {
            int i = 2 * tid - (tid & (d - 1));
            int j = i + d;

            bool ascending = ((i & s) == 0);

            float val_i = sdata[i];
            float val_j = sdata[j];

            if ((val_i > val_j) == ascending) {
                sdata[i] = val_j;
                sdata[j] = val_i;
            }
            __syncthreads();
        }
    }

    out[blockOffset + 2 * tid]     = sdata[2 * tid];
    out[blockOffset + 2 * tid + 1] = sdata[2 * tid + 1];
}

// Odd-even transposition sort: O(N) depth sorting network in shared memory
// For 1024 elements: 1024 parallel steps and 1024 barriers
__global__ void oddEvenSortShared(const float* in, float* out, int n) {
    __shared__ float sdata[1024];
    int tid = threadIdx.x; // 512 threads per block
    int blockOffset = blockIdx.x * 1024;

    sdata[2 * tid]     = in[blockOffset + 2 * tid];
    sdata[2 * tid + 1] = in[blockOffset + 2 * tid + 1];
    __syncthreads();

    for (int p = 0; p < 1024; p++) {
        if (p % 2 == 0) {
            // even phase: pairs (2*tid, 2*tid + 1)
            int i = 2 * tid;
            int j = 2 * tid + 1;
            if (sdata[i] > sdata[j]) {
                float tmp = sdata[i];
                sdata[i] = sdata[j];
                sdata[j] = tmp;
            }
        } else {
            // odd phase: pairs (2*tid + 1, 2*tid + 2)
            if (tid < 511) {
                int i = 2 * tid + 1;
                int j = 2 * tid + 2;
                if (sdata[i] > sdata[j]) {
                    float tmp = sdata[i];
                    sdata[i] = sdata[j];
                    sdata[j] = tmp;
                }
            }
        }
        __syncthreads();
    }

    out[blockOffset + 2 * tid]     = sdata[2 * tid];
    out[blockOffset + 2 * tid + 1] = sdata[2 * tid + 1];
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

bool checkBlocksSorted(const float* d_out, int n, int blockSize, const std::string& label) {
    std::vector<float> h_out(n);
    cudaMemcpy(h_out.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost);

    int numBlocks = n / blockSize;
    bool ok = true;
    for (int b = 0; b < numBlocks; b++) {
        int base = b * blockSize;
        for (int i = 0; i < blockSize - 1; i++) {
            if (h_out[base + i] > h_out[base + i + 1]) {
                ok = false;
                break;
            }
        }
        if (!ok) break;
    }
    std::cout << label << ": " << (ok ? "✓ all 1024 blocks sorted correctly" : "✗ MISMATCH") << "\n";
    return ok;
}

void run_demo() {
    const int N       = 1024 * 1024;  // 1M floats (1,048,576 elements)
    const int BLOCKS  = N / 1024;     // 1024 blocks of 1024 elements
    const int THREADS = 512;          // 512 threads handle 1024 elements

    std::cout << "=== Day 14: bitonic sort & sorting networks ===\n";
    std::cout << "N = " << N << " floats (" << (N * sizeof(float)) / (1024 * 1024) << " MB)\n";
    std::cout << "blocks = " << BLOCKS << "   block size = 1024 elements\n\n";

    // generate random floats
    std::vector<float> h_in(N);
    uint32_t seed = 42;
    for (int i = 0; i < N; i++) {
        seed = seed * 1664525u + 1013904223u;
        h_in[i] = static_cast<float>(seed % 10000) / 10.0f;
    }

    float *d_in, *d_out;
    cudaMalloc(&d_in,  N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));
    cudaMemcpy(d_in, h_in.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    auto launchBitonic = [&] {
        bitonicSortShared<<<BLOCKS, THREADS>>>(d_in, d_out, N);
    };

    auto launchOddEven = [&] {
        oddEvenSortShared<<<BLOCKS, THREADS>>>(d_in, d_out, N);
    };

    std::cout << "Spinning up T4...\n";
    warmUp(launchBitonic);
    std::cout << "Warm-up done.\n\n";

    launchBitonic(); cudaDeviceSynchronize();
    checkBlocksSorted(d_out, N, 1024, "bitonic_shared ");

    launchOddEven(); cudaDeviceSynchronize();
    checkBlocksSorted(d_out, N, 1024, "odd_even_shared");
    std::cout << "\n";

    std::vector<Bench> benches = {
        {"bitonic_shared",  launchBitonic, {}},
        {"odd_even_shared", launchOddEven, {}},
    };

    std::cout << "Timing (7 interleaved rounds, median reported):\n";
    runRounds(benches);

    float tBitonic = benches[0].median();
    float tOddEven = benches[1].median();

    std::cout << "\nspeedup (bitonic vs odd-even) = "
              << std::fixed << std::setprecision(2) << tOddEven / tBitonic << "x faster\n";

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
