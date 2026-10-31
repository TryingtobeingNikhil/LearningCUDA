// Day 13: large-array scan & stream compaction
// Run in LeetGPU Vector Addition challenge.

#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <functional>
#include <algorithm>
#include <cmath>

// Pass 1: local Kogge-Stone scan inside each 1024-thread block
__global__ void localScan(const float* in, float* out, float* blockSums, int n) {
    __shared__ float sdata[2][1024];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    int pout = 0;
    int pin  = 1;

    sdata[pout][tid] = (gid < n) ? in[gid] : 0.0f;
    __syncthreads();

    #pragma unroll
    for (int stride = 1; stride < 1024; stride *= 2) {
        pout = 1 - pout;
        pin  = 1 - pout;
        float val = sdata[pin][tid];
        if (tid >= stride) {
            val += sdata[pin][tid - stride];
        }
        sdata[pout][tid] = val;
        __syncthreads();
    }

    if (gid < n) {
        out[gid] = sdata[pout][tid];
    }

    if (tid == 1023) {
        blockSums[blockIdx.x] = sdata[pout][1023];
    }
}

// Pass 2: single block scans the block sums to compute exclusive offsets
__global__ void scanBlockSums(const float* blockSums, float* blockOffsets, int numBlocks) {
    __shared__ float sdata[2][1024];
    int tid = threadIdx.x;

    int pout = 0;
    int pin  = 1;

    sdata[pout][tid] = (tid < numBlocks) ? blockSums[tid] : 0.0f;
    __syncthreads();

    #pragma unroll
    for (int stride = 1; stride < 1024; stride *= 2) {
        pout = 1 - pout;
        pin  = 1 - pout;
        float val = sdata[pin][tid];
        if (tid >= stride) {
            val += sdata[pin][tid - stride];
        }
        sdata[pout][tid] = val;
        __syncthreads();
    }

    // exclusive scan: shift right by 1, index 0 gets 0.0
    if (tid < numBlocks) {
        blockOffsets[tid] = (tid == 0) ? 0.0f : sdata[pout][tid - 1];
    }
}

// Pass 3: add block offsets to each element
__global__ void addOffsets(float* out, const float* blockOffsets, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        out[gid] += blockOffsets[blockIdx.x];
    }
}

// Baseline compaction: global atomicAdd counter (heavy contention)
__global__ void compactAtomic(const float* in, float* out, int* counter, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n && in[gid] > 0.0f) {
        int pos = atomicAdd(counter, 1);
        out[pos] = in[gid];
    }
}

// Compaction Pass 1: compute predicate flags and local scan
__global__ void compactPass1(const float* in, float* flagsScanned, float* blockSums, int n) {
    __shared__ float sdata[2][1024];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    float flag = (gid < n && in[gid] > 0.0f) ? 1.0f : 0.0f;

    int pout = 0;
    int pin  = 1;
    sdata[pout][tid] = flag;
    __syncthreads();

    #pragma unroll
    for (int stride = 1; stride < 1024; stride *= 2) {
        pout = 1 - pout;
        pin  = 1 - pout;
        float val = sdata[pin][tid];
        if (tid >= stride) {
            val += sdata[pin][tid - stride];
        }
        sdata[pout][tid] = val;
        __syncthreads();
    }

    if (gid < n) {
        flagsScanned[gid] = sdata[pout][tid];
    }
    if (tid == 1023) {
        blockSums[blockIdx.x] = sdata[pout][1023];
    }
}

// Compaction Pass 3: scatter positive elements using scanned offsets (zero atomics)
__global__ void compactPass3(const float* in, float* out, const float* flagsScanned,
                             const float* blockOffsets, int n) {
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    if (gid < n && in[gid] > 0.0f) {
        // exclusive local rank within the block
        float localExcl = (tid == 0) ? 0.0f : flagsScanned[gid - 1];
        int globalIdx = static_cast<int>(blockOffsets[blockIdx.x] + localExcl);
        out[globalIdx] = in[gid];
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

void run_demo() {
    const int N       = 1024 * 1024;  // 1M floats (1,048,576 elements)
    const int THREADS = 1024;
    const int BLOCKS  = N / THREADS;  // 1024 blocks

    std::cout << "=== Day 13: large-array scan & stream compaction ===\n";
    std::cout << "N = " << N << " floats (" << (N * sizeof(float)) / (1024 * 1024) << " MB)\n";
    std::cout << "blocks = " << BLOCKS << "   threads per block = " << THREADS << "\n\n";

    // input with mix of positive and negative numbers (~50% positive)
    std::vector<float> h_in(N);
    std::vector<float> h_ref_scan(N);
    std::vector<float> h_ref_compact;
    double running = 0.0;
    for (int i = 0; i < N; i++) {
        float val = (i % 2 == 0) ? static_cast<float>((i % 5) + 1) : -1.0f;
        h_in[i] = val;
        running += val;
        h_ref_scan[i] = static_cast<float>(running);
        if (val > 0.0f) h_ref_compact.push_back(val);
    }

    float *d_in, *d_scan_out, *d_compact_out;
    float *d_blockSums, *d_blockOffsets, *d_flagsScanned;
    int *d_atomicCount;

    cudaMalloc(&d_in,           N * sizeof(float));
    cudaMalloc(&d_scan_out,     N * sizeof(float));
    cudaMalloc(&d_compact_out,  N * sizeof(float));
    cudaMalloc(&d_blockSums,    BLOCKS * sizeof(float));
    cudaMalloc(&d_blockOffsets, BLOCKS * sizeof(float));
    cudaMalloc(&d_flagsScanned, N * sizeof(float));
    cudaMalloc(&d_atomicCount,  sizeof(int));

    cudaMemcpy(d_in, h_in.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    // 1. Hierarchical scan: 3-pass
    auto launchHierarchicalScan = [&] {
        localScan<<<BLOCKS, THREADS>>>(d_in, d_scan_out, d_blockSums, N);
        scanBlockSums<<<1, THREADS>>>(d_blockSums, d_blockOffsets, BLOCKS);
        addOffsets<<<BLOCKS, THREADS>>>(d_scan_out, d_blockOffsets, N);
    };

    // 2. Stream compaction: atomicAdd baseline
    auto launchCompactAtomic = [&] {
        cudaMemset(d_atomicCount, 0, sizeof(int));
        compactAtomic<<<BLOCKS, THREADS>>>(d_in, d_compact_out, d_atomicCount, N);
    };

    // 3. Stream compaction: scan-based (zero atomics, stable order)
    auto launchCompactScan = [&] {
        compactPass1<<<BLOCKS, THREADS>>>(d_in, d_flagsScanned, d_blockSums, N);
        scanBlockSums<<<1, THREADS>>>(d_blockSums, d_blockOffsets, BLOCKS);
        compactPass3<<<BLOCKS, THREADS>>>(d_in, d_compact_out, d_flagsScanned, d_blockOffsets, N);
    };

    std::cout << "Spinning up T4...\n";
    warmUp(launchHierarchicalScan);
    std::cout << "Warm-up done.\n\n";

    // ---- correctness checks ----
    launchHierarchicalScan(); cudaDeviceSynchronize();
    std::vector<float> h_gpu_scan(N);
    cudaMemcpy(h_gpu_scan.data(), d_scan_out, N * sizeof(float), cudaMemcpyDeviceToHost);
    bool scanOk = true;
    for (int i = 0; i < N; i++) {
        if (std::abs(h_gpu_scan[i] - h_ref_scan[i]) > 1e-1f) {
            scanOk = false;
            break;
        }
    }
    std::cout << "hierarchical_scan       : " << (scanOk ? "✓ all 1M elements match" : "✗ MISMATCH") << "\n";

    launchCompactAtomic(); cudaDeviceSynchronize();
    int atomicTotal = 0;
    cudaMemcpy(&atomicTotal, d_atomicCount, sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "compact_atomic          : "
              << (atomicTotal == static_cast<int>(h_ref_compact.size()) ? "✓ count matches (" : "✗ count mismatch (")
              << atomicTotal << " kept)\n";

    launchCompactScan(); cudaDeviceSynchronize();
    std::vector<float> h_gpu_compact(h_ref_compact.size());
    cudaMemcpy(h_gpu_compact.data(), d_compact_out, h_ref_compact.size() * sizeof(float), cudaMemcpyDeviceToHost);
    bool compactOk = true;
    for (size_t i = 0; i < h_ref_compact.size(); i++) {
        if (h_gpu_compact[i] != h_ref_compact[i]) {
            compactOk = false;
            break;
        }
    }
    std::cout << "compact_scan (stable)   : " << (compactOk ? "✓ exact order preserved" : "✗ MISMATCH") << "\n\n";

    // ---- timing ----
    std::vector<Bench> benches = {
        {"hierarchical_scan", launchHierarchicalScan, {}},
        {"compact_atomic",    launchCompactAtomic,    {}},
        {"compact_scan",      launchCompactScan,      {}},
    };

    std::cout << "Timing (7 interleaved rounds, median reported):\n";
    runRounds(benches);

    float tScan    = benches[0].median();
    float tAtomic  = benches[1].median();
    float tCScan   = benches[2].median();

    std::cout << "\ncompact_scan vs compact_atomic = "
              << std::fixed << std::setprecision(2) << tAtomic / tCScan << "x faster\n";

    cudaError_t err = cudaGetLastError();
    std::cout << (err == cudaSuccess ? "no CUDA errors\n" : cudaGetErrorString(err));

    cudaFree(d_in);
    cudaFree(d_scan_out);
    cudaFree(d_compact_out);
    cudaFree(d_blockSums);
    cudaFree(d_blockOffsets);
    cudaFree(d_flagsScanned);
    cudaFree(d_atomicCount);
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
