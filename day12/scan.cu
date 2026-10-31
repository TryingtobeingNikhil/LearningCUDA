// Day 12: parallel scan (Kogge-Stone vs Blelloch)
// Run in LeetGPU Vector Addition challenge.

#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <functional>
#include <algorithm>
#include <cmath>

// Kogge-Stone scan: step-efficient (log2(N) steps, double-buffered)
__global__ void scanKoggeStone(const float* in, float* out, int n) {
    __shared__ float sdata[2][1024];
    int tid = threadIdx.x;
    int gid = blockIdx.x * 1024 + tid;

    int pout = 0;
    int pin = 1;

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
}

// Kogge-Stone scan: single-buffer with register temp (2x barriers)
__global__ void scanKoggeStoneSingle(const float* in, float* out, int n) {
    __shared__ float sdata[1024];
    int tid = threadIdx.x;
    int gid = blockIdx.x * 1024 + tid;

    sdata[tid] = (gid < n) ? in[gid] : 0.0f;
    __syncthreads();

    #pragma unroll
    for (int stride = 1; stride < 1024; stride *= 2) {
        float val = 0.0f;
        if (tid >= stride) {
            val = sdata[tid - stride];
        }
        __syncthreads();
        if (tid >= stride) {
            sdata[tid] += val;
        }
        __syncthreads();
    }

    if (gid < n) {
        out[gid] = sdata[tid];
    }
}

// Blelloch scan: work-efficient tree (up-sweep + down-sweep)
__global__ void scanBlelloch(const float* in, float* out, int n) {
    __shared__ float sdata[1024];
    int tid = threadIdx.x; // 512 threads scanning 1024 elements
    int blockOffset = blockIdx.x * 1024;
    int idxA = blockOffset + 2 * tid;
    int idxB = blockOffset + 2 * tid + 1;

    float origA = (idxA < n) ? in[idxA] : 0.0f;
    float origB = (idxB < n) ? in[idxB] : 0.0f;

    sdata[2 * tid]     = origA;
    sdata[2 * tid + 1] = origB;

    // Up-sweep (reduce phase)
    int offset = 1;
    for (int d = 512; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            sdata[bi] += sdata[ai];
        }
        offset *= 2;
    }

    // Clear the root for exclusive scan
    if (tid == 0) {
        sdata[1023] = 0.0f;
    }

    // Down-sweep phase
    for (int d = 1; d <= 512; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            float t = sdata[ai];
            sdata[ai] = sdata[bi];
            sdata[bi] += t;
        }
    }
    __syncthreads();

    // Convert exclusive to inclusive so results match Kogge-Stone
    if (idxA < n) out[idxA] = sdata[2 * tid]     + origA;
    if (idxB < n) out[idxB] = sdata[2 * tid + 1] + origB;
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

bool checkScan(const std::vector<float>& in, const float* d_out, int n, const std::string& label) {
    std::vector<float> h_out(n);
    cudaMemcpy(h_out.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost);

    bool ok = true;
    for (int b = 0; b < n / 1024; b++) {
        float sum = 0.0f;
        for (int i = 0; i < 1024; i++) {
            int idx = b * 1024 + i;
            sum += in[idx];
            if (std::abs(h_out[idx] - sum) > 1e-2f) {
                ok = false;
                break;
            }
        }
        if (!ok) break;
    }
    std::cout << label << ": " << (ok ? "✓ all blocks match reference" : "✗ MISMATCH") << "\n";
    return ok;
}

void run_demo() {
    const int N       = 1 << 24;      // 16M floats (64 MB)
    const int BLOCKS  = N / 1024;     // 16384 blocks of 1024 elements

    std::cout << "=== Day 12: parallel scan (Kogge-Stone vs Blelloch) ===\n";
    std::cout << "N = " << N << " floats (" << (N * sizeof(float)) / (1024 * 1024) << " MB)\n";
    std::cout << "block size = 1024 elements   blocks = " << BLOCKS << "\n\n";

    // initialize input with repeating pattern [1.0f .. 4.0f]
    std::vector<float> h_in(N);
    for (int i = 0; i < N; i++) {
        h_in[i] = static_cast<float>((i % 4) + 1);
    }

    float *d_in, *d_out;
    cudaMalloc(&d_in,  N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));
    cudaMemcpy(d_in, h_in.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    auto launchKoggeDbl = [&] {
        scanKoggeStone<<<BLOCKS, 1024>>>(d_in, d_out, N);
    };

    auto launchKoggeSngl = [&] {
        scanKoggeStoneSingle<<<BLOCKS, 1024>>>(d_in, d_out, N);
    };

    auto launchBlelloch = [&] {
        scanBlelloch<<<BLOCKS, 512>>>(d_in, d_out, N);
    };

    std::cout << "Spinning up T4...\n";
    warmUp(launchKoggeDbl);
    std::cout << "Warm-up done.\n\n";

    launchKoggeDbl(); cudaDeviceSynchronize();
    checkScan(h_in, d_out, N, "kogge_stone_double_buf");

    launchKoggeSngl(); cudaDeviceSynchronize();
    checkScan(h_in, d_out, N, "kogge_stone_single_buf");

    launchBlelloch(); cudaDeviceSynchronize();
    checkScan(h_in, d_out, N, "blelloch_tree         ");
    std::cout << "\n";

    std::vector<Bench> benches = {
        {"kogge_double_buf", launchKoggeDbl,  {}},
        {"kogge_single_buf", launchKoggeSngl, {}},
        {"blelloch_tree",    launchBlelloch,   {}},
    };

    std::cout << "Timing (7 interleaved rounds, median reported):\n";
    runRounds(benches);

    float tDbl  = benches[0].median();
    float tSngl = benches[1].median();
    float tBlel = benches[2].median();

    std::cout << "\nkogge_single vs kogge_double = "
              << std::fixed << std::setprecision(2) << tSngl / tDbl << "x\n";
    std::cout << "blelloch vs kogge_double     = "
              << std::fixed << std::setprecision(2) << tBlel / tDbl << "x\n";

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
