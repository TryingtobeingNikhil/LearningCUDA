// Day 09: parallel reduction
//
// how to run: open the Vector Addition challenge on LeetGPU, paste this whole
// file, press Run. solve() runs the experiment once (prints to Console Output),
// then does the real vector add so the challenge's test still passes.
//
// the question: summing N numbers on a GPU naively means one thread loops over
// all N — almost never the right move. the right move is a "reduction tree"
// inside shared memory: each round half the threads pair up, add, and hand one
// value forward. but the first tree layout you'd write has warp divergence.
// today we measure two tree layouts (interleaved vs sequential) and see how
// much the divergence fix is worth.
//
// v2 fixes (after first run):
//   - pass-2 now uses a grid-stride accumulation loop so every partial sum is
//     included (v1 silently dropped 65280 of the 65536 partials → got 65536
//     instead of 16 777 216)
//   - naive_global removed from timed loop (one GPU thread looping over 16M
//     floats is ~32 ms/call × 28 calls = timeout); kept only in correctness check

#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <functional>
#include <algorithm>
#include <numeric>
#include <cmath>

// ---------------------------------------------------------------------------
// kernel 0: naive global — one GPU thread sums all N elements serially.
// used only for a correctness spot-check; excluded from timing because a single
// GPU thread running a 16M-iteration loop takes ~30 ms per call → timeout.
// ---------------------------------------------------------------------------
__global__ void naiveGlobal(const float* in, float* out, int n) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float sum = 0.0f;
        for (int i = 0; i < n; i++) sum += in[i];
        out[0] = sum;
    }
}

// ---------------------------------------------------------------------------
// kernel 1: interleaved addressing — the "obvious" reduction tree
//
// imagine 8 threads with values [3 1 4 1 | 5 9 2 6] in shared memory.
//
// loading: each thread first grid-strides over its block's share of global
// memory and accumulates into a register, then stores to sdata[tid].
// (this also fixes the pass-2 bug: the single pass-2 block's 256 threads
//  each loop over 65536/256 = 256 partials before entering the tree.)
//
// tree, stride=1: active threads are those where tid % 2 == 0
//   → tids 0, 2, 4, 6 active; 1, 3, 5, 7 idle — interleaved!
//   one warp has to execute both the "if true" and "if false" paths → divergence
// tree, stride=2: active tids 0, 4 — even more sparse
// ...gets worse every round.
// ---------------------------------------------------------------------------
__global__ void interleavedReduce(const float* in, float* out, int n) {
    extern __shared__ float sdata[];   // blockDim.x floats, allocated at launch
    int tid = threadIdx.x;

    // grid-stride accumulation: each thread sums every blockDim.x-th element
    // this lets one block handle more than blockDim.x elements correctly
    float acc = 0.0f;
    for (int i = blockIdx.x * blockDim.x + tid; i < n; i += gridDim.x * blockDim.x)
        acc += in[i];
    sdata[tid] = acc;
    __syncthreads();   // wait: every thread must finish loading before anyone adds

    // divergent tree: stride grows 1 → 2 → 4 → ... → blockDim/2
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if (tid % (2 * stride) == 0) {       // ← interleaved active set = divergence
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();   // wait again: every active thread must finish before next round
    }

    if (tid == 0) out[blockIdx.x] = sdata[0];
}

// ---------------------------------------------------------------------------
// kernel 2: sequential addressing — the non-divergent fix
//
// same 8 values, same tree depth, but stride starts at blockDim/2 and halves:
//
// round 1: stride=4, thread 0 adds sdata[0]+sdata[4], ..., thread 3 adds sdata[3]+sdata[7]
//   active tids: 0, 1, 2, 3  — the FIRST HALF of the block, contiguous.
//   inactive tids: 4, 5, 6, 7 — the SECOND HALF, also contiguous.
//   → two contiguous groups = one warp path per group = no divergence!
//
// round 2: stride=2, active tids: 0, 1 — still contiguous.
// round 3: stride=1, active tid:  0 — just thread 0.
//
// same number of additions as interleaved, but the warp executes only one
// code path each round → no masked-off wasted lanes.
// ---------------------------------------------------------------------------
__global__ void sequentialReduce(const float* in, float* out, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;

    // same grid-stride accumulation as interleaved
    float acc = 0.0f;
    for (int i = blockIdx.x * blockDim.x + tid; i < n; i += gridDim.x * blockDim.x)
        acc += in[i];
    sdata[tid] = acc;
    __syncthreads();

    // sequential (non-divergent) tree: stride shrinks blockDim/2 → 1
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {                  // ← contiguous prefix: no divergence
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) out[blockIdx.x] = sdata[0];
}

// ---------------------------------------------------------------------------
// timing helpers (same as day 07/08)
// ---------------------------------------------------------------------------
template <typename Launch>
float timeMs(Launch launch, int runs = 3) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    launch();
    cudaEventRecord(start);
    for (int r = 0; r < runs; r++) launch();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
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
                  << ": " << b.median()
                  << " ms (spread " << b.lo() << " – " << b.hi() << ")\n";
}

template <typename Launch>
void warmUp(Launch launch, float targetMs = 500.0f) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    float elapsed = 0.0f;
    while (elapsed < targetMs) {
        for (int k = 0; k < 10; k++) launch();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// ---------------------------------------------------------------------------
// correctness check: compare GPU answer to std::accumulate (double precision)
// ---------------------------------------------------------------------------
bool checkSum(const std::vector<float>& hostData, float gpuAnswer, const std::string& label) {
    double ref = std::accumulate(hostData.begin(), hostData.end(), 0.0);
    float err  = std::abs(static_cast<float>(ref) - gpuAnswer);
    bool ok    = err / static_cast<float>(std::abs(ref) + 1e-8f) < 0.001f;
    std::cout << label
              << ": GPU=" << gpuAnswer
              << " ref=" << static_cast<float>(ref)
              << " err=" << err << (ok ? " ✓" : " ✗ WRONG") << "\n";
    return ok;
}

// ---------------------------------------------------------------------------
// the experiment
// ---------------------------------------------------------------------------
void run_demo() {
    // N = 2^24 = 16M elements
    const int N       = 1 << 24;
    const int THREADS = 256;
    const int BLOCKS  = (N + THREADS - 1) / THREADS;   // 65536

    std::cout << "=== Day 09: parallel reduction (v2) ===\n";
    std::cout << "N = " << N << "  THREADS = " << THREADS
              << "  BLOCKS = " << BLOCKS << "\n\n";

    // all ones → reference sum = N (easy to check)
    std::vector<float> h_in(N, 1.0f);

    float *d_in, *d_partial, *d_out;
    cudaMalloc(&d_in,      N      * sizeof(float));
    cudaMalloc(&d_partial, BLOCKS * sizeof(float));
    cudaMalloc(&d_out,     1      * sizeof(float));
    cudaMemcpy(d_in, h_in.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    size_t smem = THREADS * sizeof(float);

    // ----- launch helpers -----

    // naive: one GPU thread, all N elements — used ONLY for correctness
    auto launchNaive = [&] {
        naiveGlobal<<<1, 1>>>(d_in, d_out, N);
    };

    // interleaved two-pass:
    //   pass 1 — 65536 blocks each accumulate their chunk via grid-stride, then tree-reduce
    //   pass 2 — 1 block: 256 threads each loop over 65536/256=256 partials, then tree-reduce
    auto launchInterleaved = [&] {
        interleavedReduce<<<BLOCKS, THREADS, smem>>>(d_in,      d_partial, N);
        interleavedReduce<<<1,      THREADS, smem>>>(d_partial, d_out,     BLOCKS);
    };

    // sequential two-pass: same structure, non-divergent tree
    auto launchSequential = [&] {
        sequentialReduce<<<BLOCKS, THREADS, smem>>>(d_in,      d_partial, N);
        sequentialReduce<<<1,      THREADS, smem>>>(d_partial, d_out,     BLOCKS);
    };

    // ----- warm-up -----
    std::cout << "Spinning up T4...\n";
    warmUp(launchSequential);
    std::cout << "Warm-up done.\n\n";

    // ----- one-time correctness checks -----
    float ans;

    launchNaive(); cudaDeviceSynchronize();
    cudaMemcpy(&ans, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    checkSum(h_in, ans, "naive_global ");

    launchInterleaved(); cudaDeviceSynchronize();
    cudaMemcpy(&ans, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    checkSum(h_in, ans, "interleaved  ");

    launchSequential(); cudaDeviceSynchronize();
    cudaMemcpy(&ans, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    checkSum(h_in, ans, "sequential   ");
    std::cout << "\n";

    // ----- timed comparison (interleaved vs sequential only) -----
    // naive_global excluded: ~30 ms/call × 28 calls = timeout
    std::vector<Bench> benches = {
        {"interleaved",  launchInterleaved, {}},
        {"sequential",   launchSequential,  {}},
    };

    std::cout << "Timing (7 interleaved rounds, median reported):\n";
    runRounds(benches);

    float tIlv = benches[0].median();
    float tSeq = benches[1].median();
    std::cout << "\ninterleaved / sequential = " << tIlv / tSeq << "x\n";

    cudaError_t err = cudaGetLastError();
    std::cout << (err == cudaSuccess ? "no CUDA errors\n" : cudaGetErrorString(err));

    cudaFree(d_in);
    cudaFree(d_partial);
    cudaFree(d_out);
}

// ---------------------------------------------------------------------------
// the actual challenge: C = A + B  (must pass LeetGPU's test)
// ---------------------------------------------------------------------------
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
