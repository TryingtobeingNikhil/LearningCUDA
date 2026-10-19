// Day 10: warp shuffles
//
// how to run: open the Vector Addition challenge on LeetGPU, paste this whole
// file, press Run. solve() runs the experiment once, then does the real vector
// add so the test still passes.
//
// the question: Day 09 used shared memory to exchange values between threads in
// a block. but threads inside the SAME WARP (32 lanes) are always in lock-step
// -- they execute the same instruction together. can we exploit that to skip
// shared memory entirely and read a neighbour's register directly?
//
// yes: __shfl_down_sync(mask, val, offset)
//   "give me the value of the thread whose lane id is (my lane + offset)"
//   no __syncthreads, no shared memory, just a single instruction
//
// kernels compared:
//   shared_reduce  -- Day 09 sequential tree, all via shared memory (baseline)
//   shuffle_reduce -- within each warp use __shfl_down_sync (5 rounds → warp sum),
//                     then collect the 8 warp sums in shared memory and reduce once
//
// the four shuffle instructions (for reference):
//   __shfl_sync     (mask, val, srcLane)  -- read a specific lane (broadcast)
//   __shfl_up_sync  (mask, val, delta)    -- read from lane (tid - delta) (scan)
//   __shfl_down_sync(mask, val, delta)    -- read from lane (tid + delta) (reduction)
//   __shfl_xor_sync (mask, val, laneMask) -- butterfly swap (all-reduce)

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
// baseline: sequential shared-memory reduction (Day 09, unchanged)
// ---------------------------------------------------------------------------
__global__ void sharedReduce(const float* in, float* out, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;

    // grid-stride accumulation into a register
    float acc = 0.0f;
    for (int i = blockIdx.x * blockDim.x + tid; i < n; i += gridDim.x * blockDim.x)
        acc += in[i];
    sdata[tid] = acc;
    __syncthreads();

    // sequential (non-divergent) tree in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = sdata[0];
}

// ---------------------------------------------------------------------------
// warp-level reduction helper
//
// called with one value per thread. after 5 rounds of __shfl_down_sync,
// lane 0 of each warp holds the sum of all 32 lanes.
//
// round 1: lane L reads from lane L+16, adds. now L holds L + (L+16).
// round 2: lane L reads from lane L+8, adds. now L holds 4 consecutive sums.
// ... after 5 rounds, lane 0 = sum of all 32 original values.
//
// WHY no __syncthreads?
//   all 32 lanes of a warp execute in lock-step -- they're always at the
//   same instruction. so when lane 0 calls __shfl_down_sync asking for lane 16's
//   register, lane 16 is guaranteed to already be there. no barrier needed.
//
// 0xffffffff = "all 32 lanes are active, wait for all of them"
// (if some lanes are inactive due to divergence, the mask must match exactly;
//  here we assume a full, non-diverged warp)
// ---------------------------------------------------------------------------
__device__ float warpReduceSum(float val) {
    val += __shfl_down_sync(0xffffffff, val, 16);  // pairs: (0,16),(1,17),...
    val += __shfl_down_sync(0xffffffff, val,  8);  // pairs: (0,8),(1,9),...
    val += __shfl_down_sync(0xffffffff, val,  4);
    val += __shfl_down_sync(0xffffffff, val,  2);
    val += __shfl_down_sync(0xffffffff, val,  1);  // lane 0 = full warp sum
    return val;
}

// ---------------------------------------------------------------------------
// shuffle-based block reduction
//
// for a 256-thread block: 256 / 32 = 8 warps.
// step 1: warpReduceSum -- each warp collapses to one value in lane 0 (no smem).
// step 2: the 8 lane-0 values go into a tiny shared array (only 8 floats!).
// step 3: one __syncthreads, then the FIRST warp (threads 0-31) loads those 8
//         values and does one more warpReduceSum -- giving the block total.
//
// shared memory used: 8 floats = 32 bytes (vs 256 floats = 1024 bytes for Day 09)
// ---------------------------------------------------------------------------
__global__ void shuffleReduce(const float* in, float* out, int n) {
    // shared memory: one slot per warp (THREADS/32 = 8)
    // declared with fixed size 32 to handle up to 1024 threads/block
    __shared__ float warpSums[32];

    int tid    = threadIdx.x;
    int lane   = tid % 32;        // which lane am I within my warp?
    int warpId = tid / 32;        // which warp am I in within this block?

    // grid-stride accumulation (same as shared version)
    float acc = 0.0f;
    for (int i = blockIdx.x * blockDim.x + tid; i < n; i += gridDim.x * blockDim.x)
        acc += in[i];

    // step 1: warp-level reduce (no sync needed -- all in registers)
    acc = warpReduceSum(acc);

    // step 2: lane 0 of each warp writes its sum to shared memory
    if (lane == 0) warpSums[warpId] = acc;

    // step 3: wait for all warps to write, then the first warp reduces
    __syncthreads();

    // only the first warp participates; load 0 if there are fewer warps than 32
    int numWarps = blockDim.x / 32;
    acc = (tid < numWarps) ? warpSums[tid] : 0.0f;

    // step 4: warp 0 reduces the warp sums -- at most 32 of them, one shuffle pass
    if (warpId == 0) acc = warpReduceSum(acc);

    if (tid == 0) out[blockIdx.x] = acc;
}

// ---------------------------------------------------------------------------
// timing helpers (unchanged from Day 07–09)
// ---------------------------------------------------------------------------
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
    float median() const { std::vector<float> t=times; std::sort(t.begin(),t.end()); return t[t.size()/2]; }
    float lo()     const { return *std::min_element(times.begin(), times.end()); }
    float hi()     const { return *std::max_element(times.begin(), times.end()); }
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

bool checkSum(const std::vector<float>& h, float gpu, const std::string& label) {
    double ref = std::accumulate(h.begin(), h.end(), 0.0);
    float err  = std::abs(static_cast<float>(ref) - gpu);
    bool ok    = err / (static_cast<float>(std::abs(ref)) + 1e-8f) < 0.001f;
    std::cout << label << ": GPU=" << gpu << " ref=" << static_cast<float>(ref)
              << " err=" << err << (ok ? " ✓" : " ✗ WRONG") << "\n";
    return ok;
}

// ---------------------------------------------------------------------------
// demo: show what __shfl_sync can do besides reduction
// reads one value from lane 0 and broadcasts it to every lane in the warp
// ---------------------------------------------------------------------------
__global__ void shuffleBroadcastDemo(float* out, int n) {
    int tid  = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x % 32;

    // lane 0 has the "source" value; all others receive it
    float myVal = (lane == 0) ? 42.0f : 0.0f;
    float bcast = __shfl_sync(0xffffffff, myVal, 0);  // srcLane = 0

    if (tid < n) out[tid] = bcast;   // every lane should write 42.0
}

// ---------------------------------------------------------------------------
// the experiment
// ---------------------------------------------------------------------------
void run_demo() {
    const int N       = 1 << 24;   // 16M floats
    const int THREADS = 256;
    const int BLOCKS  = (N + THREADS - 1) / THREADS;

    std::cout << "=== Day 10: warp shuffles ===\n";
    std::cout << "N = " << N << "  THREADS = " << THREADS
              << "  BLOCKS = " << BLOCKS << "\n";
    std::cout << "warps per block = " << THREADS/32 << "  "
              << "(shuffle uses only " << THREADS/32 << " shared floats vs "
              << THREADS << " for shared_reduce)\n\n";

    std::vector<float> h_in(N, 1.0f);

    float *d_in, *d_partial, *d_out;
    cudaMalloc(&d_in,      N      * sizeof(float));
    cudaMalloc(&d_partial, BLOCKS * sizeof(float));
    cudaMalloc(&d_out,     1      * sizeof(float));
    cudaMemcpy(d_in, h_in.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    size_t smemShared  = THREADS * sizeof(float);   // 1024 bytes per block
    size_t smemShuffle = 0;                          // dynamic smem not used

    // two-pass launchers (same structure as Day 09)
    auto launchShared = [&] {
        sharedReduce<<<BLOCKS, THREADS, smemShared>>>(d_in,      d_partial, N);
        sharedReduce<<<1,      THREADS, smemShared>>>(d_partial, d_out,     BLOCKS);
    };

    auto launchShuffle = [&] {
        shuffleReduce<<<BLOCKS, THREADS>>>(d_in,      d_partial, N);
        shuffleReduce<<<1,      THREADS>>>(d_partial, d_out,     BLOCKS);
    };

    // ---- warm-up ----
    std::cout << "Spinning up T4...\n";
    warmUp(launchShuffle);
    std::cout << "Warm-up done.\n\n";

    // ---- correctness ----
    float ans;

    launchShared(); cudaDeviceSynchronize();
    cudaMemcpy(&ans, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    checkSum(h_in, ans, "shared_reduce ");

    launchShuffle(); cudaDeviceSynchronize();
    cudaMemcpy(&ans, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    checkSum(h_in, ans, "shuffle_reduce");

    // broadcast demo: every element should be 42.0
    {
        float *d_demo;
        const int DEMO_N = 64;
        cudaMalloc(&d_demo, DEMO_N * sizeof(float));
        shuffleBroadcastDemo<<<1, DEMO_N>>>(d_demo, DEMO_N);
        cudaDeviceSynchronize();
        std::vector<float> h_demo(DEMO_N);
        cudaMemcpy(h_demo.data(), d_demo, DEMO_N * sizeof(float), cudaMemcpyDeviceToHost);
        bool ok = true;
        for (float v : h_demo) if (v != 42.0f) { ok = false; break; }
        std::cout << "shuffle_broadcast (all lanes should see 42): "
                  << (ok ? "✓ all 42" : "✗ WRONG") << "\n\n";
        cudaFree(d_demo);
    }

    // ---- timing ----
    std::vector<Bench> benches = {
        {"shared_reduce",  launchShared,  {}},
        {"shuffle_reduce", launchShuffle, {}},
    };

    std::cout << "Timing (7 interleaved rounds, median reported):\n";
    runRounds(benches);

    float tShared  = benches[0].median();
    float tShuffle = benches[1].median();
    std::cout << "\nshuffle / shared = " << tShuffle / tShared << "x\n";
    std::cout << "(< 1.0 means shuffle is faster)\n";

    cudaError_t err = cudaGetLastError();
    std::cout << (err == cudaSuccess ? "no CUDA errors\n" : cudaGetErrorString(err));

    cudaFree(d_in); cudaFree(d_partial); cudaFree(d_out);
}

// ---------------------------------------------------------------------------
// the actual challenge: C = A + B
// ---------------------------------------------------------------------------
__global__ void vectorAdd(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] + B[i];
}

extern "C" void solve(const float* A, const float* B, float* C, int N) {
    static bool demoDone = false;
    if (!demoDone) { run_demo(); demoDone = true; }
    int threads = 256;
    int blocks  = (N + threads - 1) / threads;
    vectorAdd<<<blocks, threads>>>(A, B, C, N);
    cudaDeviceSynchronize();
}
