// Day 07: warp divergence
//
// how to run: open the Vector Addition challenge on LeetGPU, paste this whole
// file, press Run. solve() first runs the divergence demo once (it allocates
// its own big buffers and prints timings to Console Output), then does the
// real vector add so the challenge's test still passes.
// (if your page's solve() signature looks different, keep theirs and just
//  call run_demo() inside it)

#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <vector>

// ---------------------------------------------------------------------------
// the two "paths" a thread can take.
// they use different math (sin vs exp) on purpose: if both paths did the same
// kind of work, the compiler could merge them into one loop and the branch
// would disappear, which would ruin the experiment.
// ---------------------------------------------------------------------------
__device__ float pathA(float x, int iters) {
    for (int k = 0; k < iters; k++) x = __sinf(x) * 1.001f + 0.1f;
    return x;
}

__device__ float pathB(float x, int iters) {
    for (int k = 0; k < iters; k++) x = __expf(-x) * 0.999f + 0.2f;
    return x;
}

// baseline: every thread takes path A, no branch at all
__global__ void noBranch(float* out, int n, int iters) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = pathA(i * 1e-6f, iters);
}

// split by THREAD: neighbours inside the same warp disagree
// (even lanes -> A, odd lanes -> B), so every warp has to run A with the odd
// lanes switched off, then B with the even lanes switched off. cost ~ A + B
__global__ void branchPerThread(float* out, int n, int iters) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = i * 1e-6f;
    if (threadIdx.x % 2 == 0) out[i] = pathA(x, iters);
    else                      out[i] = pathB(x, iters);
}

// split by WARP: the same 50/50 split of work, but all 32 lanes of a warp
// agree (warp 0 -> A, warp 1 -> B, ...). each warp runs only one path. cost ~ A
__global__ void branchPerWarp(float* out, int n, int iters) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = i * 1e-6f;
    if ((threadIdx.x / 32) % 2 == 0) out[i] = pathA(x, iters);
    else                             out[i] = pathB(x, iters);
}

// a tiny branch: ReLU written with an if...
__global__ void reluIf(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (x[i] > 0.0f) y[i] = x[i];
    else             y[i] = 0.0f;
}

// ...and without one. for branches this short the compiler usually emits a
// "select" (compute both, pick one) instead of jumping, so there's no real
// divergence and both versions should take about the same time
__global__ void reluMax(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    y[i] = fmaxf(x[i], 0.0f);
}

// times any launch with cudaEvents (GPU-side clock), averaged over a few runs
template <typename Launch>
float timeMs(Launch launch, int runs = 5) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    launch();                       // warm-up, not timed
    cudaEventRecord(start);
    for (int r = 0; r < runs; r++) launch();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);     // wait until the GPU has reached `stop`
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms / runs;
}

void run_demo() {
    const int n = 1 << 22;          // ~4M threads
    const int iters = 256;          // how much math each path does
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    float* d_out = nullptr;
    cudaMalloc(&d_out, n * sizeof(float));

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "--- warp divergence (" << n << " threads, " << iters << " iterations) ---\n";

    float tNone = timeMs([&] { noBranch<<<blocks, threads>>>(d_out, n, iters); });
    float tWarp = timeMs([&] { branchPerWarp<<<blocks, threads>>>(d_out, n, iters); });
    float tThread = timeMs([&] { branchPerThread<<<blocks, threads>>>(d_out, n, iters); });

    std::cout << "no branch           : " << tNone << " ms\n";
    std::cout << "branch per warp     : " << tWarp << " ms   (no divergence)\n";
    std::cout << "branch per thread   : " << tThread << " ms   (diverges, expect ~2x)\n";

    // ReLU: inputs alternate sign so the if really goes both ways
    std::vector<float> h_x(n);
    for (int i = 0; i < n; i++) h_x[i] = (i % 2 == 0) ? static_cast<float>(i % 7) : -1.0f;
    float *d_x = nullptr, *d_y1 = nullptr, *d_y2 = nullptr;
    cudaMalloc(&d_x, n * sizeof(float));
    cudaMalloc(&d_y1, n * sizeof(float));
    cudaMalloc(&d_y2, n * sizeof(float));
    cudaMemcpy(d_x, h_x.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    float tIf = timeMs([&] { reluIf<<<blocks, threads>>>(d_x, d_y1, n); });
    float tMax = timeMs([&] { reluMax<<<blocks, threads>>>(d_x, d_y2, n); });

    std::vector<float> y1(n), y2(n);
    cudaMemcpy(y1.data(), d_y1, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(y2.data(), d_y2, n * sizeof(float), cudaMemcpyDeviceToHost);
    bool same = (y1 == y2);

    std::cout << "--- tiny branch: ReLU ---\n";
    std::cout << "relu with if        : " << tIf << " ms\n";
    std::cout << "relu with fmaxf     : " << tMax << " ms   " << (same ? "same output" : "DIFFERENT output") << "\n";

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) std::cout << "CUDA error: " << cudaGetErrorString(err) << "\n";

    cudaFree(d_out);
    cudaFree(d_x);
    cudaFree(d_y1);
    cudaFree(d_y2);
}

// ---------------------------------------------------------------------------
// the actual challenge: C = A + B
// ---------------------------------------------------------------------------
__global__ void vectorAdd(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] + B[i];   // only the last warp is split here: harmless
}

extern "C" void solve(const float* A, const float* B, float* C, int N) {
    static bool demoDone = false;    // Submit calls solve() many times; demo runs once
    if (!demoDone) {
        run_demo();
        demoDone = true;
    }
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    vectorAdd<<<blocks, threads>>>(A, B, C, N);
    cudaDeviceSynchronize();
}
