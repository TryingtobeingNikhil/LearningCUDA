// Day 08: constant memory
//
// how to run: open the Vector Addition challenge on LeetGPU, paste this whole
// file, press Run. solve() runs the experiment once (prints to Console Output),
// then does the real vector add so the challenge's test still passes.
//
// the question: every thread needs the same 64 coefficients (like the weights
// of a convolution filter). where should they live, and does it matter HOW the
// threads of a warp read them?

#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <functional>
#include <algorithm>

constexpr int K = 64;  // number of coefficients

// __constant__ = a small (64 KB) read-only area, filled by the CPU with
// cudaMemcpyToSymbol. it has its own cache, built for one special case:
// all threads of a warp reading the SAME address at the same time
__constant__ float c_coef[K];

// 1) constant memory, broadcast: at step k every lane reads c_coef[k], the
//    same address, so one read is handed to all 32 lanes at once
__global__ void constBroadcast(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i], acc = 0.0f;
    for (int k = 0; k < K; k++) acc = acc * v + c_coef[k];
    y[i] = acc;
}

// 2) constant memory, scattered: at step k, lane L reads c_coef[(k + L) % K].
//    32 lanes want 32 different addresses, and the constant cache can only
//    serve one address at a time, so the reads queue up one after another
__global__ void constScattered(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int lane = threadIdx.x % 32;
    float v = x[i], acc = 0.0f;
    for (int k = 0; k < K; k++) acc = acc * v + c_coef[(k + lane) % K];
    y[i] = acc;
}

// 3) the same coefficients from ordinary global memory, broadcast pattern.
//    after the first touch they sit in the L1 cache, so this may be close to 1)
__global__ void globalBroadcast(const float* x, float* y, const float* coef, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i], acc = 0.0f;
    for (int k = 0; k < K; k++) acc = acc * v + coef[k];
    y[i] = acc;
}

// ---- timing helpers (same as day 07: warm up, interleave, median) ----
template <typename Launch>
float timeMs(Launch launch, int runs = 3) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    launch();                       // warm-up, not timed
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
        for (auto& b : benches) b.times.push_back(timeMs(b.launch));
    for (const auto& b : benches)
        std::cout << std::left << std::setw(22) << b.name << ": " << b.median()
                  << " ms   (spread " << b.lo() << " - " << b.hi() << ")\n";
}

// keep the GPU busy ~0.5 s so its clock is at full speed before we measure
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

void run_demo() {
    const int n = 1 << 22;
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    // inputs between 0 and 1, coefficients small, so nothing blows up
    std::vector<float> h_x(n), h_coef(K);
    for (int i = 0; i < n; i++) h_x[i] = static_cast<float>(i % 1000) / 1000.0f;
    for (int k = 0; k < K; k++) h_coef[k] = 1.0f / (k + 1);

    float *d_x = nullptr, *d_y = nullptr, *d_y2 = nullptr, *d_coef = nullptr;
    cudaMalloc(&d_x, n * sizeof(float));
    cudaMalloc(&d_y, n * sizeof(float));
    cudaMalloc(&d_y2, n * sizeof(float));
    cudaMalloc(&d_coef, K * sizeof(float));
    cudaMemcpy(d_x, h_x.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_coef, h_coef.data(), K * sizeof(float), cudaMemcpyHostToDevice);
    // constant memory is filled by name ("symbol"), not by pointer
    cudaMemcpyToSymbol(c_coef, h_coef.data(), K * sizeof(float));

    auto runBroadcast = [&] { constBroadcast<<<blocks, threads>>>(d_x, d_y, n); };
    warmUp(runBroadcast);

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "--- " << K << " coefficients per element, " << n << " elements ---\n";
    std::vector<Bench> b = {
        {"constant, broadcast", runBroadcast, {}},
        {"constant, scattered", [&] { constScattered<<<blocks, threads>>>(d_x, d_y, n); }, {}},
        {"global, broadcast", [&] { globalBroadcast<<<blocks, threads>>>(d_x, d_y2, d_coef, n); }, {}},
    };
    runRounds(b);
    std::cout << "scattered vs broadcast: " << b[1].median() / b[0].median() << "x slower\n";

    // the two broadcast versions compute exactly the same thing
    runBroadcast();
    std::vector<float> y1(n), y2(n);
    cudaMemcpy(y1.data(), d_y, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(y2.data(), d_y2, n * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "constant vs global results " << (y1 == y2 ? "match" : "DIFFER") << "\n";

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) std::cout << "CUDA error: " << cudaGetErrorString(err) << "\n";

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_y2);
    cudaFree(d_coef);
}

// ---------------------------------------------------------------------------
// the actual challenge: C = A + B
// ---------------------------------------------------------------------------
__global__ void vectorAdd(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] + B[i];
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
