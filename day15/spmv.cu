// Day 15: sparse matrix-vector multiplication (CSR SpMV)
// Run in LeetGPU Vector Addition challenge.

#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <functional>
#include <algorithm>
#include <cmath>

// Scalar SpMV: 1 thread per row
__global__ void spmvScalar(int numRows, const int* rowPtr, const int* colIdx,
                           const float* values, const float* x, float* y) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < numRows) {
        float sum = 0.0f;
        int start = rowPtr[row];
        int end   = rowPtr[row + 1];
        for (int i = start; i < end; i++) {
            sum += values[i] * x[colIdx[i]];
        }
        y[row] = sum;
    }
}

// Vector SpMV: 1 warp (32 threads) per row with coalesced loads & shuffle reduce
__global__ void spmvVector(int numRows, const int* rowPtr, const int* colIdx,
                           const float* values, const float* x, float* y) {
    int warpId = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane   = threadIdx.x % 32;

    if (warpId < numRows) {
        int start = rowPtr[warpId];
        int end   = rowPtr[warpId + 1];

        float sum = 0.0f;
        for (int i = start + lane; i < end; i += 32) {
            sum += values[i] * x[colIdx[i]];
        }

        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }

        if (lane == 0) {
            y[warpId] = sum;
        }
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
        std::cout << std::left << std::setw(20) << b.name
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

void spmvCpu(int numRows, const int* rowPtr, const int* colIdx,
             const float* values, const float* x, float* yRef) {
    for (int r = 0; r < numRows; r++) {
        float sum = 0.0f;
        for (int i = rowPtr[r]; i < rowPtr[r + 1]; i++) {
            sum += values[i] * x[colIdx[i]];
        }
        yRef[r] = sum;
    }
}

bool checkResult(const std::vector<float>& ref, const float* d_y, int numRows, const std::string& label) {
    std::vector<float> h_y(numRows);
    cudaMemcpy(h_y.data(), d_y, numRows * sizeof(float), cudaMemcpyDeviceToHost);
    float maxErr = 0.0f;
    for (int i = 0; i < numRows; i++) {
        float err = std::abs(h_y[i] - ref[i]);
        if (err > maxErr) maxErr = err;
    }
    bool ok = (maxErr < 1e-2f);
    std::cout << label << ": " << (ok ? "✓ correct" : "✗ MISMATCH")
              << " (max err = " << maxErr << ")\n";
    return ok;
}

void run_demo() {
    const int NUM_ROWS = 65536;
    const int NUM_COLS = 65536;

    std::cout << "=== Day 15: CSR SpMV & load imbalance ===\n";
    std::cout << "matrix: " << NUM_ROWS << " x " << NUM_COLS << "   avg nnz/row: 32\n\n";

    // generate two matrices: uniform (fixed 32 nnz) vs skewed (power-law)
    std::vector<int> rowPtrUniform(NUM_ROWS + 1);
    std::vector<int> rowPtrSkewed(NUM_ROWS + 1);

    rowPtrUniform[0] = 0;
    rowPtrSkewed[0]  = 0;

    uint32_t seed = 1337;
    auto rnd = [&] {
        seed = seed * 1664525u + 1013904223u;
        return seed;
    };

    for (int r = 0; r < NUM_ROWS; r++) {
        rowPtrUniform[r + 1] = rowPtrUniform[r] + 32;

        // 90% short rows (8 nnz), 10% hub rows (248 nnz) -> average = 32
        int count = (rnd() % 100 < 90) ? 8 : 248;
        rowPtrSkewed[r + 1] = rowPtrSkewed[r] + count;
    }

    int nnzUniform = rowPtrUniform[NUM_ROWS];
    int nnzSkewed  = rowPtrSkewed[NUM_ROWS];

    std::cout << "uniform matrix nnz : " << nnzUniform << "\n";
    std::cout << "skewed matrix nnz  : " << nnzSkewed << " (90% rows=8, 10% rows=248)\n\n";

    std::vector<int> colIdxUniform(nnzUniform);
    std::vector<float> valUniform(nnzUniform);
    for (int i = 0; i < nnzUniform; i++) {
        colIdxUniform[i] = rnd() % NUM_COLS;
        valUniform[i]    = static_cast<float>((rnd() % 100) + 1) / 100.0f;
    }

    std::vector<int> colIdxSkewed(nnzSkewed);
    std::vector<float> valSkewed(nnzSkewed);
    for (int i = 0; i < nnzSkewed; i++) {
        colIdxSkewed[i] = rnd() % NUM_COLS;
        valSkewed[i]    = static_cast<float>((rnd() % 100) + 1) / 100.0f;
    }

    std::vector<float> h_x(NUM_COLS);
    for (int i = 0; i < NUM_COLS; i++) h_x[i] = 1.0f;

    std::vector<float> refUniform(NUM_ROWS);
    std::vector<float> refSkewed(NUM_ROWS);
    spmvCpu(NUM_ROWS, rowPtrUniform.data(), colIdxUniform.data(), valUniform.data(), h_x.data(), refUniform.data());
    spmvCpu(NUM_ROWS, rowPtrSkewed.data(), colIdxSkewed.data(), valSkewed.data(), h_x.data(), refSkewed.data());

    // allocate device buffers
    int *d_rowPtrU, *d_colIdxU, *d_rowPtrS, *d_colIdxS;
    float *d_valU, *d_valS, *d_x, *d_y;

    cudaMalloc(&d_rowPtrU, (NUM_ROWS + 1) * sizeof(int));
    cudaMalloc(&d_colIdxU, nnzUniform * sizeof(int));
    cudaMalloc(&d_valU,    nnzUniform * sizeof(float));

    cudaMalloc(&d_rowPtrS, (NUM_ROWS + 1) * sizeof(int));
    cudaMalloc(&d_colIdxS, nnzSkewed * sizeof(int));
    cudaMalloc(&d_valS,    nnzSkewed * sizeof(float));

    cudaMalloc(&d_x, NUM_COLS * sizeof(float));
    cudaMalloc(&d_y, NUM_ROWS * sizeof(float));

    cudaMemcpy(d_rowPtrU, rowPtrUniform.data(), (NUM_ROWS + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_colIdxU, colIdxUniform.data(), nnzUniform * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_valU,    valUniform.data(),    nnzUniform * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(d_rowPtrS, rowPtrSkewed.data(), (NUM_ROWS + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_colIdxS, colIdxSkewed.data(), nnzSkewed * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_valS,    valSkewed.data(),    nnzSkewed * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(d_x, h_x.data(), NUM_COLS * sizeof(float), cudaMemcpyHostToDevice);

    const int THREADS = 256;
    const int BLOCKS_SCALAR = (NUM_ROWS + THREADS - 1) / THREADS;
    const int BLOCKS_VECTOR = (NUM_ROWS * 32 + THREADS - 1) / THREADS;

    auto launchScalarU = [&] { spmvScalar<<<BLOCKS_SCALAR, THREADS>>>(NUM_ROWS, d_rowPtrU, d_colIdxU, d_valU, d_x, d_y); };
    auto launchVectorU = [&] { spmvVector<<<BLOCKS_VECTOR, THREADS>>>(NUM_ROWS, d_rowPtrU, d_colIdxU, d_valU, d_x, d_y); };
    auto launchScalarS = [&] { spmvScalar<<<BLOCKS_SCALAR, THREADS>>>(NUM_ROWS, d_rowPtrS, d_colIdxS, d_valS, d_x, d_y); };
    auto launchVectorS = [&] { spmvVector<<<BLOCKS_VECTOR, THREADS>>>(NUM_ROWS, d_rowPtrS, d_colIdxS, d_valS, d_x, d_y); };

    std::cout << "Spinning up T4...\n";
    warmUp(launchVectorU);
    std::cout << "Warm-up done.\n\n";

    launchScalarU(); cudaDeviceSynchronize();
    checkResult(refUniform, d_y, NUM_ROWS, "scalar_uniform");

    launchVectorU(); cudaDeviceSynchronize();
    checkResult(refUniform, d_y, NUM_ROWS, "vector_uniform");

    launchScalarS(); cudaDeviceSynchronize();
    checkResult(refSkewed, d_y, NUM_ROWS, "scalar_skewed ");

    launchVectorS(); cudaDeviceSynchronize();
    checkResult(refSkewed, d_y, NUM_ROWS, "vector_skewed ");
    std::cout << "\n";

    std::vector<Bench> benches = {
        {"scalar_uniform", launchScalarU, {}},
        {"vector_uniform", launchVectorU, {}},
        {"scalar_skewed",  launchScalarS, {}},
        {"vector_skewed",  launchVectorS, {}},
    };

    std::cout << "Timing (7 interleaved rounds, median reported):\n";
    runRounds(benches);

    float tScU = benches[0].median();
    float tVecU = benches[1].median();
    float tScS = benches[2].median();
    float tVecS = benches[3].median();

    std::cout << "\nuniform: vector vs scalar = " << std::fixed << std::setprecision(2) << tScU / tVecU << "x faster\n";
    std::cout << "skewed : vector vs scalar = " << std::fixed << std::setprecision(2) << tScS / tVecS << "x faster\n";

    cudaError_t err = cudaGetLastError();
    std::cout << (err == cudaSuccess ? "no CUDA errors\n" : cudaGetErrorString(err));

    cudaFree(d_rowPtrU); cudaFree(d_colIdxU); cudaFree(d_valU);
    cudaFree(d_rowPtrS); cudaFree(d_colIdxS); cudaFree(d_valS);
    cudaFree(d_x); cudaFree(d_y);
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
