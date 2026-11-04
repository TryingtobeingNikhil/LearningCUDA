#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

// ── 1. baseline: 1 thread computes 1 output (16x16 shared tile) ───────────────
const int TILE1 = 16;
__global__ void matmulTiled(const float* __restrict__ A, const float* __restrict__ B,
                            float* __restrict__ C, int M, int N, int K) {
    __shared__ float As[TILE1][TILE1];
    __shared__ float Bs[TILE1][TILE1];

    int row = blockIdx.y * TILE1 + threadIdx.y;
    int col = blockIdx.x * TILE1 + threadIdx.x;

    float sum = 0.0f;
    for (int t = 0; t < K; t += TILE1) {
        As[threadIdx.y][threadIdx.x] = (row < M && (t + threadIdx.x) < K) ? A[row * K + t + threadIdx.x] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = ((t + threadIdx.y) < K && col < N) ? B[(t + threadIdx.y) * N + col] : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE1; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = sum;
}

// ── 2. 1D register tiling: 1 thread computes 1x4 outputs along row ─────────────
__global__ void matmul1DCoarsened(const float* __restrict__ A, const float* __restrict__ B,
                                  float* __restrict__ C, int M, int N, int K) {
    __shared__ float As[16][16];
    __shared__ float Bs[16][64];

    int row = blockIdx.y * 16 + threadIdx.y;
    int col_base = blockIdx.x * 64 + threadIdx.x * 4;

    float c_reg[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int t = 0; t < K; t += 16) {
        As[threadIdx.y][threadIdx.x] = (row < M && (t + threadIdx.x) < K) ? A[row * K + t + threadIdx.x] : 0.0f;

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int b_col = col_base + i;
            Bs[threadIdx.y][threadIdx.x * 4 + i] = ((t + threadIdx.y) < K && b_col < N) ? B[(t + threadIdx.y) * N + b_col] : 0.0f;
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < 16; k++) {
            float a = As[threadIdx.y][k];
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                c_reg[i] += a * Bs[k][threadIdx.x * 4 + i];
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int c = col_base + i;
        if (row < M && c < N) C[row * N + c] = c_reg[i];
    }
}

// ── 3. 2D register tiling: 1 thread computes 4x4 outputs (outer product) ───────
const int BM = 64, BN = 64, BK = 8;
const int TM = 4, TN = 4;
__global__ void matmul2DRegisterTiled(const float* __restrict__ A, const float* __restrict__ B,
                                      float* __restrict__ C, int M, int N, int K) {
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    int tid = threadIdx.y * 16 + threadIdx.x; // 0..255

    // cooperative loading of As (64x8 = 512 elements; 2 elements per thread)
    int as_r = tid / BK;
    int as_c = tid % BK;

    // cooperative loading of Bs (8x64 = 512 elements; 2 elements per thread)
    int bs_r = tid / BN;
    int bs_c = tid % BN;

    float c_reg[TM][TN] = {0.0f};

    for (int t = 0; t < K; t += BK) {
        int a_g_row0 = blockIdx.y * BM + as_r;
        int a_g_row1 = a_g_row0 + 32;
        int a_g_col  = t + as_c;
        As[as_r][as_c]      = (a_g_row0 < M && a_g_col < K) ? A[a_g_row0 * K + a_g_col] : 0.0f;
        As[as_r + 32][as_c] = (a_g_row1 < M && a_g_col < K) ? A[a_g_row1 * K + a_g_col] : 0.0f;

        int b_g_row0 = t + bs_r;
        int b_g_row1 = b_g_row0 + 4;
        int b_g_col  = blockIdx.x * BN + bs_c;
        Bs[bs_r][bs_c]     = (b_g_row0 < K && b_g_col < N) ? B[b_g_row0 * N + b_g_col] : 0.0f;
        Bs[bs_r + 4][bs_c] = (b_g_row1 < K && b_g_col < N) ? B[b_g_row1 * N + b_g_col] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; k++) {
            float a_frag[TM];
            float b_frag[TN];

            #pragma unroll
            for (int i = 0; i < TM; i++) {
                a_frag[i] = As[threadIdx.y * TM + i][k];
            }
            #pragma unroll
            for (int j = 0; j < TN; j++) {
                b_frag[j] = Bs[k][threadIdx.x * TN + j];
            }

            #pragma unroll
            for (int i = 0; i < TM; i++) {
                #pragma unroll
                for (int j = 0; j < TN; j++) {
                    c_reg[i][j] += a_frag[i] * b_frag[j];
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM; i++) {
        int r = blockIdx.y * BM + threadIdx.y * TM + i;
        #pragma unroll
        for (int j = 0; j < TN; j++) {
            int c = blockIdx.x * BN + threadIdx.x * TN + j;
            if (r < M && c < N) C[r * N + c] = c_reg[i][j];
        }
    }
}

// ── timing helper ─────────────────────────────────────────────────────────────

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
    const int M = 1024, N = 1024, K = 1024;
    size_t bytesA = (size_t)M * K * sizeof(float);
    size_t bytesB = (size_t)K * N * sizeof(float);
    size_t bytesC = (size_t)M * N * sizeof(float);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytesA);
    cudaMalloc(&d_B, bytesB);
    cudaMalloc(&d_C, bytesC);

    std::vector<float> h_A(M * K, 1.0f);
    std::vector<float> h_B(K * N, 0.5f);
    cudaMemcpy(d_A, h_A.data(), bytesA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), bytesB, cudaMemcpyHostToDevice);

    dim3 blockTiled(TILE1, TILE1);
    dim3 gridTiled((N + TILE1 - 1) / TILE1, (M + TILE1 - 1) / TILE1);

    dim3 block1D(16, 16);
    dim3 grid1D((N + 63) / 64, (M + 15) / 16);

    dim3 block2D(16, 16);
    dim3 grid2D((N + BN - 1) / BN, (M + BM - 1) / BM);

    // warm-up
    matmulTiled<<<gridTiled, blockTiled>>>(d_A, d_B, d_C, M, N, K);
    matmul1DCoarsened<<<grid1D, block1D>>>(d_A, d_B, d_C, M, N, K);
    matmul2DRegisterTiled<<<grid2D, block2D>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();
    std::cout << "Warm-up done.\n\n";

    auto r_tiled = time7([&]{ matmulTiled<<<gridTiled, blockTiled>>>(d_A, d_B, d_C, M, N, K); cudaDeviceSynchronize(); });
    auto r_1d    = time7([&]{ matmul1DCoarsened<<<grid1D, block1D>>>(d_A, d_B, d_C, M, N, K); cudaDeviceSynchronize(); });
    auto r_2d    = time7([&]{ matmul2DRegisterTiled<<<grid2D, block2D>>>(d_A, d_B, d_C, M, N, K); cudaDeviceSynchronize(); });

    double total_flops = 2.0 * (double)M * (double)N * (double)K;
    double gflops_tiled = total_flops / ((double)r_tiled.ms_med * 1e-3) / 1e9;
    double gflops_1d    = total_flops / ((double)r_1d.ms_med * 1e-3) / 1e9;
    double gflops_2d    = total_flops / ((double)r_2d.ms_med * 1e-3) / 1e9;

    std::cout << "=== Day 17: matrix multiplication II — register tiling ===\n";
    std::cout << "matrix: " << M << " x " << K << " * " << K << " x " << N << " (" << (bytesC/1024/1024) << " MB result)\n\n";

    std::cout << "Timing (7 interleaved rounds, median reported):\n";
    std::printf("shared_tiled (1x1)  : %.3f ms (spread %.3f – %.3f)  →  %.1f GFLOPS\n",
                r_tiled.ms_med, r_tiled.ms_lo, r_tiled.ms_hi, gflops_tiled);
    std::printf("coarsened_1d (1x4)  : %.3f ms (spread %.3f – %.3f)  →  %.1f GFLOPS\n",
                r_1d.ms_med, r_1d.ms_lo, r_1d.ms_hi, gflops_1d);
    std::printf("reg_tiled_2d (4x4)  : %.3f ms (spread %.3f – %.3f)  →  %.1f GFLOPS\n",
                r_2d.ms_med, r_2d.ms_lo, r_2d.ms_hi, gflops_2d);

    std::printf("\nspeedup (1D vs shared) = %.2fx\n", r_tiled.ms_med / r_1d.ms_med);
    std::printf("speedup (2D vs shared) = %.2fx faster\n", r_tiled.ms_med / r_2d.ms_med);

    // correctness check against reference value (1.0 * 0.5 * 1024 = 512.0f)
    std::vector<float> h_C(M * N);
    matmul2DRegisterTiled<<<grid2D, block2D>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();
    cudaMemcpy(h_C.data(), d_C, bytesC, cudaMemcpyDeviceToHost);

    float max_err = 0.0f;
    for (int i = 0; i < M * N; i++) {
        float err = std::abs(h_C[i] - 512.0f);
        if (err > max_err) max_err = err;
    }
    std::printf("\ncorrectness: max err = %g (%s)\n", max_err, max_err < 1e-3f ? "✓ passed" : "✗ fail");

    cudaError_t err = cudaGetLastError();
    std::cout << (err == cudaSuccess ? "no CUDA errors\n" : cudaGetErrorString(err));

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
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
