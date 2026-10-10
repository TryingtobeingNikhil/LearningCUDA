#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define TILE 16

#define CUDA_CHECK(call)                                              \
    do {                                                              \
        cudaError_t err = (call);                                     \
        if (err != cudaSuccess) {                                     \
            printf("CUDA error %s at %s:%d\n", cudaGetErrorString(err), \
                   __FILE__, __LINE__);                               \
            exit(1);                                                  \
        }                                                             \
    } while (0)

// A is M x N, B is N x K, C = A * B is M x K (all row-major)

// 1) naive: one thread per output element C[row][col].
//    it walks along row `row` of A and column `col` of B straight from global
//    memory, so every element of A gets read K times and every element of B
//    gets read M times in total, by different threads
__global__ void matmulNaive(const float *A, const float *B, float *C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < K) {
        float sum = 0.0f;
        for (int i = 0; i < N; i++) {
            sum += A[row * N + i] * B[i * K + col];  // A[row][i] * B[i][col]
        }
        C[row * K + col] = sum;
    }
}

// 2) tiled: the block computes a TILE x TILE patch of C. It slides along N one
//    tile at a time: all threads load one tile of A and one tile of B into
//    shared memory together, then each thread uses that shared data TILE times.
//    global memory reads drop by a factor of TILE
__global__ void matmulTiled(const float *A, const float *B, float *C, int M, int N, int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;
    float sum = 0.0f;

    int numTiles = (N + TILE - 1) / TILE;
    for (int t = 0; t < numTiles; t++) {
        // each thread loads one element of each tile (0 if outside the matrix)
        int aCol = t * TILE + tx;
        int bRow = t * TILE + ty;
        As[ty][tx] = (row < M && aCol < N) ? A[row * N + aCol] : 0.0f;
        Bs[ty][tx] = (bRow < N && col < K) ? B[bRow * K + col] : 0.0f;

        __syncthreads();  // both tiles fully loaded before anyone uses them

        for (int i = 0; i < TILE; i++) {
            sum += As[ty][i] * Bs[i][tx];
        }

        __syncthreads();  // everyone done before the next tile overwrites them
    }

    if (row < M && col < K) {
        C[row * K + col] = sum;
    }
}

typedef void (*Kernel)(const float *, const float *, float *, int, int, int);

float timeKernel(Kernel k, dim3 grid, dim3 block, const float *A, const float *B, float *C,
                 int M, int N, int K) {
    const int runs = 20;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    k<<<grid, block>>>(A, B, C, M, N, K);  // warm-up
    CUDA_CHECK(cudaGetLastError());

    cudaEventRecord(start);
    for (int i = 0; i < runs; i++) k<<<grid, block>>>(A, B, C, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms / runs;
}

void matmulCPU(const float *A, const float *B, float *C, int M, int N, int K) {
    for (int r = 0; r < M; r++)
        for (int c = 0; c < K; c++) {
            float sum = 0.0f;
            for (int i = 0; i < N; i++) sum += A[r * N + i] * B[i * K + c];
            C[r * K + c] = sum;
        }
}

// float sums in a different order differ slightly, so compare with a tolerance
bool check(const float *ref, const float *out, int size) {
    for (int i = 0; i < size; i++)
        if (fabsf(ref[i] - out[i]) > 1e-3f * fmaxf(1.0f, fabsf(ref[i]))) return false;
    return true;
}

int main() {
    int M = 1024, N = 768, K = 512;  // different sizes so a mixed-up index shows up
    size_t bytesA = (size_t)M * N * sizeof(float);
    size_t bytesB = (size_t)N * K * sizeof(float);
    size_t bytesC = (size_t)M * K * sizeof(float);

    float *h_A = (float *)malloc(bytesA);
    float *h_B = (float *)malloc(bytesB);
    float *h_C = (float *)malloc(bytesC);
    float *h_ref = (float *)malloc(bytesC);
    srand(42);
    for (int i = 0; i < M * N; i++) h_A[i] = (float)rand() / RAND_MAX * 2 - 1;
    for (int i = 0; i < N * K; i++) h_B[i] = (float)rand() / RAND_MAX * 2 - 1;

    printf("computing CPU reference (takes a second)...\n");
    matmulCPU(h_A, h_B, h_ref, M, N, K);

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytesA));
    CUDA_CHECK(cudaMalloc(&d_B, bytesB));
    CUDA_CHECK(cudaMalloc(&d_C, bytesC));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytesA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytesB, cudaMemcpyHostToDevice));

    dim3 block(TILE, TILE);
    dim3 grid((K + TILE - 1) / TILE, (M + TILE - 1) / TILE);  // x covers columns of C, y covers rows

    struct { const char *name; Kernel k; } kernels[] = {
        {"naive", matmulNaive},
        {"tiled", matmulTiled},
    };

    printf("\nC (%d x %d) = A (%d x %d) * B (%d x %d)\n\n", M, K, M, N, N, K);
    for (auto &kn : kernels) {
        CUDA_CHECK(cudaMemset(d_C, 0, bytesC));
        float ms = timeKernel(kn.k, grid, block, d_A, d_B, d_C, M, N, K);
        CUDA_CHECK(cudaMemcpy(h_C, d_C, bytesC, cudaMemcpyDeviceToHost));

        // each output element needs N multiplies + N adds
        double gflops = 2.0 * M * N * K / (ms / 1000.0) / 1e9;
        printf("%s  %7.3f ms  %7.1f GFLOP/s  %s\n", kn.name, ms, gflops,
               check(h_ref, h_C, M * K) ? "PASSED" : "FAILED");
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_ref);
    return 0;
}
