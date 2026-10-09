#include <stdio.h>
#include <stdlib.h>

#define TILE 32

#define CUDA_CHECK(call)                                              \
    do {                                                              \
        cudaError_t err = (call);                                     \
        if (err != cudaSuccess) {                                     \
            printf("CUDA error %s at %s:%d\n", cudaGetErrorString(err), \
                   __FILE__, __LINE__);                               \
            exit(1);                                                  \
        }                                                             \
    } while (0)

// in is rows x cols, out is cols x rows

// 1) naive: reads are coalesced (neighbouring threads read neighbouring
//    columns of the same row), but writes are strided (neighbouring threads
//    write to different rows of out, `rows` floats apart)
__global__ void transposeNaive(const float *in, float *out, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < rows && col < cols) {
        out[col * rows + row] = in[row * cols + col];
    }
}

// 2) tiled: the block loads a TILE x TILE tile into shared memory row by row
//    (coalesced), waits, then writes it back row by row (coalesced) while
//    reading the tile column-wise. The "flip" happens in fast shared memory.
//    PAD = 1 adds an extra column so a column of the tile is spread across
//    different shared-memory banks (no bank conflicts).
template <int PAD>
__global__ void transposeTiled(const float *in, float *out, int rows, int cols) {
    __shared__ float tile[TILE][TILE + PAD];

    int col = blockIdx.x * TILE + threadIdx.x;
    int row = blockIdx.y * TILE + threadIdx.y;
    if (row < rows && col < cols) {
        tile[threadIdx.y][threadIdx.x] = in[row * cols + col];
    }

    __syncthreads();  // whole tile must be loaded before anyone reads it

    // this block now writes the mirrored tile: swap the block coordinates
    col = blockIdx.y * TILE + threadIdx.x;
    row = blockIdx.x * TILE + threadIdx.y;
    if (row < cols && col < rows) {
        out[row * rows + col] = tile[threadIdx.x][threadIdx.y];
    }
}

typedef void (*Kernel)(const float *, float *, int, int);

// runs a kernel a few times and returns the average time in ms
float timeKernel(Kernel k, dim3 grid, dim3 block, const float *d_in, float *d_out,
                 int rows, int cols) {
    const int runs = 50;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    k<<<grid, block>>>(d_in, d_out, rows, cols);  // warm-up
    CUDA_CHECK(cudaGetLastError());

    cudaEventRecord(start);
    for (int i = 0; i < runs; i++) k<<<grid, block>>>(d_in, d_out, rows, cols);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms / runs;
}

bool check(const float *in, const float *out, int rows, int cols) {
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++)
            if (out[c * rows + r] != in[r * cols + c]) return false;
    return true;
}

int main() {
    int rows = 4096, cols = 2048;  // not square, to make sure the indexing is right
    size_t bytes = (size_t)rows * cols * sizeof(float);

    float *h_in = (float *)malloc(bytes);
    float *h_out = (float *)malloc(bytes);
    for (int i = 0; i < rows * cols; i++) h_in[i] = (float)i;

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    dim3 block(TILE, TILE);  // 32 x 32 = 1024 threads, one per element of a tile
    dim3 grid((cols + TILE - 1) / TILE, (rows + TILE - 1) / TILE);

    struct { const char *name; Kernel k; } kernels[] = {
        {"naive               ", transposeNaive},
        {"tiled               ", transposeTiled<0>},
        {"tiled + padding (+1)", transposeTiled<1>},
    };

    printf("transposing a %d x %d matrix\n\n", rows, cols);
    for (auto &kn : kernels) {
        CUDA_CHECK(cudaMemset(d_out, 0, bytes));
        float ms = timeKernel(kn.k, grid, block, d_in, d_out, rows, cols);
        CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

        // each element is read once and written once
        float gbps = 2.0f * bytes / (ms / 1000.0f) / 1e9f;
        printf("%s  %7.3f ms  %7.1f GB/s  %s\n", kn.name, ms, gbps,
               check(h_in, h_out, rows, cols) ? "PASSED" : "FAILED");
    }

    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);
    return 0;
}
