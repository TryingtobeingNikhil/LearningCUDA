#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define R 8           // radius: each output averages 2R + 1 = 17 inputs
#define BLOCK 256

#define CUDA_CHECK(call)                                              \
    do {                                                              \
        cudaError_t err = (call);                                     \
        if (err != cudaSuccess) {                                     \
            printf("CUDA error %s at %s:%d\n", cudaGetErrorString(err), \
                   __FILE__, __LINE__);                               \
            exit(1);                                                  \
        }                                                             \
    } while (0)

// out[i] = average of in[i-R .. i+R], values outside the array count as 0

// 1) global only: every input element is read 2R+1 times from global memory,
//    once by each of its neighbours (the L1/L2 caches soften this a bit)
__global__ void stencilGlobal(const float *in, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float sum = 0.0f;  // lives in a register: private to this thread, fastest
    for (int k = -R; k <= R; k++) {
        int j = i + k;
        if (j >= 0 && j < n) sum += in[j];
    }
    out[i] = sum / (2 * R + 1);
}

// 2) shared memory: the block copies its BLOCK elements plus R extra on each
//    side (the "halo") into shared memory once, then every thread reads its
//    neighbours from there instead of global memory
__global__ void stencilShared(const float *in, float *out, int n) {
    __shared__ float tile[BLOCK + 2 * R];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int t = threadIdx.x + R;  // position inside the tile

    tile[t] = (i < n) ? in[i] : 0.0f;
    if (threadIdx.x < R) {
        // the first R threads also load the left and right halos
        int left = i - R, right = i + BLOCK;
        tile[t - R] = (left >= 0) ? in[left] : 0.0f;
        tile[t + BLOCK] = (right < n) ? in[right] : 0.0f;
    }
    __syncthreads();

    if (i >= n) return;
    float sum = 0.0f;
    for (int k = -R; k <= R; k++) sum += tile[t + k];
    out[i] = sum / (2 * R + 1);
}

typedef void (*Kernel)(const float *, float *, int);

float timeKernel(Kernel k, int blocks, const float *in, float *out, int n) {
    const int runs = 50;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    k<<<blocks, BLOCK>>>(in, out, n);  // warm-up
    CUDA_CHECK(cudaGetLastError());

    cudaEventRecord(start);
    for (int r = 0; r < runs; r++) k<<<blocks, BLOCK>>>(in, out, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms / runs;
}

int main() {
    int n = 1 << 24;  // 16M floats
    size_t bytes = (size_t)n * sizeof(float);

    float *h_in = (float *)malloc(bytes);
    float *h_out = (float *)malloc(bytes);
    float *h_ref = (float *)malloc(bytes);
    for (int i = 0; i < n; i++) h_in[i] = (float)(i % 100);

    for (int i = 0; i < n; i++) {
        float sum = 0.0f;
        for (int k = -R; k <= R; k++)
            if (i + k >= 0 && i + k < n) sum += h_in[i + k];
        h_ref[i] = sum / (2 * R + 1);
    }

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    int blocks = (n + BLOCK - 1) / BLOCK;
    struct { const char *name; Kernel k; } kernels[] = {
        {"global only  ", stencilGlobal},
        {"shared memory", stencilShared},
    };

    printf("1D stencil, radius %d, %d elements\n\n", R, n);
    for (auto &kn : kernels) {
        CUDA_CHECK(cudaMemset(d_out, 0, bytes));
        float ms = timeKernel(kn.k, blocks, d_in, d_out, n);
        CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

        bool ok = true;
        for (int i = 0; i < n && ok; i++)
            if (fabsf(h_out[i] - h_ref[i]) > 1e-3f) ok = false;
        printf("%s  %7.3f ms  %s\n", kn.name, ms, ok ? "PASSED" : "FAILED");
    }

    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);
    free(h_ref);
    return 0;
}
