#include <stdio.h>
#include <stdlib.h>

#define CUDA_CHECK(call)                                              \
    do {                                                              \
        cudaError_t err = (call);                                     \
        if (err != cudaSuccess) {                                     \
            printf("CUDA error %s at %s:%d\n", cudaGetErrorString(err), \
                   __FILE__, __LINE__);                               \
            exit(1);                                                  \
        }                                                             \
    } while (0)

// one thread per element: needs n / 256 blocks, however big n is
__global__ void addOnePerThread(const float *a, const float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

// grid-stride loop: launch just enough threads to fill the GPU, and let each
// thread walk through the array in steps of (total number of threads).
// works for any n with any grid size
__global__ void addGridStride(const float *a, const float *b, float *c, int n) {
    int stride = gridDim.x * blockDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        c[i] = a[i] + b[i];
    }
}

float timeIt(void (*k)(const float *, const float *, float *, int), int blocks, int threads,
             const float *a, const float *b, float *c, int n) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    k<<<blocks, threads>>>(a, b, c, n);  // warm-up
    CUDA_CHECK(cudaGetLastError());
    cudaEventRecord(start);
    for (int r = 0; r < 20; r++) k<<<blocks, threads>>>(a, b, c, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms / 20;
}

int main() {
    int n = 1 << 25;
    size_t bytes = (size_t)n * sizeof(float);
    float *h_a = (float *)malloc(bytes), *h_b = (float *)malloc(bytes), *h_c = (float *)malloc(bytes);
    for (int i = 0; i < n; i++) { h_a[i] = i % 7; h_b[i] = i % 5; }

    float *a, *b, *c;
    CUDA_CHECK(cudaMalloc(&a, bytes));
    CUDA_CHECK(cudaMalloc(&b, bytes));
    CUDA_CHECK(cudaMalloc(&c, bytes));
    CUDA_CHECK(cudaMemcpy(a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b, h_b, bytes, cudaMemcpyHostToDevice));

    int sms;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
    int threads = 256;

    float ms = timeIt(addOnePerThread, (n + threads - 1) / threads, threads, a, b, c, n);
    printf("one thread per element  (%7d blocks): %.3f ms\n", (n + threads - 1) / threads, ms);

    // a few blocks per SM is enough to keep every SM busy
    for (int perSm = 1; perSm <= 32; perSm *= 2) {
        int blocks = sms * perSm;
        ms = timeIt(addGridStride, blocks, threads, a, b, c, n);
        CUDA_CHECK(cudaMemcpy(h_c, c, bytes, cudaMemcpyDeviceToHost));
        bool ok = true;
        for (int i = 0; i < n && ok; i++) ok = h_c[i] == h_a[i] + h_b[i];
        printf("grid-stride, %2d blocks/SM (%7d blocks): %.3f ms  %s\n", perSm, blocks, ms,
               ok ? "PASSED" : "FAILED");
    }

    cudaFree(a); cudaFree(b); cudaFree(c);
    free(h_a); free(h_b); free(h_c);
    return 0;
}
