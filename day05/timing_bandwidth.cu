#include <stdio.h>
#include <stdlib.h>
#include <chrono>

#define CUDA_CHECK(call)                                              \
    do {                                                              \
        cudaError_t err = (call);                                     \
        if (err != cudaSuccess) {                                     \
            printf("CUDA error %s at %s:%d\n", cudaGetErrorString(err), \
                   __FILE__, __LINE__);                               \
            exit(1);                                                  \
        }                                                             \
    } while (0)

__global__ void vectorAdd(const float *a, const float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    int n = 1 << 25;  // 32M floats = 128 MB per array
    size_t bytes = (size_t)n * sizeof(float);

    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));
    CUDA_CHECK(cudaMemset(d_a, 0, bytes));
    CUDA_CHECK(cudaMemset(d_b, 0, bytes));

    int threads = 256, blocks = (n + threads - 1) / threads;
    vectorAdd<<<blocks, threads>>>(d_a, d_b, d_c, n);  // warm-up
    CUDA_CHECK(cudaDeviceSynchronize());

    // ---- the wrong way: CPU timer, no sync ----
    auto t0 = std::chrono::high_resolution_clock::now();
    vectorAdd<<<blocks, threads>>>(d_a, d_b, d_c, n);
    auto t1 = std::chrono::high_resolution_clock::now();
    double wrongMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("CPU timer, no sync:  %.3f ms  <- only measured the launch, not the kernel\n", wrongMs);

    // ---- the right way: cudaEvents, recorded on the GPU's own timeline ----
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    vectorAdd<<<blocks, threads>>>(d_a, d_b, d_c, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);  // wait until the GPU reaches `stop`
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    printf("cudaEvent timing:    %.3f ms  <- the real kernel time\n\n", ms);

    // ---- bandwidth vs block size ----
    int dev = 0, memClockKHz, busWidthBits;
    cudaDeviceGetAttribute(&memClockKHz, cudaDevAttrMemoryClockRate, dev);
    cudaDeviceGetAttribute(&busWidthBits, cudaDevAttrGlobalMemoryBusWidth, dev);
    double peakGBs = 2.0 * memClockKHz * 1e3 * (busWidthBits / 8) / 1e9;

    // vector add reads a and b and writes c: 3 arrays of memory traffic
    double movedBytes = 3.0 * bytes;
    printf("block size   time (ms)   GB/s    %% of peak (~%.0f GB/s)\n", peakGBs);
    for (int t = 32; t <= 1024; t *= 2) {
        int b = (n + t - 1) / t;
        const int runs = 20;
        cudaEventRecord(start);
        for (int r = 0; r < runs; r++) vectorAdd<<<b, t>>>(d_a, d_b, d_c, n);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        CUDA_CHECK(cudaGetLastError());
        cudaEventElapsedTime(&ms, start, stop);
        ms /= runs;
        double gbs = movedBytes / (ms / 1000.0) / 1e9;
        printf("%10d   %9.3f   %5.0f   %5.0f%%\n", t, ms, gbs, 100.0 * gbs / peakGBs);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;
}
