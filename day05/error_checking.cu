#include <stdio.h>
#include <stdlib.h>

// wrap every CUDA API call: on failure, print the error + file + line and stop
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = (call);                                          \
        if (err != cudaSuccess) {                                          \
            printf("  CUDA error: %s (%s:%d)\n", cudaGetErrorString(err),  \
                   __FILE__, __LINE__);                                    \
        }                                                                  \
    } while (0)
// (a real program would exit(1) here; this demo keeps going to show more cases)

__global__ void fill(float *x, int n, float value) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = value;
}

int main() {
    int n = 1024;
    float *d_x;
    float h_x[1024];
    CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_x, 0, n * sizeof(float)));

    // ---- 1) a bad launch fails silently ----
    printf("1) launch with 2048 threads per block (max is 1024), no checks:\n");
    fill<<<1, 2048>>>(d_x, n, 7.0f);
    cudaMemcpy(h_x, d_x, n * sizeof(float), cudaMemcpyDeviceToHost);
    printf("  no error printed, but x[0] = %.1f (expected 7.0). the kernel never ran\n\n", h_x[0]);
    cudaGetLastError();  // clear the error so the next demo starts clean

    // ---- 2) the same launch, checked ----
    printf("2) same launch, checked with cudaGetLastError() right after:\n");
    fill<<<1, 2048>>>(d_x, n, 7.0f);
    CUDA_CHECK(cudaGetLastError());  // catches launch/config errors immediately
    printf("\n");

    // ---- 3) errors *inside* a kernel show up later ----
    // launches are async: the kernel is still running when the next line runs,
    // so a crash inside it is only reported by the next call that waits for
    // the GPU (cudaDeviceSynchronize, cudaMemcpy, ...)
    printf("3) kernel writes through a null pointer:\n");
    fill<<<4, 256>>>(nullptr, n, 7.0f);
    CUDA_CHECK(cudaGetLastError());
    printf("  launch looked fine...\n");
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("  ...the error only appeared at the sync.\n");
    printf("  this one is 'sticky': every CUDA call after it fails too, so the\n");
    printf("  program has to restart. that's why it's the last demo.\n");

    cudaFree(d_x);
    return 0;
}
