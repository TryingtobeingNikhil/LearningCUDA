#include <stdio.h>
#include <stdlib.h>

// each thread adds one pair of elements: C[i] = A[i] + B[i]
__global__ void vectorAdd(const float *A, const float *B, float *C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {  // the last block can have extra threads past the end
        C[i] = A[i] + B[i];
    }
}

int main() {
    int n = 1 << 20;  // ~1M elements
    size_t bytes = n * sizeof(float);

    // host (CPU) memory
    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_C = (float *)malloc(bytes);
    for (int i = 0; i < n; i++) {
        h_A[i] = i;
        h_B[i] = 2 * i;
    }

    // device (GPU) memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    // CPU -> GPU
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // enough blocks of 256 threads to cover all n elements (rounded up)
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    vectorAdd<<<blocks, threads>>>(d_A, d_B, d_C, n);

    // GPU -> CPU (this waits for the kernel to finish)
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    // check the result
    int errors = 0;
    for (int i = 0; i < n; i++) {
        if (h_C[i] != h_A[i] + h_B[i]) errors++;
    }
    printf("%s (%d errors)\n", errors == 0 ? "PASSED" : "FAILED", errors);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    return 0;
}
