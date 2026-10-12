#include <stdio.h>
#include <stdlib.h>

// this program has a bug, but it prints PASSED and reports no CUDA error.
// run it normally, then run it under compute-sanitizer:
//   compute-sanitizer ./buggy
// and see if you can spot the bug before reading the comment at the bottom.

__global__ void scale(const float *in, float *out, int n, float s) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i <= n) {
        out[i] = in[i] * s;
    }
}

int main() {
    int n = 1000;  // not a multiple of the block size, so extra threads exist
    size_t bytes = n * sizeof(float);

    float *h_in = (float *)malloc(bytes);
    float *h_out = (float *)malloc(bytes);
    for (int i = 0; i < n; i++) h_in[i] = (float)i;

    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    scale<<<blocks, threads>>>(d_in, d_out, n, 2.0f);
    cudaError_t err = cudaDeviceSynchronize();
    printf("CUDA says: %s\n", cudaGetErrorString(err));

    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    bool ok = true;
    for (int i = 0; i < n; i++)
        if (h_out[i] != h_in[i] * 2.0f) ok = false;
    printf("%s\n", ok ? "PASSED" : "FAILED");

    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);
    return 0;
}

// the bug: `i <= n` should be `i < n`. thread i = n reads in[n] and writes
// out[n], one element past the end of both arrays. cudaMalloc usually hands
// out a bit more memory than asked for, so nothing crashes and the answer
// looks right. compute-sanitizer catches it as an invalid global read/write.
