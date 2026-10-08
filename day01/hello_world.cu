#include <stdio.h>

// __global__ = runs on the GPU (device), launched from the CPU (host)
__global__ void hello() {
    printf("hello from gpu to my macbook air m2\n");
}

int main() {
    // <<<blocks, threads per block>>> -> 1 block with 1 thread
    hello<<<1, 1>>>();

    // kernel launches are async: wait for the GPU to finish,
    // otherwise main() exits before the printf buffer is flushed
    cudaDeviceSynchronize();

    return 0;
}
