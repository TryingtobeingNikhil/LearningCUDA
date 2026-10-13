#include <stdio.h>

// occupancy = how many warps are actually living on an SM / the max it can hold.
// more resident warps = more chances to hide memory latency.
// it's capped by whichever per-SM resource runs out first:
//   threads, registers, shared memory, or the max number of blocks

// 1) light kernel: few registers, no shared memory
__global__ void light(float *x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = x[i] * 2.0f;
}

// 2) register-heavy kernel: lots of independent values kept alive at once,
//    so the compiler needs many registers per thread
__global__ void heavyRegisters(float *x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float acc[48];
#pragma unroll
    for (int k = 0; k < 48; k++) acc[k] = x[i] * (k + 1);
#pragma unroll
    for (int r = 0; r < 8; r++)
#pragma unroll
        for (int k = 0; k < 48; k++) acc[k] = acc[k] * acc[(k + 7) % 48] + 1.0f;
    float sum = 0.0f;
#pragma unroll
    for (int k = 0; k < 48; k++) sum += acc[k];
    x[i] = sum;
}

// 3) shared-memory-heavy kernel: asks for a big dynamic shared memory buffer
__global__ void heavyShared(float *x, int n) {
    extern __shared__ float buf[];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    buf[threadIdx.x] = (i < n) ? x[i] : 0.0f;
    __syncthreads();
    if (i < n) x[i] = buf[blockDim.x - 1 - threadIdx.x];
}

void report(const char *name, const void *kernel, size_t sharedPerBlock, const cudaDeviceProp &p) {
    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, kernel);
    int maxWarps = p.maxThreadsPerMultiProcessor / p.warpSize;

    printf("%s: %d registers/thread, %zu bytes shared/block\n", name, attr.numRegs, sharedPerBlock);
    printf("  block size   blocks/SM   warps/SM   occupancy\n");
    for (int t = 64; t <= 1024; t *= 2) {
        int blocksPerSm;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSm, kernel, t, sharedPerBlock);
        int warps = blocksPerSm * t / p.warpSize;
        printf("  %10d   %9d   %8d   %8.0f%%\n", t, blocksPerSm, warps, 100.0 * warps / maxWarps);
    }

    int minGrid, bestBlock;
    cudaOccupancyMaxPotentialBlockSize(&minGrid, &bestBlock, kernel, sharedPerBlock);
    printf("  CUDA suggests block size %d\n\n", bestBlock);
}

int main() {
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);
    printf("%s per SM limits: %d threads (%d warps), %d blocks, %d registers, %zu KB shared\n\n",
           p.name, p.maxThreadsPerMultiProcessor, p.maxThreadsPerMultiProcessor / p.warpSize,
           p.maxBlocksPerMultiProcessor, p.regsPerMultiprocessor, p.sharedMemPerMultiprocessor / 1024);

    report("light", (const void *)light, 0, p);
    report("heavyRegisters", (const void *)heavyRegisters, 0, p);
    report("heavyShared (16 KB)", (const void *)heavyShared, 16 * 1024, p);

    // try one by hand before reading the table above:
    // 256 threads x 64 registers = 16384 registers per block
    // 65536 registers per SM / 16384 = 4 blocks = 1024 threads = 32 warps = 100%
    // at 128 registers/thread that halves: 2 blocks, 16 warps, 50%
    return 0;
}
