#include <stdio.h>
#include <stdlib.h>

// which SM (streaming multiprocessor) is this thread running on?
// %smid is a special hardware register you can read with inline PTX
__device__ unsigned int smId() {
    unsigned int id;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(id));
    return id;
}

__global__ void whereAmI(int *blockToSm) {
    int warp = threadIdx.x / 32;  // threads are grouped into warps of 32
    int lane = threadIdx.x % 32;  // position inside the warp

    if (threadIdx.x == 0) blockToSm[blockIdx.x] = smId();

    // print a few threads of block 0 to see the warp / lane split
    if (blockIdx.x == 0 && (lane == 0 || lane == 31)) {
        printf("block 0, thread %3d -> warp %d, lane %2d\n", threadIdx.x, warp, lane);
    }
}

int main() {
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);
    printf("%s: %d SMs, warp size %d, max %d threads per SM\n\n",
           p.name, p.multiProcessorCount, p.warpSize, p.maxThreadsPerMultiProcessor);

    int blocks = 2 * p.multiProcessorCount;
    int threads = 128;  // = 4 warps per block
    int *d_map, *h_map = (int *)malloc(blocks * sizeof(int));
    cudaMalloc(&d_map, blocks * sizeof(int));

    whereAmI<<<blocks, threads>>>(d_map);
    cudaDeviceSynchronize();
    cudaMemcpy(h_map, d_map, blocks * sizeof(int), cudaMemcpyDeviceToHost);

    // a whole block always runs on ONE SM; the hardware decides which one
    printf("\nblock -> SM (first 12 blocks):\n");
    for (int b = 0; b < 12 && b < blocks; b++) printf("  block %2d -> SM %2d\n", b, h_map[b]);

    int *perSm = (int *)calloc(p.multiProcessorCount, sizeof(int));
    for (int b = 0; b < blocks; b++) perSm[h_map[b]]++;
    int used = 0;
    for (int s = 0; s < p.multiProcessorCount; s++) used += perSm[s] > 0;
    printf("\n%d blocks were spread across %d of the %d SMs\n", blocks, used, p.multiProcessorCount);

    cudaFree(d_map);
    free(h_map);
    free(perSm);
    return 0;
}
