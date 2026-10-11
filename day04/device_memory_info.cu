#include <stdio.h>

// prints how much of each memory type the GPU has, so the
// "fast but tiny vs big but slow" trade-off has real numbers on it
int main() {
    int dev = 0;
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, dev);

    int memClockKHz, busWidthBits;
    cudaDeviceGetAttribute(&memClockKHz, cudaDevAttrMemoryClockRate, dev);
    cudaDeviceGetAttribute(&busWidthBits, cudaDevAttrGlobalMemoryBusWidth, dev);
    // x2 because GDDR/HBM transfers data twice per clock
    double peakGBs = 2.0 * memClockKHz * 1e3 * (busWidthBits / 8) / 1e9;

    printf("%s, %d SMs, warp size %d\n\n", p.name, p.multiProcessorCount, p.warpSize);

    printf("registers (per thread, fastest)\n");
    printf("  per SM:     %d x 32-bit  (%d KB)\n", p.regsPerMultiprocessor, p.regsPerMultiprocessor * 4 / 1024);
    printf("  per block:  %d x 32-bit\n\n", p.regsPerBlock);

    printf("shared memory (per block, fast)\n");
    printf("  per SM:     %zu KB\n", p.sharedMemPerMultiprocessor / 1024);
    printf("  per block:  %zu KB\n\n", p.sharedMemPerBlock / 1024);

    printf("constant memory (read-only, cached)\n");
    printf("  total:      %zu KB\n\n", p.totalConstMem / 1024);

    printf("L2 cache (shared by all SMs)\n");
    printf("  total:      %d KB\n\n", p.l2CacheSize / 1024);

    printf("global memory (everyone + CPU, big but slow)\n");
    printf("  total:      %.1f GB\n", p.totalGlobalMem / 1e9);
    printf("  peak bandwidth: ~%.0f GB/s\n", peakGBs);
    return 0;
}
