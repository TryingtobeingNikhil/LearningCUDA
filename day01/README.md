# Day 01: hello gpu + vector addition

- [hello_world.cu](hello_world.cu): `__global__` = runs on the GPU, launched from the CPU. `<<<blocks, threads>>>` sets how many threads. Launches are async, so without `cudaDeviceSynchronize()` nothing gets printed.

  ![hello gpu](../images/day01_hello_world.png)

- [vector_addition.cu](vector_addition.cu): one thread per element, `i = blockIdx.x * blockDim.x + threadIdx.x`, with `if (i < n)` so extra threads don't write past the end. Data goes CPU → GPU → CPU with `cudaMemcpy`.

  ![vector addition](../images/day01_vector_addition.png)
