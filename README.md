# Learning CUDA

Writing GPU code every day and posting what I learn, with a hand-drawn diagram for each day.

## Progress

### Day 01

- wrote my first two kernels:
  - [hello_world.cu](day01/hello_world.cu)
  - [vector_addition.cu](day01/vector_addition.cu)

- `__global__` marks a function that runs on the GPU and is launched from the CPU. `<<<blocks, threads>>>` says how many threads to launch. Kernel launches are async, so without `cudaDeviceSynchronize()` the program can exit before the GPU's `printf` shows up.

  ![hello gpu](images/day01_hello_world.png)

- vector addition: one thread per element. Each thread finds its element with `i = blockIdx.x * blockDim.x + threadIdx.x`. The `if (i < n)` check stops the extra threads in the last block from writing past the end. The data goes CPU → GPU with `cudaMemcpy`, gets added on the GPU, then gets copied back.

  ![vector addition](images/day01_vector_addition.png)

### Day 02

- threads can be laid out in 1D, 2D or 3D (`dim3` has `.x`, `.y`, `.z`), which makes it natural to give each thread one cell of a matrix: `row = blockIdx.y * blockDim.y + threadIdx.y`, `col = blockIdx.x * blockDim.x + threadIdx.x`, then `idx = row * width + col`.
  - [thread_indexing_2d.cu](day02/thread_indexing_2d.cu) prints which block wrote each cell and its flat index

  ![2d indexing](images/day02_2d_indexing.png)

- matrix transpose: [matrix_transpose.cu](day02/matrix_transpose.cu), three versions timed side by side
  - **naive**: gives the right answer, but while reads are coalesced (neighbouring threads read neighbouring addresses), writes go down a column, so neighbouring threads write a whole row apart
  - **tiled**: each block loads a 32×32 tile into shared memory row by row, waits at `__syncthreads()` (a barrier for every thread in the block), then writes the flipped tile out row by row. Both reads and writes are coalesced, and the flip happens in fast shared memory
  - **tiled + padding**: `tile[32][33]` instead of `tile[32][32]`. Reading a tile column would otherwise hit the same shared-memory bank 32 times (a bank conflict); the extra column shifts each row onto a different bank

  ![transpose](images/day02_transpose.png)

- big lesson: GPU speed isn't just about doing less math, it's about how threads touch memory
