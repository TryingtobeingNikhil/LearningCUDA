# Learning CUDA

Writing GPU code every day and posting what I learn, with a hand-drawn diagram for each day.

| Day | Topic | Code |
|-----|-------|------|
| [01](day01) | hello gpu, vector addition | [hello_world.cu](day01/hello_world.cu), [vector_addition.cu](day01/vector_addition.cu) |
| [02](day02) | 2D indexing, matrix transpose (naive vs shared-memory tiling) | [thread_indexing_2d.cu](day02/thread_indexing_2d.cu), [matrix_transpose.cu](day02/matrix_transpose.cu) |
| [03](day03) | matrix multiplication (naive vs shared-memory tiling) | [matmul.cu](day03/matmul.cu) |
| [04](day04) | memory types: registers, shared, global; 1D stencil with shared memory | [device_memory_info.cu](day04/device_memory_info.cu), [stencil_1d.cu](day04/stencil_1d.cu) |
| [05](day05) | error checking, compute-sanitizer, timing with cudaEvents, bandwidth | [error_checking.cu](day05/error_checking.cu), [buggy_kernel.cu](day05/buggy_kernel.cu), [timing_bandwidth.cu](day05/timing_bandwidth.cu) |
| [06](day06) | SMs, warps, latency hiding, occupancy, grid-stride loops | [warps_and_sms.cu](day06/warps_and_sms.cu), [occupancy.cu](day06/occupancy.cu), [grid_stride.cu](day06/grid_stride.cu) |
| [07](day07) | warp divergence, when branches cost you (and when they don't) | [warp_divergence.cu](day07/warp_divergence.cu) |
| [08](day08) | constant memory (broadcast vs scattered), convolution with a halo | [constant_memory.cu](day08/constant_memory.cu) |
| [09](day09) | parallel reduction: shared-memory tree, sequential addressing, two-pass | [reduction.cu](day09/reduction.cu) |

Each day's folder has its own README with notes and diagrams.
