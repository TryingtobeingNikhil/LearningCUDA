#include <stdio.h>

// a 6 x 8 matrix covered by a 2 x 2 grid of 4 x 3 blocks
#define WIDTH  8
#define HEIGHT 6

// every thread writes which block it belongs to, and its flat index
__global__ void whoAmI(int *blockOf, int *flatIdx) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < HEIGHT && col < WIDTH) {
        int idx = row * WIDTH + col;  // 2D (row, col) -> 1D memory index
        blockOf[idx] = blockIdx.y * gridDim.x + blockIdx.x;
        flatIdx[idx] = idx;
    }
}

void printMatrix(const char *name, int *m) {
    printf("%s:\n", name);
    for (int r = 0; r < HEIGHT; r++) {
        for (int c = 0; c < WIDTH; c++) printf("%3d", m[r * WIDTH + c]);
        printf("\n");
    }
    printf("\n");
}

int main() {
    size_t bytes = WIDTH * HEIGHT * sizeof(int);
    int h_block[WIDTH * HEIGHT], h_idx[WIDTH * HEIGHT];
    int *d_block, *d_idx;
    cudaMalloc(&d_block, bytes);
    cudaMalloc(&d_idx, bytes);

    // dim3 also has a .z, for 3D data (volumes, video, ...)
    dim3 block(4, 3);  // 4 threads in x, 3 in y = 12 threads per block
    dim3 grid((WIDTH + block.x - 1) / block.x,    // 2 blocks in x
              (HEIGHT + block.y - 1) / block.y);  // 2 blocks in y
    whoAmI<<<grid, block>>>(d_block, d_idx);

    cudaMemcpy(h_block, d_block, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_idx, d_idx, bytes, cudaMemcpyDeviceToHost);

    printMatrix("which block wrote each cell", h_block);
    printMatrix("flat index of each cell", h_idx);

    cudaFree(d_block);
    cudaFree(d_idx);
    return 0;
}
