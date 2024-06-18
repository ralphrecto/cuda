// Parallelized Conway's Game of Life
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <cuda.h>

#define GRID_WIDTH   640
#define GRID_HEIGHT  480
#define SEED         1337

// grid helpers

// representation: 1 cell == 1 unsigned char
// TODO: pack this more densely (1 bit/cell instead of 1 byte/cell)
typedef unsigned char* gol_grid;

static int grid_index(unsigned int x, unsigned int y) {
    return y * GRID_WIDTH + x;
}

// set grid to 1/0 at x/y position; handle shadow rows/cols
void set(gol_grid grid, unsigned int x, unsigned int y, bool live) {
    int i = grid_index(x, y);
    grid[i] = (unsigned char) (live ? 1 : 0);
}

bool is_live(gol_grid grid, unsigned int x, unsigned int y) {
    int i = grid_index(x, y);
    return grid[i] == 0;
}

size_t get_grid_size(unsigned int width, unsigned int height) {
    // pad w/ shadow/wraparound cells
    unsigned int num_cells = (width + 2) * (height + 2);
    return (size_t) ceil(num_cells / sizeof(unsigned char));
}

// allocate + randomly initialize the grid
gol_grid create_grid() {
    size_t grid_size = get_grid_size(GRID_WIDTH, GRID_HEIGHT);
    gol_grid grid = (gol_grid) malloc(grid_size);

    // initialize non-shadow cells
    for (int x = 1; x < GRID_WIDTH + 1; x++) {
        for (int y = 1; y < GRID_HEIGHT + 1; y++) {
            int i = grid_index(x, y);
            grid[i] = (unsigned char) (rand() % sizeof(unsigned char));
        }
    }

    // initialize shadow rows
    for (int x = 1; x < GRID_WIDTH + 1; x++) {
        int i_first = grid_index(x, 0);
        int i_first_shadow = grid_index(x, GRID_HEIGHT - 1);

        int i_last = grid_index(x, GRID_HEIGHT);
        int i_last_shadow = grid_index(x, 1);

        grid[i_first] = grid[i_first_shadow];
        grid[i_last] = grid[i_last_shadow];
    }

    // initialize shadow columns
    for (int y = 1; y < GRID_HEIGHT + 1; y++) {
        int i_first = grid_index(0, y);
        int i_first_shadow = grid_index(GRID_WIDTH - 1, y);

        int i_last = grid_index(GRID_WIDTH, y);
        int i_last_shadow = grid_index(1, y);

        grid[i_first] = grid[i_first_shadow];
        grid[i_last] = grid[i_last_shadow];
    }

    return grid;
}

// kernel code

__global__ void gol_kernel() {
    extern __shared__ gol_grid block_grid;

    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= GRID_WIDTH || y >= GRID_HEIGHT) return;
}

int main() {
    srand(SEED);

    // initialize state
    size_t grid_size = get_grid_size(GRID_WIDTH, GRID_HEIGHT);
    gol_grid h_grid = create_grid();

    gol_grid d_grid;
    cudaMalloc((void**) &d_grid, grid_size);
    cudaMemcpy(d_grid, h_grid, grid_size, cudaMemcpyHostToDevice);

    free(h_grid);
    cudaFree(d_grid);
}