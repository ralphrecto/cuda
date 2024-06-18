// Parallelized Conway's Game of Life
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <cuda.h>

#define GRID_WIDTH   640
#define GRID_HEIGHT  480
#define SEED         1337

// grid helpers

// - representation: 1 cell == 1 unsigned char
// - TODO: pack this more densely (1 bit/cell instead of 1 byte/cell)
// - first/last row/col are shadows -- copies of wraparound. so actual cells
//   start at (1, 1) for example.
typedef unsigned char* gol_grid;

__host__ __device__ static int grid_index(int x, int y) {
    return y * GRID_WIDTH + x;
}

// set grid to 1/0 at x/y position; handle shadow rows/cols
__host__ __device__ void set(gol_grid grid, unsigned int x, unsigned int y, bool live) {
    int i = grid_index(x, y);
    grid[i] = (unsigned char) (live ? 1 : 0);
}

// indicates whether cell at (x, y) is alive
__host__ __device__ bool is_live(gol_grid grid, unsigned int x, unsigned int y) {
    int i = grid_index(x, y);
    return grid[i] == 0;
}

// computes byte size of entire grid
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

__global__ void gol_kernel(gol_grid* grid) {
    extern __shared__ gol_grid block_grid;

    // logical coords (start at (0, 0))
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= GRID_WIDTH || y >= GRID_HEIGHT) return;

    // mapped to grid coords, taking shadow rows/cols into account
    int grid_x = x+1;
    int grid_y = y+1;

    // initialize block grid in shared mem
    for (int xi = grid_x - 1; xi < grid_x + 2 && xi <= GRID_WIDTH && xi >= 0; xi++) {
        for (int yi = grid_y - 1; yi < grid_y + 2 && yi <= GRID_HEIGHT && yi >= 0; yi++) {
            // TODO remove redundant reads/writes of overlaps
            int i = grid_index(xi, yi);
            block_grid[i] = (*grid)[i];
        }
    }

    __syncthreads();

    // compute state for tick
    int live_neighbors = 0;
    for (int xi = grid_x - 1; xi < grid_x + 2 && xi <= GRID_WIDTH && xi >= 0; xi++) {
        for (int yi = grid_y - 1; yi < grid_y + 2 && yi <= GRID_HEIGHT && yi >= 0; yi++) {
            if (xi == grid_x && yi == grid_y) continue;

            if (is_live(block_grid, xi, yi)) {
                live_neighbors += 1;
            }
        }
    }

    bool currently_alive = is_live(block_grid, grid_x, grid_y);
    bool next_alive = false;
    if (currently_alive) {
        if (live_neighbors == 2 || live_neighbors == 3) {
            next_alive = true;
        } else {
            next_alive = false;
        }
    } else if (live_neighbors == 3) {
        next_alive = true;
    }

    // TODO WIP
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