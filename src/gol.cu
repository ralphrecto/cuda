// Parallelized Conway's Game of Life
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <cuda.h>
#include <iostream>

#define GRID_WIDTH   64
#define GRID_HEIGHT  64
#define SEED         1337
#define BLOCK_WIDTH  32
#define BLOCK_HEIGHT 32

// grid helpers

// - representation: 1 cell == 1 unsigned char
// - TODO: pack this more densely (1 bit/cell instead of 1 byte/cell)
// - first/last row/col are shadows -- copies of wraparound. so actual cells
//   start at (1, 1) for example.
typedef unsigned char* gol_grid;

__host__ __device__ static int block_index(int x, int y) {
    return y * BLOCK_WIDTH + x;
}

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
    return grid[i] > 0;
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

__global__ void gol_kernel(
    gol_grid grid,
    gol_grid next_grid
) {
    extern __shared__ unsigned char block_grid[];

    // logical coords (start at (0, 0))
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= GRID_WIDTH || y >= GRID_HEIGHT) return;

    // mapped to grid coords, taking shadow rows/cols into account
    int grid_x = x+1;
    int grid_y = y+1;

    // initialize block grid in shared mem
    for (int dx = -1; dx < 2; dx++) {
        for (int dy = -1; dy < 2; dy++) {
            int xi = grid_x + dx;
            int yi = grid_y + dy;

            if (xi < 0 || xi >= GRID_WIDTH || yi < 0 || yi >= GRID_HEIGHT) continue;

            // TODO remove redundant reads/writes of overlaps
            // TODO double check this pointer math
            int i = grid_index(xi, yi);
            int block_i = block_index(xi % BLOCK_WIDTH, yi % BLOCK_HEIGHT);

            block_grid[block_i] = grid[i];
        }
    }

    __syncthreads();

    // compute state for tick
    int live_neighbors = 0;
    for (int xi = grid_x - 1; xi < grid_x + 2 && xi <= GRID_WIDTH && xi >= 0; xi++) {
        for (int yi = grid_y - 1; yi < grid_y + 2 && yi <= GRID_HEIGHT && yi >= 0; yi++) {
            if (xi == grid_x && yi == grid_y) continue;

            int block_i = block_index(xi % BLOCK_WIDTH, yi % BLOCK_HEIGHT);
            if (block_grid[block_i] > 0) {
                live_neighbors += 1;
            }
        }
    }

    bool currently_alive = block_grid[block_index(grid_x % BLOCK_WIDTH, grid_y % BLOCK_HEIGHT)];
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

    set(next_grid, grid_x, grid_y, next_alive);
}

void checkCudaError(const char* message) {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA Error after " << message << ": " << cudaGetErrorName(error) << "; " << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char* argv[]) {
    int num_iter;
    if (argc != 2) {
        printf("%d [num iterations]\n");
        exit(1);
    } else {
        num_iter = atoi(argv[1]);
    }

    srand(SEED);

    // initialize state
    size_t grid_size = get_grid_size(GRID_WIDTH, GRID_HEIGHT);
    gol_grid h_grid = create_grid();

    gol_grid d_grid;
    gol_grid d_grid_next;

    cudaMalloc((void**) &d_grid, grid_size);
    cudaMalloc((void**) &d_grid_next, grid_size);
    cudaMemcpy(d_grid, h_grid, grid_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_grid_next, h_grid, grid_size, cudaMemcpyHostToDevice);

    dim3 grid_dim(GRID_WIDTH / BLOCK_WIDTH, GRID_HEIGHT / BLOCK_HEIGHT);
    dim3 block_dim(BLOCK_WIDTH, BLOCK_HEIGHT);
    size_t block_size = (BLOCK_WIDTH + 2) * (BLOCK_HEIGHT + 2) * sizeof(unsigned char);

    printf("starting game of life sim...\n");
    printf("\x1b[2J");

    for (int i = 0; i < num_iter; i++) {
        gol_kernel<<<grid_dim, block_dim, block_size>>>(d_grid, d_grid_next);
        cudaDeviceSynchronize();
        checkCudaError("game of life kernel");

        gol_grid d_temp = d_grid;
        d_grid = d_grid_next;
        d_grid_next = d_temp;

        if (i % 100 == 0 && i > 0 || i == 0) {
            printf("num iterations: %d\n", i);

            cudaMemcpy(h_grid, d_grid, grid_size, cudaMemcpyDeviceToHost);
            for (int x = 1; x < GRID_WIDTH + 1; x++) {
                for (int y = 1; y < GRID_HEIGHT + 1; y++) {
                    int i = grid_index(x, y);
                    if (h_grid[i] > 0) {
                        printf("x");
                    } else {
                        printf("%d", h_grid[i], i);
                    }
                }
                printf("\n");
            }
        }
    }

    printf("done!\n");
    printf("total num iterations: %d\n", num_iter);

    free(h_grid);
    cudaFree(d_grid);

    return 0;
}