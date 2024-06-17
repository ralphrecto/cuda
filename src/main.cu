#include <stdio.h>
#include <cuda.h>
#include <assert.h>
#include <iostream>

#define THREADS_PER_BLOCK 256

__global__ void vector_add_kernel(float* a, float* b, float* c, int n) {
  int i = blockDim.x*blockIdx.x + threadIdx.x;
  if (i < n) {
    c[i] = a[i] + b[i];
  }
}

void vector_add(float* a, float* b, float* c, int n) {
  float* d_a;
  float* d_b;
  float* d_c;

  size_t size = (size_t) n * sizeof(float);

  cudaMalloc((void**) &d_a, size);
  cudaMalloc((void**) &d_b, size);
  cudaMalloc((void**) &d_c, size);

  cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_c, c, size, cudaMemcpyHostToDevice);

  vector_add_kernel<<<ceil(n / (1.0 * THREADS_PER_BLOCK)), THREADS_PER_BLOCK>>>(d_a, d_b, d_c, n);

  cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
}

typedef float (*mapf)(float);

__device__ float incr(float a) {
  return a + 1.0;
}

__device__ mapf d_incr = incr;

__global__ void map_kernel(float* a, float* b, int n, float (*func)(float)) {
  int i = blockDim.x*blockIdx.x + threadIdx.x;
  if (i < n) b[i] = func(a[i]);
}

void map(float* a, float* b, int n, float (*func)(float)) {
  float* d_a;
  float* d_b;
  size_t size = (size_t) n * sizeof(float);

  cudaMalloc((void**) &d_a, size);
  cudaMalloc((void**) &d_b, size);

  cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
map_kernel<<<ceil(n / (1.0 * THREADS_PER_BLOCK)), THREADS_PER_BLOCK>>>(d_a, d_b, n, func);

  cudaMemcpy(b, d_b, size, cudaMemcpyDeviceToHost);

  cudaFree(d_a);
  cudaFree(d_b);
}

// puzzles 1-3 

int map_main() {
  float a[4] = {1, 1, 1, 1};
  float* c = new float[4];

  mapf h_incr;
  cudaMemcpyFromSymbol(&h_incr, d_incr, sizeof(mapf));

  map(a, c, 4, h_incr);
  printf("%f %f %f %f", c[0], c[1], c[2], c[3]);

  return 0;
}

__global__ void matmul(
  float* a,
  dim3 dim_a,
  float* b,
  dim3 dim_b,
  float* c,
  dim3 dim_c
) {
  assert(dim_a.y == dim_b.x);
  assert(dim_a.x == dim_c.x);
  assert(dim_b.y == dim_c.y);

  int col = blockDim.x * blockIdx.x + threadIdx.x;
  int row = blockDim.y * blockIdx.y + threadIdx.y;

  if (col >= dim_b.y || row >= dim_a.x) {
    return;
  }

  float dotprod = 0;
  for (int i = 0; i < dim_a.y; i++) {
    dotprod += (a[(row * dim_a.x) + i] * b[(i * dim_b.x) + col]);
  }

  c[(row * dim_c.x) + col] = dotprod;
}

void checkCudaError(const char* message) {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA Error after " << message << ": " << cudaGetErrorName(error) << "; " << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    // Initialize the 2D array
    float h_a[3][3] = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
        {7.0, 8.0, 9.0}
    };

    float h_b[3][3] = {
        {1.0, 1.0, 1.0},
        {1.0, 1.0, 1.0},
        {1.0, 1.0, 1.0},
    };

    float h_c[3][3] = {
        {0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0}
    };

    float* d_a;
    float* d_b;
    float* d_c;

    cudaMalloc((void**) &d_a, sizeof(h_a));
    cudaMalloc((void**) &d_b, sizeof(h_b));
    cudaMalloc((void**) &d_c, sizeof(h_c));

    cudaMemcpy(d_a, h_a, sizeof(h_a), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sizeof(h_b), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_c, sizeof(h_c), cudaMemcpyHostToDevice);

    dim3 blockSize(3, 3);
    dim3 gridSize(1, 1);

    dim3 dim_a(3, 3);
    dim3 dim_b(3, 3);
    dim3 dim_c(3, 3);

    matmul<<<gridSize, blockSize>>>(
      d_a,
      dim_a,
      d_b,
      dim_b,
      d_c,
      dim_c
    );
    cudaDeviceSynchronize();
    checkCudaError("matmul kernel");

    cudaMemcpy(h_c, d_c, sizeof(h_c), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    for (int x = 0; x < dim_c.x; x++) {
      for (int y = 0; y < dim_c.y; y++) {
        printf("%f ", h_c[x][y]);
      }
      printf("\n");
    }

    printf("%d", sizeof(h_c));
 
    return 0;
}