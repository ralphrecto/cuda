#include "stdio.h"
#include "cuda.h"

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

__global__ void map_kernel(float* a, float* b, float (*func)(float)) {
  int i = blockDim.x*blockIdx.x + threadIdx.x;
  b[i] = func(a[i]);
}

void map(float* a, float* b, int n, float (*func)(float)) {
  float* d_a;
  float* d_b;
  size_t size = (size_t) n * sizeof(float);

  cudaMalloc((void**) &d_a, size);
  cudaMalloc((void**) &d_b, size);

  cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);

  map_kernel<<<ceil(n / (1.0 * THREADS_PER_BLOCK)), THREADS_PER_BLOCK>>>(d_a, d_b, func);

  cudaMemcpy(b, d_b, size, cudaMemcpyDeviceToHost);

  cudaFree(d_a);
  cudaFree(d_b);
}

int main() {
  float a[4] = {1, 1, 1, 1};
  float b[4] = {1, 2, 1, 0};
  float* c = new float[4];

  mapf h_incr;
  cudaMemcpyFromSymbol(&h_incr, d_incr, sizeof(mapf));

  map(a, c, 4, h_incr);
  printf("%f %f %f %f", c[0], c[1], c[2], c[3]);
}
