#include "stdio.h"
#include "cuda.h"

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

  vector_add_kernel<<<1,3>>>(d_a, d_b, d_c, 3);

  cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
}

int main() {
  float a[3] = {1, 1, 1};
  float b[3] = {1, 2, 1};
  float* c = new float[3];

  vector_add(a, b, c, 3);
  printf("%f %f %f", c[0], c[1], c[2]);
}
