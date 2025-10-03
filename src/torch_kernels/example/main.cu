// NOTE: This is a copy of cuda_extension_kernel.cu. It's kept here to test
// collision handling when a C++ file and CUDA file share the same filename.
// Setuptools can't deal with this at all, so the setup.py-based test uses
// cuda_extension_kernel.cu and the JIT test uses this file. Symlinks don't
// work well on Windows, so this is the most thorough solution right now.

#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>

#include <ATen/ATen.h>

__global__ void sigmoid_add_kernel(
    const float* __restrict__ x,
    const float* __restrict__ y,
    float* __restrict__ output,
    const int size) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    const float sigmoid_x = 1.0f / (1.0f + __expf(-x[index]));
    const float sigmoid_y = 1.0f / (1.0f + __expf(-y[index]));
    output[index] = sigmoid_x + sigmoid_y;
  }
}

void sigmoid_add_cuda(const float* x, const float* y, float* output, int size) {
  const int threads = 1024;
  const int blocks = (size + threads - 1) / threads;
  sigmoid_add_kernel<<<blocks, threads>>>(x, y, output, size);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

__global__ void sum_kernel(
    const float* __restrict__ x,
    float* __restrict__ output,
    const int size) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    atomicAdd(output, x[index]);
  }
}

void sum_cuda(const float* x, float* output, int size) {
  const int threads = 1024;
  const int blocks = (size + threads - 1) / threads;
  sum_kernel<<<blocks, threads>>>(x, output, size);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

__global__ void sum_fast_kernel(
    const float* __restrict__ x,
    float* __restrict__ output,
    const int size) {
  const int tid = threadIdx.x;
  const int index = blockIdx.x * blockDim.x + tid;
  __shared__ float s_x[1024];

  if (tid >= 1024) {
    return;
  }

  s_x[tid] = x[index];
  __syncthreads();

  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      s_x[tid] += s_x[tid + stride];
    } 

    __syncthreads();
  }

  if (tid == 0) atomicAdd(output, s_x[0]);
}

void sum_fast_cuda(const float* x, float* output, int size) {
  const int threads = 1024;
  const int blocks = (size + threads - 1) / threads;
  sum_kernel<<<blocks, threads>>>(x, output, size);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}