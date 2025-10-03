#include <torch/extension.h>

// Declare the function from cuda_extension.cu. It will be compiled
// separately with nvcc and linked with the object file of cuda_extension.cpp
// into one shared library.
void sigmoid_add_cuda(const float* x, const float* y, float* output, int size);
void sum_cuda(const float* x, float* output, int size);
void sum_fast_cuda(const float* x, float* output, int size);

torch::Tensor sigmoid_add(torch::Tensor x, torch::Tensor y) {
  TORCH_CHECK(x.device().is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(y.device().is_cuda(), "y must be a CUDA tensor");
  // Throw if tensors are not contiguous - explicit requirement
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  TORCH_CHECK(y.is_contiguous(), "y must be contiguous");
  auto output = torch::zeros_like(x);
  sigmoid_add_cuda(
      x.data_ptr<float>(), y.data_ptr<float>(), output.data_ptr<float>(), output.numel());
  return output;
}

torch::Tensor sum(torch::Tensor x) {
  TORCH_CHECK(x.device().is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  // Create a scalar tensor (1 element) on the same device as input
  auto output = torch::zeros({1}, x.options());
  sum_cuda(x.data_ptr<float>(), output.data_ptr<float>(), x.numel()); 
  return output;
}

torch::Tensor sum_fast(torch::Tensor x) {
  TORCH_CHECK(x.device().is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  auto output = torch::zeros({1}, x.options());
  sum_fast_cuda(x.data_ptr<float>(), output.data_ptr<float>(), x.numel());
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sigmoid_add", &sigmoid_add, "sigmoid(x) + sigmoid(y)");
  m.def("sum", &sum, "sum of all elements");
  m.def("sum_fast", &sum_fast, "sum of all elements (fast)");
}