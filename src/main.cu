#include "stdio.h"

__global__ void cuda_hello(){
  printf("Hello World from GPU!\n");
}

int main() {
  cuda_hello<<<1,1>>>();
  printf("Hello World from CPU!\n");
  return 0;
}
