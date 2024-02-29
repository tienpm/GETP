#include <cstdio>

__global__ void kernel_add(const int *a, const int *b, int *c) { *c = *a + *b; }

int main() {
  int a = 1, b = 2, c;
  int *d_a, *d_b, *d_c;

  // 1. Allocate device memory
  cudaMalloc(&d_a, sizeof(int));
  cudaMalloc(&d_b, sizeof(int));
  cudaMalloc(&d_c, sizeof(int));

  // 2. Transfer input data to device memory
  cudaMemcpy(d_a, &a, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, &b, sizeof(int), cudaMemcpyHostToDevice);

  // 3. Execute kernel
  kernel_add<<<1, 1>>>(d_a, d_b, d_c);

  // 4. Transfer output data to host memory
  cudaMemcpy(&c, d_c, sizeof(int), cudaMemcpyDeviceToHost);
  printf("c: %d\n", c);
}
