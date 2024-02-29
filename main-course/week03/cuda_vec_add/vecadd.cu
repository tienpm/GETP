#include <cstdio>
#include <cstdlib>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>

#include "util.h"
#include "vecadd.h"

#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t status_ = call;                                                \
    if (status_ != cudaSuccess) {                                              \
      fprintf(stderr, "CUDA error (%s:%d): %s:%s\n", __FILE__, __LINE__,       \
              cudaGetErrorName(status_), cudaGetErrorString(status_));         \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

__global__ void vecadd_kernel(const int N, const float *a, const float *b,
                              float *c) {
  int tidx = blockIdx.x * blockDim.x + threadIdx.x;
  c[tidx] = a[tidx] + b[tidx];
}

int ceil(int a, int b) {
  if (a == 0)
    return 0;
  return (a - 1) / b + 1;
}

// Device(GPU) pointers
static float *A_gpu, *B_gpu, *C_gpu;

void vecadd(float *_A, float *_B, float *_C, int N) {
  // (TODO) Upload A and B vector to GPU
  double Agpu_h2d_start = get_time();
  cudaMemcpy(A_gpu, _A, N * sizeof(float), cudaMemcpyHostToDevice);
  double Agpu_h2d_duration = get_time() - Agpu_h2d_start;
  cudaMemcpy(B_gpu, _B, N * sizeof(float), cudaMemcpyHostToDevice);

  // Launch kernel on a GPU
  dim3 gridDim((N + 511) / 512);
  dim3 blockDim(512);
  vecadd_kernel<<<gridDim, blockDim>>>(N, A_gpu, B_gpu, C_gpu);
  CHECK_CUDA(cudaGetLastError());

  // (TODO) Download C vector from GPU
  cudaMemcpy(_C, C_gpu, N * sizeof(float), cudaMemcpyDeviceToHost);

  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
}

void vecadd_init(int N) {
  // (TODO) Allocate device memory
  cudaMalloc(&A_gpu, N * sizeof(float));
  cudaMalloc(&B_gpu, N * sizeof(float));
  cudaMalloc(&C_gpu, N * sizeof(float));

  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
}

void vecadd_cleanup(float *_A, float *_B, float *_C, int N) {
  // (TODO) Do any post-vecadd cleanup work here.
  cudaFree(A_gpu);
  cudaFree(B_gpu);
  cudaFree(C_gpu);

  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
}
