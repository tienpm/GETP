#include <cstdio>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>

#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t status_ = call;                                                \
    if (status_ != cudaSuccess) {                                              \
      fprintf(stderr, "CUDA error (%s:%d): %s:%s\n", __FILE__, __LINE__,       \
              cudaGetErrorName(status_), cudaGetErrorString(status_));         \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

__global__ void pythagoras(int *pa, int *pb, int *pc, int *presult) {
  int a = *pa;
  int b = *pb;
  int c = *pc;

  if ((a * a + b * b) == c * c)
    *presult = 1;
  else
    *presult = 0;
}

int main(int argc, char *argv[]) {
  if (argc != 4) {
    printf("Usage: %s <num 1> <num 2> <num 3>\n", argv[0]);
    return 0;
  }

  int a = atoi(argv[1]);
  int b = atoi(argv[2]);
  int c = atoi(argv[3]);
  int result = 0;

  // TODO: 1. allocate device memory
  int *pa, *pb, *pc, *presult;
  cudaMalloc(&pa, sizeof(int));
  cudaMalloc(&pb, sizeof(int));
  cudaMalloc(&pc, sizeof(int));
  cudaMalloc(&presult, sizeof(int));

  // TODO: 2. copy data to device
  cudaMemcpy(pa, &a, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(pb, &b, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(pc, &c, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(presult, &result, sizeof(int), cudaMemcpyHostToDevice);

  // TODO: 3. launch kernel
  pythagoras<<<1, 1>>>(pa, pb, pc, presult);

  // TODO: 4. copy result back to host

  cudaMemcpy(&result, presult, sizeof(int), cudaMemcpyDeviceToHost);

  if (result)
    printf("YES\n");
  else
    printf("NO\n");

  return 0;
}
