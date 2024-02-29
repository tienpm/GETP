#include <cstdio>
#include <cuda_device_runtime_api.h>

#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t status_ = call;                                                \
    if (status_ != cudaSuccess) {                                              \
      fprintf(stderr, "CUDA error (%s:%d): %s:%s\n", __FILE__, __LINE__,       \
              cudaGetErrorName(status_), cudaGetErrorString(status_));         \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

int main() {
  int count;
  CHECK_CUDA(cudaGetDeviceCount(&count));

  printf("Number of devices: %d\n", count);
  cudaDeviceProp props[4];
  for (int i = 0; i < count; ++i) {
    printf("\tdevice %d:\n", i);
    // TODO: get and print device properties
    cudaGetDeviceProperties(&props[i], i);
    printf("\tname: %s\n", props[i].name);
    printf("\t\tmultiProcessorCount: %d\n", props[i].multiProcessorCount);
    printf("\t\tmaxThreadsPerBlock: %d\n", props[i].maxThreadsPerBlock);
    printf("\t\tsharedMemPerBlock: %ld\n", props[i].sharedMemPerBlock);
  }

  return 0;
}
