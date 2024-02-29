#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>

#define CHECK_OPENCL(err)                                         \
  if (err != CL_SUCCESS) {                                        \
    printf("[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err); \
    exit(EXIT_FAILURE);                                           \
  }

int main() {
  cl_int err;

  /*
    You will use the following APIs:
    clGetPlatformIDs
    clGetPlatformInfo
    clGetDeviceIDs
    clGetDeviceInfo

    [Example output]
    Number of platforms: 1
    platform: 0
    - CL_PLATFORM_NAME: NVIDIA CUDA
    - CL_PLATFORM_VENDOR: NVIDIA Corporation
    Number of devices: 1
    device: 0
    - CL_DEVICE_TYPE: 4
    - CL_DEVICE_NAME: NVIDIA GeForce RTX 3090
    - CL_DEVICE_MAX_WORK_GROUP_SIZE: 1024
    - CL_DEVICE_GLOBAL_MEM_SIZE: 25446907904
    - CL_DEVICE_LOCAL_MEM_SIZE: 49152
    - CL_DEVICE_MAX_MEM_ALLOC_SIZE: 6361726976
    ...
  */
  
  // clGetPlatformIDs
  cl_uint num_platforms;
  err = clGetPlatformIDs(0, nullptr, &num_platforms); CHECK_OPENCL(err);
  printf("Number of platforms: %d\n", num_platforms);

  // clGetPlatformInfo
  cl_platform_id *platforms;
  platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * num_platforms);
  err = clGetPlatformIDs(num_platforms, platforms, NULL); CHECK_OPENCL(err);
  for (int plt_id = 0; plt_id < num_platforms; ++plt_id) {
    size_t name_size;
    char* platform_name;
    size_t vendor_size;
    char* platform_vendor;

    err = clGetPlatformInfo(platforms[plt_id], CL_PLATFORM_NAME, 0, NULL, &name_size); CHECK_OPENCL(err); 
    platform_name = (char*)malloc(name_size); 
    err = clGetPlatformInfo(platforms[plt_id], CL_PLATFORM_NAME, name_size, platform_name, NULL); CHECK_OPENCL(err);
    err = clGetPlatformInfo(platforms[plt_id], CL_PLATFORM_VENDOR, 0, NULL, &vendor_size); CHECK_OPENCL(err); 
    platform_vendor = (char*)malloc(vendor_size); 
    err = clGetPlatformInfo(platforms[plt_id], CL_PLATFORM_VENDOR, vendor_size, platform_vendor, NULL); CHECK_OPENCL(err);
    printf("platforms: %d\n", plt_id);
    printf("- CL_PLARTFORM_NAME: %s\n", platform_name);
    printf("- CL_PLARTFORM_VENDOR: %s\n", platform_vendor);

    cl_uint num_devices;
    cl_device_id *devices;
    err = clGetDeviceIDs(platforms[plt_id], CL_DEVICE_TYPE_ALL, 0, nullptr, &num_devices); CHECK_OPENCL(err);
    devices = (cl_device_id*)malloc(sizeof(cl_device_id) * num_devices);
    err = clGetDeviceIDs(platforms[plt_id], CL_DEVICE_TYPE_ALL, num_devices, devices, NULL); CHECK_OPENCL(err);
    printf("Number of devices: %d\n", num_devices);
    for (cl_uint device_id = 0; device_id < num_devices; device_id++) {
      printf("device %d\n", device_id);
      size_t device_type;
      size_t device_name_size;
      char* device_name;
      size_t device_max_work_group_size;
      cl_ulong device_global_mem_size;
      cl_ulong device_local_mem_size;
      cl_ulong device_max_mem_alloc_size;

      err = clGetDeviceInfo(devices[device_id], CL_DEVICE_TYPE, sizeof(size_t), &device_type, NULL); CHECK_OPENCL(err);
      printf("- CL_DEVICE_TYPE: %ld\n", device_type);

      err = clGetDeviceInfo(devices[device_id], CL_DEVICE_NAME, 0, NULL, &device_name_size); CHECK_OPENCL(err);
      device_name = (char*)malloc(sizeof(char*) * device_name_size);
      err = clGetDeviceInfo(devices[device_id], CL_DEVICE_NAME, device_name_size, device_name, NULL);
      printf("- CL_DEVICE_NAME: %s\n", device_name);

      err = clGetDeviceInfo(devices[device_id], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(cl_ulong), &device_max_work_group_size, NULL); CHECK_OPENCL(err);
      printf("- CL_DEVICE_MAX_WORK_GROUP_SIZE: %ld\n", device_max_work_group_size);

      err = clGetDeviceInfo(devices[device_id], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &device_global_mem_size, NULL); CHECK_OPENCL(err);
      printf("- CL_DEVICE_GLOBAL_MEM_SIZE: %ld\n", device_global_mem_size);

      err = clGetDeviceInfo(devices[device_id], CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &device_local_mem_size, NULL); CHECK_OPENCL(err);
      printf("- CL_DEVICE_LOCAL_MEM_SIZE: %ld\n", device_local_mem_size);

      err = clGetDeviceInfo(devices[device_id], CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong), &device_max_mem_alloc_size, NULL); CHECK_OPENCL(err);
      printf("- CL_DEVICE_MAX_MEM_ALLOC_SIZE: %ld\n", device_max_mem_alloc_size);
    }
  }

  return 0;
}
