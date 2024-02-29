#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include <time.h>

#define CHECK_OPENCL(err)                                         \
  if (err != CL_SUCCESS) {                                        \
    printf("[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err); \
    exit(EXIT_FAILURE);                                           \
  }

int main() {
  cl_int err;
  cl_platform_id platform;
  cl_device_id device;
  cl_context context;
  cl_command_queue queue;
  cl_program program;
  cl_kernel kernel;
  cl_int N = 16384;

  // 1. Get OpenCL platform
  // clGetPlatformIDs(..., &platform, ...);
  err = clGetPlatformIDs(1, &platform, NULL); CHECK_OPENCL(err);

  // 2. Get OpenCL device
  // clGetDeviceIDs(..., &device, ...);
  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, NULL); CHECK_OPENCL(err);

  // 3. Create OpenCL context
  // context = clCreateContext(..., &err);
  context = clCreateContext(NULL, 1, &device, NULL, NULL, &err); CHECK_OPENCL(err);

  // 4. Create OpenCL command queue
  // queue = clCreateCommandQueue(..., &err);
  queue = clCreateCommandQueue(context, device, 0, &err); CHECK_OPENCL(err);

  // 5. Compile program from "kernel.cl"
  // program = ...;
  // Obtain the length of the source code
  char file_name[] = "kernel.cl";
  FILE *file = fopen(file_name, "rb");
  if (file == nullptr) {
    printf("Failed to open %s\n", file_name);
    exit(EXIT_FAILURE);
  }
  fseek(file, 0, SEEK_END);
  size_t source_size = ftell(file);
  rewind(file);

  // Read and create a null-terminated string with the source code
  char *source_code = (char*)malloc(source_size + 1);
  if (fread(source_code, sizeof(char), source_size, file) != source_size) {
    printf("Failed to read %s\n", file_name);
    exit(EXIT_FAILURE);
  }
  source_code[source_size] = '\0';
  fclose(file);

  // Create a program object
  program = clCreateProgramWithSource(context, 1, (const char **) &source_code, &source_size, &err);
  CHECK_OPENCL(err);
  free(source_code);
  // BUILDING OpenCL PROGRAM
  err = clBuildProgram(program, 1, &device, "-cl-fast-relaxed-math -w", NULL, NULL);
  if (err == CL_BUILD_PROGRAM_FAILURE) {
    size_t log_size;
    CHECK_OPENCL(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size));
    char *log = (char*) malloc(log_size + 1);
    CHECK_OPENCL(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL));
    log[log_size] = 0;
    printf("Compile error: \n%s\n", log);
    free(log);
  }
  CHECK_OPENCL(err);

  // 6. Extract kernel from compiled program
  // kernel = clCreateKernel(..., &err);
  kernel = clCreateKernel(program, "vec_add", &err); CHECK_OPENCL(err);

  // 7. Create buffers
  // a_d = clCreateBuffer(..., &err);
  // CHECK_OPENCL(err);
  // b_d = clCreateBuffer(..., &err);
  // CHECK_OPENCL(err);
  // c_d = clCreateBuffer(..., &err);
  // CHECK_OPENCL(err);
  cl_mem a_d;
  a_d = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * N, NULL, &err);
  CHECK_OPENCL(err);
  cl_mem b_d;
  b_d = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * N, NULL, &err);
  CHECK_OPENCL(err);
  cl_mem c_d; 
  c_d = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * N, NULL, &err);
  CHECK_OPENCL(err);

  // 8. Write to device
  // clEnqueueWriteBuffer(..., a_d, ..., A, ...);
  // clEnqueueWriteBuffer(..., b_d, ..., B, ...);
  /* Intitalize random seed */
  srand(time(NULL));
  float *A, *B;
  // Create float array A, B, and C of length 16384 on the host
  A = (float*)malloc(sizeof(float) * N);
  B = (float*)malloc(sizeof(float) * N);
  for (int i = 0; i < N; i++) {
    A[i] = rand() % 100 + 1;
    B[i] = rand() % 100 + 1;
  }

  float *C;
  C = (float*)malloc(sizeof(float) * N);
  err = clEnqueueWriteBuffer(queue, a_d, CL_FALSE, 0, sizeof(float) * N, A, 0, NULL, NULL);
  CHECK_OPENCL(err);
  err = clEnqueueWriteBuffer(queue, b_d, CL_FALSE, 0, sizeof(float) * N, B, 0, NULL, NULL);
  CHECK_OPENCL(err);
  err = clEnqueueWriteBuffer(queue, c_d, CL_FALSE, 0, sizeof(float) * N, C, 0, NULL, NULL);
  CHECK_OPENCL(err);

  // 9. Setup kernel arguments
  // clSetKernelArg(kernel, 0, ...);
  // clSetKernelArg(kernel, 1, ...);
  // clSetKernelArg(kernel, 2, ...);
  // clSetKernelArg(kernel, 3, ...);
  err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_d); CHECK_OPENCL(err);
  err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_d); CHECK_OPENCL(err);
  err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &c_d); CHECK_OPENCL(err);
  err = clSetKernelArg(kernel, 3, sizeof(cl_int), &N); CHECK_OPENCL(err);


  // 10. Setup global work size and local work size
  // By OpenCL spec, global work size should be MULTIPLE of local work size
  // size_t gws[1] = {...}, lws[1] = {...};
  int bsize = 32;
  size_t gws[1] = {N}, lws[1] = {bsize};

  // 11. Run kernel
  // clEnqueueNDRangeKernel(..., gws, lws, ...);
  clEnqueueNDRangeKernel(queue, kernel, 1, NULL, gws, lws, 0, NULL, NULL);

  // 12. Read from device
  // clEnqueueReadBuffer(..., c_d, ..., C, ...);
  err = clEnqueueReadBuffer(queue, c_d, CL_TRUE, 0, sizeof(float) * N, C, 0, NULL, NULL);
  CHECK_OPENCL(err);
  for (int i = 0; i < N; i++)
    printf("%.2f ", C[i]);
  printf("\n");

  // 13. Free resources
  CHECK_OPENCL(clReleaseMemObject(a_d));
  CHECK_OPENCL(clReleaseMemObject(b_d));
  CHECK_OPENCL(clReleaseMemObject(c_d));
  CHECK_OPENCL(clReleaseKernel(kernel));
  CHECK_OPENCL(clReleaseProgram(program));
  CHECK_OPENCL(clReleaseCommandQueue(queue));
  CHECK_OPENCL(clReleaseContext(context));

  return 0;
}
