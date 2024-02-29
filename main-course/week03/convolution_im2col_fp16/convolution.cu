#include <cstdio>
#include <cstdlib>
#include <mma.h>

#include "convolution.cuh"

#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t status_ = call;                                                \
    if (status_ != cudaSuccess) {                                              \
      fprintf(stderr, "CUDA error (%s:%d): %s:%s\n", __FILE__, __LINE__,       \
              cudaGetErrorName(status_), cudaGetErrorString(status_));         \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#define WARP_SIZE 32
#define FRAG_SIZE 16
#define BLOCK_SIZE 4

static half *I_gpu;
static half *Col_gpu;
static half *F_gpu;
static float *O_gpu;

__global__ void gpu_im2col(half *_I, half *workspace, int N, int C, int H,
                           int W, int R, int S, int pad_h, int pad_w,
                           int stride_h, int stride_w, int dilation_h,
                           int dilation_w) {
  half *I = _I;

  // GPU im2col
  const int OH = (H + 2 * pad_h - (((R - 1) * dilation_h) + 1)) / stride_h + 1;
  const int OW = (W + 2 * pad_w - (((S - 1) * dilation_w) + 1)) / stride_w + 1;
  const int FILTER = C * R * S;
  const int BATCH_OFFSET = FILTER * OH * OW;

  int batch = blockIdx.y * blockDim.y + threadIdx.y;
  int out = blockIdx.x * blockDim.x + threadIdx.x;

  if (out < OH * OW) {
    int y = stride_h * (out / OW) - pad_h;
    int x = stride_w * (out % OW) - pad_w;
    int idx = 0;
    for (int c = 0; c < C; c++) {
      for (int h = 0; h < R; h++) {
        for (int w = 0; w < S; w++) {
          int dy = y + h * dilation_h;
          int dx = x + w * dilation_w;
          if ((0 <= dy && dy < H) && (0 <= dx && dx < W)) {
            workspace[batch * BATCH_OFFSET + (idx * OH * OW) + out] =
                I[batch * (C * H * W) + c * (H * W) + dy * W + dx];
          } else {
            workspace[batch * BATCH_OFFSET + (idx * OH * OW) + out] = 0;
          }
          idx++;
        }
      }
    }
  }
}

__global__ void gpu_matmul(half *A, half *B, float *C, int M, int K, int N) {
  __shared__ half sA[WARP_SIZE][WARP_SIZE + 1];
  __shared__ half sB[WARP_SIZE][WARP_SIZE];

  int batch = blockIdx.z * blockDim.z + threadIdx.z;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int sy = threadIdx.y;
  int sx = threadIdx.x;

  half sum = 0;
  for (int t = 0; t < K; t += WARP_SIZE) {
    // A
    if (y < M && (sx + t) < K) {
      sA[sy][sx] = A[y * K + (sx + t)];
    } else {
      sA[sy][sx] = 0;
    }
    // B
    if (x < N && (sy + t) < K) {
      sB[sy][sx] = B[batch * K * N + (sy + t) * N + x];
    } else {
      sB[sy][sx] = 0;
    }
    // sync
    __syncthreads();
    // matmul
    for (int k = 0; k < WARP_SIZE && k + t < K; k++) {
      sum += sA[sy][k] * sB[k][sx];
    }
    __syncthreads();
  }
  if (y < M && x < N) {
    C[batch * M * N + y * N + x] = sum;
  }
}

__global__ void gpu_matmul_kernel(half *A, half *B, float *C, int M, int N,
                                  int K) {

  using namespace nvcuda::wmma;

  int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int global_idy = blockIdx.y * blockDim.y + threadIdx.y;

  int j = global_idx / WARP_SIZE * FRAG_SIZE;
  int i = global_idy * FRAG_SIZE;

  int local_j = threadIdx.x / WARP_SIZE;
  int local_i = threadIdx.y;
  int line = threadIdx.x % WARP_SIZE;

  fragment<matrix_a, FRAG_SIZE, FRAG_SIZE, FRAG_SIZE, half, row_major> a_frag;
  fragment<matrix_b, FRAG_SIZE, FRAG_SIZE, FRAG_SIZE, half, row_major> b_frag;
  fragment<accumulator, FRAG_SIZE, FRAG_SIZE, FRAG_SIZE, float> c_frag;

  fill_fragment(c_frag, 0.0f);

  __shared__ half a[BLOCK_SIZE][FRAG_SIZE][FRAG_SIZE];
  __shared__ half b[BLOCK_SIZE][FRAG_SIZE][FRAG_SIZE];

  int ii = local_j * 4 + line / 8;
  int jj = local_i * 4 + line / 8;
  int kk = line % 8 * 2 + 0;
  for (int k = 0; k < K; k += FRAG_SIZE) {
    a[local_i][ii][kk + 0] = A[(i + ii) * K + (k + kk + 0)];
    a[local_i][ii][kk + 1] = A[(i + ii) * K + (k + kk + 1)];
    b[local_j][jj][kk + 0] = B[(k + jj) * N + (j + kk + 0)];
    b[local_j][jj][kk + 1] = B[(k + jj) * N + (j + kk + 1)];
    __syncthreads();
    load_matrix_sync(a_frag, &a[local_i][0][0], FRAG_SIZE);
    load_matrix_sync(b_frag, &b[local_j][0][0], FRAG_SIZE);
    mma_sync(c_frag, a_frag, b_frag, c_frag);
    __syncthreads();
  }

  store_matrix_sync(&C[i * N + j], c_frag, N, mem_row_major);
}

__global__ void gpu_reshape(float *_src, float *_dst, int N, int K, int OH,
                            int OW) {}

void gpu_convolution_im2col(half *_I, half *_F, float *_O, int N, int C, int H,
                            int W, int K, int R, int S, int pad_h, int pad_w,
                            int stride_h, int stride_w, int dilation_h,
                            int dilation_w) {
  half *I = _I, *F = _F;
  float *O = _O;
  const int OH = 1 + (H + 2 * pad_h - (((R - 1) * dilation_h) + 1)) / stride_h;
  const int OW = 1 + (W + 2 * pad_w - (((S - 1) * dilation_w) + 1)) / stride_w;
  const int FILTER = C * R * S;
  const int OUTPUT = OH * OW;
  // Move I, F to GPU
  CHECK_CUDA(cudaMemcpy(I_gpu, I, N * C * H * W * sizeof(half),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(
      cudaMemcpy(F_gpu, F, K * FILTER * sizeof(half), cudaMemcpyHostToDevice));

  // im2col
  dim3 blockDim_im2col(WARP_SIZE * WARP_SIZE, 1);
  dim3 gridDim_im2col((OUTPUT + blockDim_im2col.x - 1) / blockDim_im2col.x, N);
  gpu_im2col<<<gridDim_im2col, blockDim_im2col>>>(
      I_gpu, Col_gpu, N, C, H, W, R, S, pad_h, pad_w, stride_h, stride_w,
      dilation_h, dilation_w);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  // matmul
  // dim3 blockDim_matmul(WARP_SIZE, WARP_SIZE, 1);
  // dim3 gridDim_matmul((OUTPUT + blockDim_matmul.x - 1) / blockDim_matmul.x,
  //                     (K + blockDim_matmul.y - 1) / blockDim_matmul.y, N);
  // gpu_matmul<<<gridDim_matmul, blockDim_matmul>>>(F_gpu, Col_gpu, O_gpu, K,
  //                                                 FILTER, OUTPUT);
  dim3 blockDim(BLOCK_SIZE * WARP_SIZE, BLOCK_SIZE);
  dim3 gridDim((OUTPUT + blockDim.x - 1) / blockDim.x,
               (K + blockDim.y - 1) / blockDim.y, N);
  gpu_matmul_kernel<<<gridDim, blockDim>>>(F_gpu, Col_gpu, O_gpu, K, FILTER,
                                           OUTPUT);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  // reshape
  CHECK_CUDA(cudaMemcpy(O, O_gpu, N * K * OUTPUT * sizeof(float),
                        cudaMemcpyDeviceToHost));
  // gpu_reshape<<<gridDim, blockDim>>>(BUF2, O, N, K, OH, OW);
  // CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
}

void convolution(half *_I, half *_F, float *_O, half *_BUF1, float *_BUF2,
                 int N, int C, int H, int W, int K, int R, int S, int pad_h,
                 int pad_w, int stride_h, int stride_w, int dilation_h,
                 int dilation_w) {
  // Remove this line after you complete the convolution on GPU
  gpu_convolution_im2col(_I, _F, _O, N, C, H, W, K, R, S, pad_h, pad_w,
                         stride_h, stride_w, dilation_h, dilation_w);
}

void convolution_initialize(int N, int C, int H, int W, int K, int R, int S,
                            int pad_h, int pad_w, int stride_h, int stride_w,
                            int dilation_h, int dilation_w) {
  const int OH = (H + 2 * pad_h - (((R - 1) * dilation_h) + 1)) / stride_h + 1;
  const int OW = (W + 2 * pad_w - (((S - 1) * dilation_w) + 1)) / stride_w + 1;
  const int FILTER = C * R * S;
  const int OUTPUT = OH * OW;
  CHECK_CUDA(cudaMalloc((void **)&I_gpu, N * C * H * W * sizeof(half)));
  CHECK_CUDA(
      cudaMalloc((void **)&Col_gpu,
                 N * FILTER * OUTPUT * sizeof(half))); // N * FILTER * OUTPUT
  CHECK_CUDA(
      cudaMalloc((void **)&F_gpu, K * FILTER * sizeof(half))); // K * FILTER
  CHECK_CUDA(cudaMalloc((void **)&O_gpu,
                        N * K * OUTPUT *
                            sizeof(float))); // (K * FILTER) * (N * FILTER *
                                             // OUTPUT) = (N * K * OUTPUT)

  CHECK_CUDA(cudaDeviceSynchronize());
}

void convolution_cleanup(half *_I, half *_F, float *_O, int N, int C, int H,
                         int W, int K, int R, int S, int pad_h, int pad_w,
                         int stride_h, int stride_w, int dilation_h,
                         int dilation_w) {
  CHECK_CUDA(cudaFree(I_gpu));
  CHECK_CUDA(cudaFree(Col_gpu));
  CHECK_CUDA(cudaFree(F_gpu));
  CHECK_CUDA(cudaFree(O_gpu));
}
