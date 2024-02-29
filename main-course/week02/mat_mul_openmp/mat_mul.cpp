#include "mat_mul.h"

#include <pthread.h>
#include <immintrin.h>
#include <algorithm>
#include <iostream>
#include <vector>
#include <omp.h>
#include <sched.h>
#include <sys/time.h>


#define ITILESIZE (32)
#define JTILESIZE (1024)
#define KTILESIZE (1024)


void mat_mul_naive(float *A, float *B, float *C,
             int M, int N, int K, int num_threads);
void mat_mul_tiling_kij(float *A, float *B, float *C,
             int M, int N, int K, int num_threads);
void mat_mul_tiling_kij_unrolling(float *A, float *B, float *C,
             int M, int N, int K, int num_threads);

// static double get_time() {
//     struct timeval tv;
//     gettimeofday(&tv, 0);
//     return tv.tv_sec + tv.tv_usec * 1e-6;
// }

void mat_mul(float *_A, float *_B, float *_C,
             int _M, int _N, int _K, int _num_threads) {
  float *A = _A, *B = _B, *C = _C;
  int M = _M, N = _N, K = _K;

  // IMPLEMENT HERE
  // Naive implementation
  // mat_mul_naive(A, B, C, M, N, K, _num_threads);

  // Tiling with K-I-J order
  // mat_mul_tiling_kij(A, B, C, M, N, K, _num_threads);

  // Tiling with K-I-J order and Loop Unrolling
  mat_mul_tiling_kij_unrolling(A, B, C, M, N, K, _num_threads);
}
void mat_mul_naive(float *A, float *B, float *C,
             int M, int N, int K, int _num_threads) {
  omp_set_num_threads(_num_threads);
  #pragma omp parallel shared(M, N, K, A, B, C)
  {
    #pragma omp for nowait // schedule(static, chunk_size) 
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j++) {
        float sum  = 0;
        for (int k = 0; k < K; k++) {
          sum += A[i * K + k] * B[j + K * k];
        }
        C[i * N + j] = sum;
      }
    }
  }
}

void mat_mul_tiling_kij(float *A, float *B, float *C,
             int M, int N, int K, int num_threads) {
  // Tiling KIJ order
  omp_set_num_threads(num_threads);
  // Block partition
  #pragma omp parallel 
  {
    int tid = omp_get_thread_num();
    cpu_set_t cpu_set;
    CPU_ZERO(&cpu_set);
    CPU_SET(tid, &cpu_set);
    sched_setaffinity(0, sizeof(cpu_set_t), &cpu_set);
    #pragma omp for nowait collapse(3) // schedule(static)
    for (int ii = 0; ii < M; ii += ITILESIZE) {
      for (int jj = 0; jj < N; jj += JTILESIZE) {
        for (int kk = 0; kk < K; kk += KTILESIZE) {
          for (int k = kk; k < std::min(kk + KTILESIZE, K); k++) {
            for (int i = ii; i < std::min(ii + ITILESIZE, M); i++) {
              float ar = A[i * K + k];
              for (int j = jj; j < std::min(jj + JTILESIZE, N); j++) {
                C[i * N + j] += ar * B[k * N + j];
              }
            }
          }
        }
      }
    }
  }
}

void mat_mul_tiling_kij_unrolling(float *A, float *B, float *C,
             int M, int N, int K, int num_threads) {
  // Tiling KIJ order + Loop Unrolling
  // #pragma omp parallel for num_threads(num_threads)
  //   for (int ii = 0; ii < M; ii += ITILESIZE) {
  //     // int tid = omp_get_thread_num();
  //     // cpu_set_t cpu_set;
  //     // CPU_ZERO(&cpu_set);
  //     // CPU_SET(tid, &cpu_set);
  //     // sched_setaffinity(0, sizeof(cpu_set_t), &cpu_set);
  //     for (int jj = 0; jj < N; jj += JTILESIZE) {
  //       for (int kk = 0; kk < K; kk += KTILESIZE) {
  //         for (int k = kk; k < std::min(K, kk + KTILESIZE); k+=4) { 
  //           for (int i = ii; i < std::min(M, ii + ITILESIZE); i++) {
  //             float a0 = A[i * K + (k + 0)];
  //             float a1 = A[i * K + (k + 1)];
  //             float a2 = A[i * K + (k + 2)];
  //             float a3 = A[i * K + (k + 3)];
  //             for (int j = jj; j < std::min(N, jj + JTILESIZE); j++) {
  //               float b0 = B[(k + 0) * N + (j + 0)];
  //               float b1 = B[(k + 1) * N + (j + 0)];
  //               float b2 = B[(k + 2) * N + (j + 0)];
  //               float b3 = B[(k + 3) * N + (j + 0)];
  //
  //               float c0 = C[i * N + (j + 0)];
  //
  //               c0 += a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3;
  //
  //               C[i * N + (j + 0)] = c0;
  //             }
  //           }
  //         }
  //       }
  //     }
  //   }

  // for (int kk = 0; kk < K; kk += KTILESIZE) {
  //   #pragma omp parallel for // nowait collapse(3)
  //   for (int ii= 0; ii < M; ii += ITILESIZE) {
  //   int tid = omp_get_thread_num();
  //   cpu_set_t cpu_set;
  //   CPU_ZERO(&cpu_set);
  //   CPU_SET(tid, &cpu_set);
  //   sched_setaffinity(0, sizeof(cpu_set_t), &cpu_set);
  //     for (int jj = 0; jj < N; jj += JTILESIZE) {
  //       for (int k = kk; k < std::min(K, kk + KTILESIZE); k+=4) { 
  //         for (int i = ii; i < std::min(M, ii + ITILESIZE); i++) {
  //           float a0 = A[i * K + (k + 0)];
  //           float a1 = A[i * K + (k + 1)];
  //           float a2 = A[i * K + (k + 2)];
  //           float a3 = A[i * K + (k + 3)];
  //           for (int j = jj; j < std::min(N, jj + JTILESIZE); j++) {
  //             float b0 = B[(k + 0) * N + (j + 0)];
  //             float b1 = B[(k + 1) * N + (j + 0)];
  //             float b2 = B[(k + 2) * N + (j + 0)];
  //             float b3 = B[(k + 3) * N + (j + 0)];
  //
  //             float c0 = C[i * N + (j + 0)];
  //
  //             c0 += a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3;
  //
  //             C[i * N + (j + 0)] = c0;
  //           }
  //         }
  //       }
  //     }
  //   }
  // }
  
  printf("\n");
  #pragma omp parallel num_threads(num_threads)
  {
    int tid = omp_get_thread_num();
    cpu_set_t cpu_set;
    CPU_ZERO(&cpu_set);
    CPU_SET(tid, &cpu_set);
    sched_setaffinity(0, sizeof(cpu_set_t), &cpu_set);
    int i_start = M / num_threads * tid + std::min(tid, M % num_threads);
    int i_end = M / num_threads * (tid + 1) + std::min(tid + 1, M % num_threads);
    // double start_time = get_time();
    for (int kk = 0; kk < K; kk += KTILESIZE) {
      for (int ii= i_start; ii < i_end; ii += ITILESIZE) {
        for (int jj = 0; jj < N; jj += JTILESIZE) {
          int mK = std::min(K, kk + KTILESIZE);
          int mI = std::min(i_end, ii + ITILESIZE);
          int mJ = std::min(N, jj + JTILESIZE);
          for (int k = kk; k < mK; k+=4) { 
            for (int i = ii; i < mI; i++) {
              float a0 = A[i * K + (k + 0)];
              float a1 = A[i * K + (k + 1)];
              float a2 = A[i * K + (k + 2)];
              float a3 = A[i * K + (k + 3)];
              for (int j = jj; j < mJ; j++) {
                float b0 = B[(k + 0) * N + (j + 0)];
                float b1 = B[(k + 1) * N + (j + 0)];
                float b2 = B[(k + 2) * N + (j + 0)];
                float b3 = B[(k + 3) * N + (j + 0)];

                float c0 = C[i * N + (j + 0)];

                c0 += a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3;

                C[i * N + (j + 0)] = c0;
              }
            }
          }
        }
      }
    }
    // double duration = get_time() - start_time;
    // printf("thread num: %d - duration: %f\n", tid, duration);
  }
}
