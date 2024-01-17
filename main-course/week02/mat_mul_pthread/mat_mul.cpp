#include "mat_mul.h"

#include <pthread.h>
#include <immintrin.h>
#include <algorithm>
#include <iostream>

#define ITILESIZE (32)
#define JTILESIZE (1024)
#define KTILESIZE (1024)

struct ThreadArg {
  int start, end;
  int M, N, K;
  float* sub_A;
  float* sub_B;
  float* sub_C;
}

void* _mat_mul_unrolling(void* args);
void* _mat_mul_tiling(void* args);

void mat_mul(float *_A, float *_B, float *_C,
             int _M, int _N, int _K, int _num_threads) {
  float *A = _A, *B = _B, *C = _C;
  int M = _M, N = _N, K = _K;

  // IMPLEMENT HERE
  // Unrolling
  pthread_t tid[num_threads];
  ThreadArg thread_args[num_threads];
  int chunk_size = M / num_threads;
  for (int i = 0; i < num_threads; i++) {
    thread_args[tid].start = tid * chunk_size;
    thread_args[tid].end = (tid == num_threads - 1) ? m : thread_args[i] + chunk_size;
    thread_args[tid].M = M;
    thread_args[tid].N = N;
    thread_args[tid].K = K;
    pthread_create(&tid[i], nullptr, _mat_mul_unrolling, (void *)&thread_args);
  }

  for (int i = 0; i < num_threads; i++) {
    pthread_join(tid[i], nullptr);
  }
}

void* _mat_mul_unrolling(void* args) { 
  /* Unrolling */
  ThreadArg* local_arg = (ThreadArg*)args;
  int M = local_arg.M, N = local_arg.N, K = local_arg.K;
  for (int ri = local_arg.start; ri < local_arg.end; ri++) {
    for (int j = 0; j < local_arg.N; j++) {
      for (int k = 0; k < local_arg.K; k++) {
        C[ri * N + j] = A[ri * K + k] * B[i + j * K];
      }
    }
  }
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < K; k++) {
        C[i * N + j] += A[i * K + k] * B[j + k * K]; 
      }
    }
  }
}
