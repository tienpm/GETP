#include "mat_mul.h"
#include "util.h"

#include <pthread.h>
#include <immintrin.h>
#include <algorithm>
#include <iostream>
#include <vector>
#include <sys/time.h>

#define ITILESIZE (32)
#define JTILESIZE (1024)
#define KTILESIZE (1024)

struct ThreadArg {
  int M, N, K;
  int row_start, row_end;
  float *A, *B, *C;
  pthread_mutex_t mutex;
};

struct myThreadArg {
  int ii, jj, kk;
  int M, N, K;
  float* A;
  float* B;
  float* C;
};

void* mat_mul_naive(void* args);
void* mat_mul_tiling_kij(void* args);
void* mat_mul_tiling_kij_unrolling(void* args);
void* mat_mul_my_tiling(void* args);

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
  // Naive
  ThreadArg thread_arg[_num_threads];
  pthread_t threads[_num_threads];
  pthread_attr_t thread_attrs[_num_threads];
  cpu_set_t thread_on_cpus[_num_threads]; 
  for (int tid = 0; tid < _num_threads; tid++) {
    // printf("\n");
    pthread_attr_init(&thread_attrs[tid]);
    CPU_ZERO(&thread_on_cpus[tid]);
    CPU_SET(tid, &thread_on_cpus[tid]);
    // Init thread with specific CPU if CPU is busy, it will move to another CPU -> avoid switch thread context
    pthread_attr_setaffinity_np(&thread_attrs[tid], sizeof(cpu_set_t), &thread_on_cpus[tid]);
    // Assign thread arguments
    // TODO: Fix bug -> thread_arg = (struct naiveThreadArg*)malloc(sizeof(struct naiveThreadArg)); // Initialize thread arguments
    thread_arg[tid].M = M, thread_arg[tid].N = N, thread_arg[tid].K = K;
    thread_arg[tid].A = A, thread_arg[tid].B = B, thread_arg[tid].C = C;
    thread_arg[tid].row_start = M / _num_threads * tid + std::min(tid, M % _num_threads);
    thread_arg[tid].row_end = M / _num_threads * (tid + 1) + std::min(tid+1, M % _num_threads);
    // pthread_create(&threads[tid], &thread_attrs[tid], mat_mul_naive, (void*)&thread_arg[tid]);  // TODO: Catch Exeption
    // pthread_create(&threads[tid], &thread_attrs[tid], mat_mul_tiling_kij, (void*)&thread_arg[tid]);  // TODO: Catch Exeption
    pthread_create(&threads[tid], &thread_attrs[tid], mat_mul_tiling_kij_unrolling, (void*)&thread_arg[tid]);  // TODO: Catch Exeption
    // free(thread_arg);
  }

  for (int tid = 0; tid < _num_threads; tid++) {
    pthread_join(threads[tid], nullptr);
  }
  
  // Tiling + Loop Unrolling

  // My Tiling order
  // ThreadArg* thread_arg;
  /* ================================================ */
  // for (int ii = 0; ii < M; ii += ITILESIZE) {
  //   for (int jj = 0; jj < N; jj += JTILESIZE) {
  //     for (int kk = 0; kk < K; kk += KTILESIZE) {
  //       pthread_t threads[_num_threads];
  //       pthread_attr_t thread_attrs[_num_threads];
  //       cpu_set_t thread_on_cpus[_num_threads];             
  //       for (int tid = 0; tid < _num_threads; tid++) {
  //         pthread_attr_init(&thread_attrs[tid]);
  //         CPU_ZERO(&thread_on_cpus[tid]);
  //         CPU_SET(tid, &thread_on_cpus[tid]);
  //         // Init thread with specific CPU if CPU is busy, it will move to another CPU -> avoid switch thread context
  //         pthread_attr_setaffinity_np(&thread_attrs[tid], sizeof(cpu_set_t), &thread_on_cpus[tid]);
  //         // Assign thread arguments
  //         thread_arg = (struct ThreadArg*)malloc(sizeof(struct ThreadArg)); // Initialize thread arguments
  //         thread_arg->M = M, thread_arg->N = N, thread_arg->K = K;
  //         thread_arg->ii = ii, thread_arg->jj = jj, thread_arg->kk = kk;
  //         thread_arg->A = A, thread_arg->B = B, thread_arg->C = C;
  //         pthread_create(&threads[tid], &thread_attrs[tid], mat_mul_my_tiling, (void*)&thread_arg);
  //       }
  //
  //       for (int tid = 0; tid < _num_threads; tid++) {
  //         pthread_join(threads[tid], nullptr);
  //       }
  //     }
  //   }
  // }
}

void* mat_mul_naive(void* args) {
  ThreadArg* arg = (ThreadArg*)args;
  int N = arg->N, K = arg->K;
  float *A = arg->A, *B = arg->B, *C = arg->C;
  int i, j, k;
  float sum;
  int i_start = arg->row_start, i_end = arg->row_end;
  for (i = i_start; i < i_end; i++) {
    for (j = 0; j < N; j++) {
      sum  = 0;
      for (k = 0; k < K; k++) {
        sum += A[i * K + k] * B[j + K * k];
      }
      C[i * N + j] = sum;
    }
  }

  return nullptr;
}

void* mat_mul_tiling_kij(void* args) {
  // Tiling KIJ order
  ThreadArg* arg = (ThreadArg*)args;
  int N = arg->N, K = arg->K;
  float *A = arg->A, *B = arg->B, *C = arg->C;
  int i_start = arg->row_start, i_end = arg->row_end;
  for (int ii = i_start; ii < i_end; ii += ITILESIZE) {
    for (int jj = 0; jj < N; jj += JTILESIZE) {
      for (int kk = 0; kk < K; kk += KTILESIZE) {
        for (int k = kk; k < std::min(K, kk + KTILESIZE); k++) {
          for (int i = ii; i < std::min(i_end, ii + ITILESIZE); i++) {
            float ar = A[i * K + k];
            for (int j = jj; j < std::min(N, jj + JTILESIZE); j++) {
              C[i * N + j] += ar * B[k * N + j];
            }
          }
        }
      }
    }
  }

  return nullptr;
}

void* mat_mul_tiling_kij_unrolling(void* args) {
  // Tiling KIJ order + Loop Unrolling
  ThreadArg* arg = (ThreadArg*)args;
  int N = arg->N, K = arg->K;
  float *A = arg->A, *B = arg->B, *C = arg->C;
  int i_start = arg->row_start, i_end = arg->row_end;
  // double start_time = get_time();
  for (int kk = 0; kk < K; kk += KTILESIZE) {
    for (int ii = i_start; ii < i_end; ii += ITILESIZE) {
      for (int jj = 0; jj < N; jj += JTILESIZE) {
        for (int k = kk; k < std::min(K, kk + KTILESIZE); k+=4) {
          for (int i = ii; i < std::min(i_end, ii + ITILESIZE); i++) {
            float a0 = A[i * K + (k + 0)];
            float a1 = A[i * K + (k + 1)];
            float a2 = A[i * K + (k + 2)];
            float a3 = A[i * K + (k + 3)];
            for (int j = jj; j < std::min(jj + JTILESIZE, N); j += 1) {
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
  // printf("duration: %lf\n", duration);

  return nullptr;
}

void* mat_mul_my_tiling(void* args) {
  myThreadArg* arg = (myThreadArg*)args;
  int kk = arg->kk, ii = arg->ii, jj = arg->jj;
  int M = arg->M, N = arg->N, K = arg->K;
  float *A = arg->A, *B = arg->B, *C = arg->C;
  // Multiply current tile (K-I-J order)
  for (int k = kk; k < std::min(K, kk + KTILESIZE); k++) {
    for (int i = ii; i < std::min(M, ii + ITILESIZE); i++) {
      float ar = A[i * K + k];
      for (int j = jj; j < std::min(N, jj + JTILESIZE); j++) {
        C[i * N + j] += ar * B[k * N + j];
      }
    }
  }
  
  return nullptr;
}
