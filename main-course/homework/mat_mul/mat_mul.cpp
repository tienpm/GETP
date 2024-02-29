#include "mat_mul.h"
#include "util.h"

#include <cstdio>
#include <cstdlib>
#include <mpi.h>
#include <omp.h>
#include <pthread.h>
#include <sched.h>

#define ITILESIZE 64
#define JTILESIZE 512
#define KTILESIZE 128

#define CHECK_MPI(call) \
  do { \
    int code = call; \
    if (code != MPI_SUCCESS) { \
      char estr[MPI_MAX_ERROR_STRING]; \
      int elen; \
      MPI_Error_string(code, estr, &elen); \
      fprintf(stderr, "MPI error (%s:%d): %s\n", __FILE__, __LINE__, estr); \
      MPI_Abort(MPI_COMM_WORLD, code); \
    } \
  } while (0)

static float *A, *B, *C;
static int M, N, K;
static int num_threads;
static int mpi_rank, mpi_world_size;

struct ThreadArg {
  int tid;
  int M, N, K;
  int row_start, row_end;
  float *A, *B, *C;
  pthread_mutex_t mutex;
};

void matmul_omp_tiling_kij_unrolling(float *A, float *B, float *C, int M, int N, int K, int num_threads);
void matmul_pthread_tiling_kij_unrolling(float *A, float *B, float *C, int M, int N, int K, int num_threads);
void* best_matmul_pthread(void* args);

void mat_mul(float *_A, float *_B, float *_C, int _M, int _N, int _K,
             int _num_threads, int _mpi_rank, int _mpi_world_size) {
  A = _A, B = _B, C = _C;
  M = _M, N = _N, K = _K;
  num_threads = _num_threads, mpi_rank = _mpi_rank,
  mpi_world_size = _mpi_world_size;

  // TODO: parallelize & optimize matrix multiplication on multi-node
  // You must allocate & initialize A, B, C for non-root processes
  int nrows_per_proc = M / mpi_world_size;
  int main_worker = 0;
  
  // Scatter matrix A to nodes with chunk of data and broadcast matrix B
  MPI_Request bcast_request;
  CHECK_MPI(MPI_Ibcast(B, K * N, MPI_FLOAT, main_worker, MPI_COMM_WORLD, &bcast_request));
  CHECK_MPI(MPI_Scatter(A, nrows_per_proc * K, MPI_FLOAT, A, nrows_per_proc * K, MPI_FLOAT, main_worker, MPI_COMM_WORLD));
  MPI_Wait(&bcast_request, MPI_SUCCESS);

  // Calculate
  matmul_omp_tiling_kij_unrolling(A, B, C, nrows_per_proc, N, K, num_threads);

  // matmul_pthread_tiling_kij_unrolling(A, B, C, nrows_per_proc, N, K, num_threads);
  
  CHECK_MPI(MPI_Gather(C, nrows_per_proc * N, MPI_FLOAT, C, nrows_per_proc * N, MPI_FLOAT, main_worker, MPI_COMM_WORLD));
}

void matmul_omp_tiling_kij_unrolling(float *A, float *B, float *C, int M, int N, int K, int num_threads) {
  #pragma omp parallel num_threads(num_threads) // proc_bind(spread)
  {
    int tid = omp_get_thread_num();
    cpu_set_t cpu_set;
    CPU_ZERO(&cpu_set);
    CPU_SET(tid, &cpu_set);
    sched_setaffinity(0, sizeof(cpu_set_t), &cpu_set);
    int i_start = M / num_threads * tid + std::min(tid, M % num_threads);
    int i_end = M / num_threads * (tid + 1) + std::min(tid + 1, M % num_threads);
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
  }
}

void matmul_pthread_tiling_kij_unrolling(float *A, float *B, float *C, int M, int N, int K, int num_threads) {
  ThreadArg thread_arg[num_threads];
  pthread_t threads[num_threads];
  pthread_attr_t thread_attrs[num_threads];
  cpu_set_t thread_on_cpus[num_threads]; 
  for (int tid = 0; tid < num_threads; tid++) {
    pthread_attr_init(&thread_attrs[tid]);
    CPU_ZERO(&thread_on_cpus[tid]);
    CPU_SET(tid, &thread_on_cpus[tid]);
    // Init thread with specific CPU if CPU is busy, it will move to another CPU -> avoid switch thread context
    pthread_attr_setaffinity_np(&thread_attrs[tid], sizeof(cpu_set_t), &thread_on_cpus[tid]);

    // Assign thread arguments
    thread_arg[tid].tid = tid;
    thread_arg[tid].M = M, thread_arg[tid].N = N, thread_arg[tid].K = K;
    thread_arg[tid].A = A, thread_arg[tid].B = B, thread_arg[tid].C = C;
    thread_arg[tid].row_start = M / num_threads * tid + std::min(tid, M % num_threads);
    thread_arg[tid].row_end = M / num_threads * (tid + 1) + std::min(tid+1, M % num_threads);
    pthread_create(&threads[tid], &thread_attrs[tid], best_matmul_pthread, (void*)&thread_arg[tid]);  // TODO: Catch Exeption
  }

  for (int tid = 0; tid < num_threads; tid++) {
    pthread_join(threads[tid], nullptr);
  }
}

void* best_matmul_pthread(void* args) {
  // Tiling KIJ order + Loop Unrolling
  ThreadArg* arg = (ThreadArg*)args;
  int tid = arg->tid;
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
