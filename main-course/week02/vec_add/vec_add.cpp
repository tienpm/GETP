#include "vec_add.h"

#include <pthread.h>
#include <immintrin.h>
#include <cstdio>
#include <algorithm>

struct ThreadArg {
  int tid;
  int start;
  int end;
  float *sub_A;
  float *sub_B;
  float *sub_C;
};

void* fadd(void *arg) {
  ThreadArg* local_arg = (ThreadArg*)arg;
  for (int i = local_arg->start; i < local_arg->end; i++) {
    local_arg->sub_C[i] = local_arg->sub_A[i] + local_arg->sub_B[i];
  }
  
  return NULL;
}

void vec_add(float *_A, float *_B, float *_C, int _M, int _num_threads) {
  // IMPLEMENT HERE
  pthread_t tid[_num_threads];
  ThreadArg arg[_num_threads];
  int chunk_size = _M / _num_threads;
  for (int i = 0; i < _num_threads; i += 1) {
    arg[i].tid = i;
    arg[i].start = i * chunk_size;
    arg[i].end = (i == _num_threads - 1) ? _M : arg[i].start + chunk_size;
    arg[i].sub_A = _A;
    arg[i].sub_B = _B;
    arg[i].sub_C = _C;
    pthread_create(&tid[i], NULL, fadd, (void *)&arg[i]);
  }

  for (int id = 0; id < _num_threads; id += 1) {
    pthread_join(tid[id], NULL);
  }
  // Vector Intrinsic
}
