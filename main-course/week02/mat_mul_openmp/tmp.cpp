
  // Block partion
  for(int tid = 0; tid < num_threads; tid++) {
    i_start = M / num_threads * tid + std::min(tid, M % num_threads);
    i_end = M / num_threads * (tid + 1) + std::min(tid, M % num_threads);
    #pragma omp parallel num_threads(num_threads) private(i, j, k, ii, jj, kk)
    {
      #pragma omp for nowait schedule(static) collapse(3)
      for (ii = i_start; ii < i_end; ii += ITILESIZE) {
        for (jj = 0; jj < N; jj += JTILESIZE) {
          for (kk = 0; kk < K; kk += KTILESIZE) {
            for (k = kk; k < std::min(K, kk + KTILESIZE); k++) {
              for (i = ii; i < std::min(i_end, ii + ITILESIZE); i++) {
                float ar = A[i * K + k];
                for (j = jj; j < std::min(N, jj + JTILESIZE); j++) {
                  C[i * N + j] += ar * B[k * N + j];
                }
              }
            }
          }
        }
      }
    }
  }


  // Cyclic partition
  #pragma omp parallel private(i, j, k, ii, jj, kk)
  {
    #pragma omp for nowait schedule(static)
    for (ii = 0; ii < M; ii += ITILESIZE) {
      int i_end = std::min(ii + ITILESIZE, M);
      for (jj = 0; jj < N; jj += JTILESIZE) {
        for (kk = 0; kk < K; kk += KTILESIZE) {
          for (k = kk; k < std::min(K, kk + KTILESIZE); k++) {
            for (i = ii; i < i_end; i++) {
              float ar = A[i * K + k];
              for (j = jj; j < std::min(N, jj + JTILESIZE); j++) {
                C[i * N + j] += ar * B[k * N + j];
              }
            }
          }
        }
      }
    }
  }
