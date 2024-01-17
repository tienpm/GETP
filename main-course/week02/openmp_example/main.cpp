#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifdef _OPENMP
#include <omp.h>
#endif

void hello(void);
void nowait_example(int n, int m, float *a, float* b, float* y, float *z);

int main(int argc, char* argv[]) {
  // hello();

  int n, m;
  scanf("%d", &n);
  scanf("%d", &m);
  float a[n]; 
  float b[n]; 
  float y[m]; 
  float z[m];
  for (int i = 0; i < n; i++) {
    a[i] = rand() % 10;
    b[i] = rand() % 10;
  }

  for (int i = 0; i < m; i++) {
    y[i] = rand() % 1000;
    z[i] = i * i;
  }
  
  printf("=== BEGIN ===\n");
  printf("A: ");
  for (int i = 0; i < n; i++) printf("%.2f ", a[i]);
  printf("\n");
  printf("B: ");
  for (int i = 0; i < n; i++) printf("%.2f ", b[i]);
  printf("\n");
  printf("Y: ");
  for (int i = 0; i < n; i++) printf("%.2f ", y[i]);
  printf("\n");
  printf("Z: ");
  for (int i = 0; i < n; i++) printf("%.2f ", z[i]);
  printf("\n");

  // int n_threads = strtol(argv[1], nullptr, 10);
  // #pragma omp parallel num_threads(n_threads)
  nowait_example(n, m, a, b, y, z);
  printf("=== AFTER ===\n");
  printf("A: ");
  for (int i = 0; i < n; i++) printf("%.2f ", a[i]);
  printf("\n");
  printf("B: ");
  for (int i = 0; i < n; i++) printf("%.2f ", b[i]);
  printf("\n");
  printf("Y: ");
  for (int i = 0; i < n; i++) printf("%.2f ", y[i]);
  printf("\n");
  printf("Z: ");
  for (int i = 0; i < n; i++) printf("%.2f ", z[i]);
  printf("\n");

  return 0;
}

void hello(void) {
  int my_id = omp_get_thread_num();
  int num_threads = omp_get_num_threads();

  printf("Master thread ID: %d - Num threads: %d\n", my_id, num_threads);
}

void nowait_example(int n, int m, float *a, float* b, float* y, float *z) {
  int i;

  #pragma omp parallel
  {
    #pragma omp for nowait
    for (i = 1; i < n; i++) {
      b[i] = (a[i] + a[i-1]) / 2.0;
      int thread_id = omp_get_thread_num();

      printf("Thread ID: %d\n", thread_id);
    }

    #pragma omp for nowait
    for (i = 0; i < n; i++)
      y[i] = sqrt(z[i]);
  }
}
