#include <getopt.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#include "convolution.cuh"
#include "util.h"

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

static bool print = false;
static bool validation = false;
static int N = 1;
static int C = 3;
static int H = 3;
static int W = 3;
static int K = 3;
static int R = 3;
static int S = 3;
static int pad_h = 0;
static int pad_w = 0;
static int stride_h = 1;
static int stride_w = 1;
static int dilation_h = 1;
static int dilation_w = 1;

static int num_iterations = 1;
static int mpi_rank, mpi_world_size;

static void print_help(const char *prog_name) {
  printf(
      "Usage: %s [-pvh] [-n num_iterations] N C H W K R S pad_h pad_w "
      "stride_h stride_w dilation_h dilation_w\n",
      prog_name);
  printf("Options:\n");
  printf("     -p : print tensor. (default: off)\n");
  printf("     -v : validate convolution. (default: off)\n");
  printf("     -h : print this page.\n");
  printf("     -n : number of iterations (default: 1)\n");
  printf("      N : batch size (default: 1)\n");
  printf("      C : input channel size (default: 3)\n");
  printf("      H : input height (default: 3)\n");
  printf("      W : input width (default: 3)\n");
  printf("      K : output channel size (default: 3)\n");
  printf("      R : filter height (default: 3)\n");
  printf("      S : filter width (default: 3)\n");
  printf("      pad_h : top and bottom padding (default: 0)\n");
  printf("      pad_w : left and right padding (default: 0)\n");
  printf("      stride_h : vertical stride (default: 1)\n");
  printf("      stride_w : horizontal stride (default: 1)\n");
  printf("      dilation_h : vertical dilation (default: 1)\n");
  printf("      dilation_w : horizontal dilation (default: 1)\n");
}

static void parse_opt(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "pvht:n:m:")) != -1) {
    switch (c) {
      case 'p': print = true; break;
      case 'v': validation = true; break;
      case 'n': num_iterations = atoi(optarg); break;
      case 'h':
      default: print_help(argv[0]); exit(0);
    }
  }
  for (int i = optind, j = 0; i < argc; ++i, ++j) {
    switch (j) {
      case 0: N = (size_t) atoi(argv[i]); break;
      case 1: C = (size_t) atoi(argv[i]); break;
      case 2: H = (size_t) atoi(argv[i]); break;
      case 3: W = (size_t) atoi(argv[i]); break;
      case 4: K = (size_t) atoi(argv[i]); break;
      case 5: R = (size_t) atoi(argv[i]); break;
      case 6: S = (size_t) atoi(argv[i]); break;
      case 7: pad_h = (size_t) atoi(argv[i]); break;
      case 8: pad_w = (size_t) atoi(argv[i]); break;
      case 9: stride_h = (size_t) atoi(argv[i]); break;
      case 10: stride_w = (size_t) atoi(argv[i]); break;
      case 11: dilation_h = (size_t) atoi(argv[i]); break;
      case 12: dilation_w = (size_t) atoi(argv[i]); break;
      default: break;
    }
  }

  printf(
      "Problem size: N = %d, C = %d, H = %d, W = %d, K = %d, R = %d, S = "
      "%d\n",
      N, C, H, W, K, R, S);
  printf("              pad_h = %d, pad_w = %d, stride_h = %d, stride_w = %d\n",
         pad_h, pad_w, stride_h, stride_w);
  printf("              dilation_h = %d, dilation_w = %d\n", dilation_h,
         dilation_w);
  printf("Number of iterations: %d\n", num_iterations);
  printf("Print tensor: %s\n", print ? "on" : "off");
  printf("Validation: %s\n", validation ? "on" : "off");
  printf("\n");
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_world_size);

  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(processor_name, &name_len);

  printf("Hello world from processor %s, rank %d out of %d\n", processor_name,
         mpi_rank, mpi_world_size);

  parse_opt(argc, argv);

  const int ON = N;
  const int OC = K;
  const int OH = 1 + (H + 2 * pad_h - (((R - 1) * dilation_h) + 1)) / stride_h;
  const int OW = 1 + (W + 2 * pad_w - (((S - 1) * dilation_w) + 1)) / stride_w;
  /* Allocate and initialize tensor on CPU */
  printf("Initializing... \n");
  float *I, *F, *O, *BUF1, *BUF2;
  if (mpi_rank == 0) {
    printf("[rank %d] Initializing matrix...\n", mpi_rank);
    I = alloc_tensor(N, C, H, W);
    F = alloc_tensor(K, C, R, S);
    O = alloc_tensor(ON, OC, OH, OW);
    BUF1 = alloc_tensor(C, R, S, N * OH * OW);
    BUF2 = alloc_tensor(K, N, OH, OW);
    rand_tensor(I, N, C, H, W);
    rand_tensor(F, K, C, R, S);
    printf("[rank %d] Initializing matrix done!\n", mpi_rank);
  }
  else {
    printf("[rank %d] Initializing matrix...\n", mpi_rank);
    I = alloc_tensor(N / mpi_world_size, C, H, W);
    F = alloc_tensor(K, C, R, S);
    O = alloc_tensor(ON / mpi_world_size, OC, OH, OW);
    BUF1 = alloc_tensor(C, R, S, N * OH * OW);
    BUF2 = alloc_tensor(K, N, OH, OW);
    printf("[rank %d] Initializing matrix done!\n", mpi_rank);
  }

  /* Initialize Convolution for each processor */
  convolution_init(N / mpi_world_size, C, H, W, K, R, S, pad_h, pad_w, stride_h, stride_w,
                       dilation_h, dilation_w);

  MPI_Barrier(MPI_COMM_WORLD);
  /* Run convolution for num_iterations */
  double elapsed_time_sum = 0;

  for (int i = 0; i < num_iterations; ++i) {
    if (mpi_rank == 0) {
      printf("Calculating...(iter=%d)", i);
      fflush(stdout);
      zero_tensor(O, ON, OC, OH, OW);
    }
    else {
      zero_tensor(O, ON / mpi_world_size, OC, OH, OW);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = get_time();
    convolution(I, F, O, BUF1, BUF2, N, C, H, W, K, R, S, pad_h, pad_w,
                stride_h, stride_w, dilation_h, dilation_w, mpi_rank, mpi_world_size);
    MPI_Barrier(MPI_COMM_WORLD);
    double elapsed_time = get_time() - start_time;
    
    if (mpi_rank == 0) {
      printf("%f sec\n", elapsed_time);
      elapsed_time_sum += elapsed_time;
    }
  }

  if (mpi_rank == 0) {
    if (print) {
      printf("INPUT:\n"); 
      print_tensor(I, N, C, H, W);
      printf("FILTER:\n");
      print_tensor(F, K, C, R, S);
      printf("OUTPUT:\n");
      print_tensor(O, ON, OC, OH, OW);
    }

    if (validation) {
      check_convolution(I, F, O, N, C, H, W, K, R, S, pad_h, pad_w, stride_h,
                        stride_w, dilation_h, dilation_w);
    }

    /* Print performance results */
    double elapsed_time_avg = elapsed_time_sum / num_iterations;
    printf("Avg. time: %f sec\n", elapsed_time_avg);
    printf("Avg. throughput: %f GFLOPS\n",
           2.0 * ON * OC * OH * OW * C * R * S / elapsed_time_avg / 1e9);

  }

  /* Cleanup convolution */
  convolution_cleanup(I, F, O, N, C, H, W, K, R, S, pad_h, pad_w, stride_h,
                      stride_w, dilation_h, dilation_w);


  MPI_Finalize();

  return 0;
}
