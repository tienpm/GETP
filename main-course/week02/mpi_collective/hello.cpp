#include <mpi.h>
#include <stdio.h>

#define CHECK_MPI(call) \
  do {  \
    int code = call;  \
    if (code != MPI_SUCCESS) {  \
      char estr[MPI_MAX_ERROR_STRING];  \
      int elen; \
      MPI_Error_string(code, estr, &elen);  \
      fprintf(stderr, "MPI error (%s:%d): %s\n", __FILE__, __LINE__, estr); \
      MPI_Abort(MPI_COMM_WORLD, code);  \
    } \
  } \
  while (0)

int main(int argc, char **argv) {
  MPI_Init(nullptr, nullptr);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  char hostname[MPI_MAX_PROCESSOR_NAME];
  int hostnamelen;
  MPI_Get_processor_name(hostname, &hostnamelen);
  printf("[%s] Hello, I am rank %d of size %d world!\n", hostname, rank, size);
  MPI_Finalize();

  return 0;
}
