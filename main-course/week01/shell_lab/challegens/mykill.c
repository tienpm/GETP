#include <asm-generic/errno-base.h>
#include <assert.h>
#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

int main(int argc, char **argv) {
  if (argc != 2) {
    fprintf(stderr, "Usage: %s <procress_id>", argv[0]);
    return 1;
  }

  // Validate process id for safety
  pid_t pid = atoi(argv[1]);
  if (pid <= 1 || (kill(pid, 0) == -1 && errno != ESRCH)) {
    fprintf(stderr, "Invalid process id or permission denied.\n");
    return 1;
  }

  // Send SIGURS1  signal
  if (kill(pid, SIGUSR1) == -1) {
    perror("Error kill");
    return 1;
  }

  printf("Signal SIGUSR1 sent to process %d.\n", pid);

  return 0;
}
