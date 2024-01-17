#include <ctype.h>
#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

const int MAX = 13;

static void doFib(int n, int doPrint);

/*
 * unix_error - unix-style error routine.
 */
inline static void unix_error(char *msg) {
  fprintf(stdout, "%s: %s\n", msg, strerror(errno));
  exit(1);
}

int main(int argc, char **argv) {
  int arg;
  int print;

  if (argc != 2) {
    fprintf(stderr, "Usage: fib <num>\n");
    exit(-1);
  }

  if (argc >= 3) {
    print = 1;
  }

  arg = atoi(argv[1]);
  if (arg < 0 || arg > MAX) {
    fprintf(stderr, "number must be between 0 and %d\n", MAX);
    exit(-1);
  }

  doFib(arg, 1);

  return 0;
}

/*
 * Recursively compute the specified number. If print is
 * true, print it. Otherwise, provide it to my parent process.
 *
 * NOTE: The solution must be recursive and it must fork
 * a new child for each call. Each process should call
 * doFib() exactly once.
 */
static void doFib(int n, int doPrint) {
  if (n == 0) {
    if (doPrint) {
      printf("0\n");
    }
    exit(0);
  }

  if (n == 1) {
    if (doPrint) {
      printf("1\n");
    }
    exit(1);
  }
  int status;
  pid_t pid1, pid2;
  int left, right;
  // n - 1
  pid1 = fork();
  if (pid1 < 0)
    perror("Fork error in Fork 1");
  else if (pid1 == 0) {
    doFib(n - 1, 0);
  }
  waitpid(pid1, &status, 0);
  left = WEXITSTATUS(status);

  // n - 2
  pid2 = fork();
  if (pid2 < 0) {
    perror("Fork error in Fork 2");
  } else if (pid2 == 0) {
    doFib(n - 2, 0);
  }
  waitpid(pid2, &status, 0);
  right = WEXITSTATUS(status);

  int ans = left + right;
  if (doPrint) {
    printf("%d\n", ans);
  } else {
    exit(ans);
  }
}
