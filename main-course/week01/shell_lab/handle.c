#include "util.h"
#include <assert.h>
#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

/*
 * First, print out the process ID of this process.
 *
 * Then, set up the signal handler so that ^C causes
 * the program to print "Nice try.\n" and continue looping.
 *
 * Finally, loop forever, printing "Still here\n" once every
 * second.
 */
void signal_handler01(int signum);
void signal_handler02(int signum);

int main(int argc, char **argv) {
  printf("Process ID: %d\n", getpid());

  // signal handler
  signal(SIGINT, signal_handler01);
  signal(SIGUSR1, signal_handler02);

  while (1) {
    struct timespec remaining, request = {1, 0};
    printf("Still here\n");
    nanosleep(&request, &remaining);
  }

  return 0;
}

void signal_handler01(int signum) {
  size_t bytes;
  const int STDOUT = 1;
  bytes = write(STDOUT, "\nNice try.\n", 15);
  if (bytes != 15)
    exit(-999);
}

void signal_handler02(int signum) {
  printf("exiting\n");
  exit(0);
}
