#!/bin/bash

salloc -N 2 --exclusive                              \
  mpirun --bind-to none -mca btl ^openib -npernode 1 \
  numactl --interleave=all ./main -n 10 -v 16 8 16 256 256 3 8 8 

# salloc -N 2 --exclusive                              \ 
#   mpirun --bind-to none -mca btl ^openib -npernode 1 \
#   numactl --interleave=all ./main -t 32 -n 10 4096 4096 4096 
