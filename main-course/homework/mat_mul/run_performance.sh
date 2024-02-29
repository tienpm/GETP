#!/bin/bash

salloc -N 2 --exclusive                              \
  mpirun --bind-to none -mca btl ^openib -npernode 1 \
  numactl --interleave=all ./main -t 32 -n 10 4096 4096 4096 

# salloc -N 2 --exclusive                              \ 
#   mpirun --bind-to none -mca btl ^openib -npernode 1 \
#   numactl --interleave=all ./main -t 32 -n 10 4096 4096 4096 
