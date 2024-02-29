# Problem : Best matrix multiplycation on CPU

## Setup

```
cd mat_mul
make
```

## Execution

Examaple run:

```
srun run.sh -v -n 10 32 8 16 256 256 3 8 8
```

Run perfomance (have to include "numactl --interleave=all" in salloc command):

```
srun run_performance.sh
```
