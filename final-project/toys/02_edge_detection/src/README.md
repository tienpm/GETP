# Moreh GPU Enginner Training Program

## Toy Project 02 - Edge detection

### Installation

- GNU compiler
- ROCm/HIP package

### Usage

- Build

```bash
cd 02_edge_detection
make
```

- Edge detection with Sobel filter on GPU

```bash
srun -p EM --gress=gpu:1 ./edge <input filename> <output filename on CPU> <output filename on GPU> <verification 0|1>
```

Example:

```bash
srun -p EM --gress=gpu:1 ./edge Very_Big_Tiger_Cub.jpg cpu_out.jpg cpu_out_very.jpg gpu_out_very.jpg 1
```

### Authors

Tien M. Pham - email: tien.pham@moreh.com.vn

### License

[MIT](https://choosealicense.com/licenses/mit/)
