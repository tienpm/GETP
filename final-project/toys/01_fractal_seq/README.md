# Moreh GPU Enginner Training Program

## Toy Project 01

### Installation

- GNU compiler
- ROCm/HIP packagle

### Usage

- Build

```bash
cd 01_fractal_seq
make
```

- Mandelbrot set

```bash
srun -p EM --gress=gpu:1 ./fractals 1 <output filename> <size=w=h>
```

Example:

```bash
srun -p EM --gress=gpu:1 ./fractals 1 mandelbrot_8192.jpg 8192
```

- Julia set

```bash
srun -p EM --gress=gpu:1 ./fractals 2 julia_8192.jpg 8192
```

### Authors

Tien M. Pham - email: tien.pham@moreh.com.vn

### License

[MIT](https://choosealicense.com/licenses/mit/)
