#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <hip/hip_runtime.h>
#include "hip_helper.h"
#include "timers.h"

#define COUNT_MAX   2000
#define MIN(x,y)    ((x) < (y) ? (x) : (y))
#define numChunk 4

#ifdef SAVE_JPG
void save_jpeg_image(const char* filename, int* r, int* g, int* b, int image_width, int image_height);
#endif

typedef struct {
  unsigned int r;
  unsigned int g;
  unsigned int b;
} RgbColor;

__device__ RgbColor HSVtoRGB(unsigned h, unsigned s, unsigned v);

__global__ void hip_julia(int* r, int* g, int* b, const int w, const int h, 
                          const int maxIterations, const double cRe, const double cIm, 
                          const double zoom, const double moveX, const double moveY, const int beginH, const int endH) {
  int y = blockDim.y * blockIdx.y + threadIdx.y;
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int i;
  double newRe, newIm, oldRe, oldIm;

  y += beginH;

  // if x and y are ouf of alloc memory then return
  if (y >= endH || x >= w)
    return;
  // calculate the initial real and imaginary part of z,
  // based on the pixel location and zoom and position values
  newRe = 1.5 * (x - w / 2) / (0.5 * zoom * w) + moveX;
  newIm = (y - h / 2) / (0.5 * zoom * h) + moveY;

  //start the iteration process
  for (i = 0; i < maxIterations; i++)
  {
    // remember value of previous iteration
    oldRe = newRe;
    oldIm = newIm;

    // the actual iteration, the real and imaginary part are calculated
    newRe = oldRe * oldRe - oldIm * oldIm + cRe;
    newIm = 2 * oldRe * oldIm + cIm;

    // if the point is outside the circle with radius 2: stop
    if ((newRe * newRe + newIm * newIm) > 4) break;
  }

  // use color model conversion to get rainbow palette, 
  // make brightness black if maxIterations reached
  RgbColor color = HSVtoRGB(i % 256, 255, 255 * (i < maxIterations));
  r[y*w+x] = color.r;
  g[y*w+x] = color.g;
  b[y*w+x] = color.b;     
}

// Main part of the below code is originated from Lode Vandevenne's code.
// Please refer to http://lodev.org/cgtutor/juliamandelbrot.html
void julia(int w, int h, char* output_filename) {
  // each iteration, it calculates: new = old*old + c,
  // where c is a constant and old starts at current pixel

  // real and imaginary part of the constant c
  // determinate shape of the Julia Set
  double cRe, cIm;

  // you can change these to zoom and change position
  double zoom = 1, moveX = 0, moveY = 0;

  // after how much iterations the function should stop
  int maxIterations = COUNT_MAX;

#ifndef SAVE_JPG
  FILE *output_unit;
#endif

  double wtime;

  // pick some values for the constant c
  // this determines the shape of the Julia Set
  cRe = -0.7;
  cIm = 0.27015;

  int* r;
  int* g;
  int* b;
  HIP_ERRCHECK(hipHostMalloc((void**)&r, w * h * sizeof(int), hipMemAllocationTypePinned));
  HIP_ERRCHECK(hipHostMalloc((void**)&g, w * h * sizeof(int), hipMemAllocationTypePinned));
  HIP_ERRCHECK(hipHostMalloc((void**)&b, w * h * sizeof(int), hipMemAllocationTypePinned));

  printf( "  Sequential C version\n" );
  printf( "\n" );
  printf( "  Create an ASCII PPM image of the Julia set.\n" );
  printf( "\n" );
  printf( "  An image of the set is created using\n" );
  printf( "    W = %d pixels in the X direction and\n", w );
  printf( "    H = %d pixels in the Y direction.\n", h );

  timer_init();
  timer_start(0);
  
  // 1. Alloc memory for r, g, b on the compute device
  int* d_r;
  int* d_g;
  int* d_b;
  HIP_ERRCHECK(hipMalloc((void**)&d_r, w * h * sizeof(int)));
  HIP_ERRCHECK(hipMalloc((void**)&d_g, w * h * sizeof(int)));
  HIP_ERRCHECK(hipMalloc((void**)&d_b, w * h * sizeof(int)));

  // 2. Divide r, g, b into chunk
  int chunkSize = h / numChunk;
  int chunkBegin[numChunk];
  int chunkEnd[numChunk];
  for (int i = 0; i < numChunk; i++) {
    chunkBegin[i] = chunkSize * i; 
    if (i == chunkSize - 1) {
      chunkEnd[i] = h;
    }
    else {
      chunkEnd[i] = chunkSize * (i + 1);
    }
    // std::cout << chunkBegin[i] << " " << chunkEnd[i] << "\n";
  }

  // 3. Create stream and events
  hipStream_t kernelStream, DtoHStream;
  HIP_ERRCHECK(hipStreamCreate(&kernelStream));
  HIP_ERRCHECK(hipStreamCreate(&DtoHStream));

  hipEvent_t events[numChunk];
  for (int i = 0; i < numChunk; i++) {
    HIP_ERRCHECK(hipEventCreate(&events[i]));
  }

  // 4. Run the kernel to Compute mandelbrot fractal
  int blockSize = 32;
  int sharedMemBytes = 0;
  for (int i = 0; i < numChunk; i++) {
    dim3 blockDim(blockSize, blockSize, 1);
    dim3 gridDim((w + blockSize - 1) / blockSize, ((chunkEnd[i] - chunkBegin[i]) + blockSize - 1) / blockSize, 1);
    // dim3 gridDim((n + blockSize - 1) / blockSize, ((chunkEnd[i] - chunkBegin[i]) + blockSize - 1) / blockSize, 1);
    hipLaunchKernelGGL(hip_julia, gridDim, blockDim, sharedMemBytes, kernelStream, 
                     d_r, d_g, d_b, w, h, maxIterations, cRe, cIm, zoom, moveX, moveY, chunkBegin[i], chunkEnd[i]);
    HIP_ERRCHECK(hipGetLastError());
    HIP_ERRCHECK(hipEventRecord(events[i], kernelStream));
  }

  // 5. Copy the buffer for r, g, b on the compute device to host
  // HIP_ERRCHECK(hipStreamSynchronize(kernelStream));
  for (int i = 0; i < numChunk; i++) {
    HIP_ERRCHECK(hipStreamWaitEvent(DtoHStream, events[i], 0));
    HIP_ERRCHECK(hipMemcpyAsync(&r[chunkBegin[i] * w], &d_r[chunkBegin[i] * w], (chunkEnd[i] - chunkBegin[i]) * w * sizeof(int), hipMemcpyDeviceToHost, DtoHStream));
    HIP_ERRCHECK(hipMemcpyAsync(&g[chunkBegin[i] * w], &d_g[chunkBegin[i] * w], (chunkEnd[i] - chunkBegin[i]) * w * sizeof(int), hipMemcpyDeviceToHost, DtoHStream));
    HIP_ERRCHECK(hipMemcpyAsync(&b[chunkBegin[i] * w], &d_b[chunkBegin[i] * w], (chunkEnd[i] - chunkBegin[i]) * w * sizeof(int), hipMemcpyDeviceToHost, DtoHStream));
  }

  // 6. Synchronize all stream
  HIP_ERRCHECK(hipStreamSynchronize(DtoHStream));
  HIP_ERRCHECK(hipDeviceSynchronize());

  // 7. Free buffer, destruct stream and event on the compute device
  HIP_ERRCHECK(hipFree(d_r));
  HIP_ERRCHECK(hipFree(d_g));
  HIP_ERRCHECK(hipFree(d_b));
  HIP_ERRCHECK(hipStreamDestroy(kernelStream));
  HIP_ERRCHECK(hipStreamDestroy(DtoHStream));
  for (int i = 0; i < numChunk; i++) {
    HIP_ERRCHECK(hipEventDestroy(events[i]));
  }
  timer_stop(0);
  wtime = timer_read(0);
  printf( "\n" );
  printf( "  Time = %lf seconds.\n", wtime );

#ifdef SAVE_JPG
  save_jpeg_image(output_filename, r, g, b, w, h);
#else
  // Write data to an ASCII PPM file.
  output_unit = fopen( output_filename, "wt" );

  fprintf( output_unit, "P3\n" );
  fprintf( output_unit, "%d  %d\n", h, w );
  fprintf( output_unit, "%d\n", 255 );
  for ( int i = 0; i < h; i++ )
  {
    for ( int jlo = 0; jlo < w; jlo = jlo + 4 )
    {
      int jhi = MIN( jlo + 4, w );
      for ( int j = jlo; j < jhi; j++ )
      {
        fprintf( output_unit, "  %d  %d  %d", r[i*w+j], g[i*w+j], b[i*w+j] );
      }
      fprintf( output_unit, "\n" );
    }
  }

  fclose( output_unit );
#endif
  printf( "\n" );
  printf( "  Graphics data written to \"%s\".\n\n", output_filename );

  // Terminate.
  HIP_ERRCHECK(hipHostFree(r));
  HIP_ERRCHECK(hipHostFree(g));
  HIP_ERRCHECK(hipHostFree(b));
}


__device__ RgbColor HSVtoRGB(unsigned h, unsigned s, unsigned v)
{
  RgbColor rgb;
  unsigned char region, remainder, p, q, t;

  if (s == 0)
  {
    rgb.r = v;
    rgb.g = v;
    rgb.b = v;
    return rgb;
  }

  region = h / 43;
  remainder = (h - (region * 43)) * 6; 

  p = (v * (255 - s)) >> 8;
  q = (v * (255 - ((s * remainder) >> 8))) >> 8;
  t = (v * (255 - ((s * (255 - remainder)) >> 8))) >> 8;

  switch (region)
  {
    case 0:
      rgb.r = v; rgb.g = t; rgb.b = p;
      break;
    case 1:
      rgb.r = q; rgb.g = v; rgb.b = p;
      break;
    case 2:
      rgb.r = p; rgb.g = v; rgb.b = t;
      break;
    case 3:
      rgb.r = p; rgb.g = q; rgb.b = v;
      break;
    case 4:
      rgb.r = t; rgb.g = p; rgb.b = v;
      break;
    default:
      rgb.r = v; rgb.g = p; rgb.b = q;
      break;
  }

  return rgb;
}
