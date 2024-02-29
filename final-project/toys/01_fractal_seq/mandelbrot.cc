#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <hip/hip_runtime.h>
#include "timers.h"
#include "hip_helper.h"

#define COUNT_MAX   5000
#define MIN(x,y)    ((x) < (y) ? (x) : (y))
#define numChunk 4

#ifdef SAVE_JPG
void save_jpeg_image(const char* filename, int* r, int* g, int* b, int image_width, int image_height);
#endif

__global__ void hip_mandelbrot(int* r, int* g, int* b, const int m, const int n, const int count_max, 
                               const float x_max, const float x_min, const float y_max, const float y_min, 
                               const int beginM, const int endM) {
  int i = blockDim.x * blockIdx.x + threadIdx.x; 
  int j = blockDim.y * blockIdx.y + threadIdx.y;
  int k, count;
  float x, y, x1, y1, x2, y2;

  i += beginM;
  // if i or j is out of alloc memory bound then return
  if (i >= endM || j >= n)
    return;

  x = ( ( float ) (     j - 1 ) * x_max  
      + ( float ) ( m - j     ) * x_min )
      / ( float ) ( m     - 1 );

  y = ( ( float ) (     i - 1 ) * y_max  
      + ( float ) ( n - i     ) * y_min )
      / ( float ) ( n     - 1 );

  count = 0;

  
  x1 = x;
  y1 = y;
  
  for (k = 1; k <= count_max; k++ )
  {
    x2 = x1 * x1 - y1 * y1 + x;
    y2 = 2.0 * x1 * y1 + y;

    if ( x2 < -2.0 || 2.0 < x2 || y2 < -2.0 || 2.0 < y2 )
    {
      count = k;
      break;
    }
    x1 = x2;
    y1 = y2;
  }

  if ( ( count % 2 ) == 1 )
  {
    r[i * n + j] = 255;
    g[i * n + j] = 255;
    b[i * n + j] = 255;
  }
  else
  {
    int c = ( int ) ( 255.0 * sqrtf ( sqrtf ( sqrtf (
      ( ( float ) ( count ) / ( float ) ( count_max ) ) ) ) ) );
    r[i * n + j] = 3 * c / 5;
    g[i * n + j] = 3 * c / 5;
    b[i * n + j] = c;
  }
}

void mandelbrot(int m, int n, char* output_filename) {
  int count_max = COUNT_MAX;
#ifndef SAVE_JPG
  int jhi, jlo;
  FILE *output_unit;
#endif
  double wtime;

  float x_max =   1.25;
  float x_min = - 2.25;
  float y_max =   1.75;
  float y_min = - 1.75;

  int* r;
  int* g;
  int* b;
  HIP_ERRCHECK(hipHostMalloc((void**)&r, m * n * sizeof(int), hipMemAllocationTypePinned));
  HIP_ERRCHECK(hipHostMalloc((void**)&g, m * n * sizeof(int), hipMemAllocationTypePinned));
  HIP_ERRCHECK(hipHostMalloc((void**)&b, m * n * sizeof(int), hipMemAllocationTypePinned));

  printf( "  Sequential C version\n" );
  printf( "\n" );
  printf( "  Create an ASCII PPM image of the Mandelbrot set.\n" );
  printf( "\n" );
  printf( "  For each point C = X + i*Y\n" );
  printf( "  with X range [%g,%g]\n", x_min, x_max );
  printf( "  and  Y range [%g,%g]\n", y_min, y_max );
  printf( "  carry out %d iterations of the map\n", count_max );
  printf( "  Z(n+1) = Z(n)^2 + C.\n" );
  printf( "  If the iterates stay bounded (norm less than 2)\n" );
  printf( "  then C is taken to be a member of the set.\n" );
  printf( "\n" );
  printf( "  An image of the set is created using\n" );
  printf( "    M = %d pixels in the X direction and\n", m );
  printf( "    N = %d pixels in the Y direction.\n", n );

  timer_init();
  timer_start(0);

  // 1. Alloc memory for r, g, b on the compute device
  int *d_r;
  int *d_g;
  int *d_b;
  HIP_ERRCHECK(hipMalloc((void**)&d_r, m * n * sizeof(int)));
  HIP_ERRCHECK(hipMalloc((void**)&d_g, m * n * sizeof(int)));
  HIP_ERRCHECK(hipMalloc((void**)&d_b, m * n * sizeof(int)));

  // 2. Divide r, g, b into chunk
  int chunkSize = m / numChunk;
  int chunkBegin[numChunk];
  int chunkEnd[numChunk];
  for (int i = 0; i < numChunk; i++) {
    chunkBegin[i] = chunkSize * i; 
    if (i == chunkSize - 1) {
      chunkEnd[i] = m;
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
    dim3 gridDim(((chunkEnd[i] - chunkBegin[i]) + blockSize - 1) / blockSize, (n + blockSize - 1) / blockSize, 1);
    // dim3 gridDim((n + blockSize - 1) / blockSize, ((chunkEnd[i] - chunkBegin[i]) + blockSize - 1) / blockSize, 1);
    hipLaunchKernelGGL(hip_mandelbrot, gridDim, blockDim, sharedMemBytes, kernelStream, 
                       d_r, d_g, d_b, m, n, count_max, x_max, x_min, y_max, y_min, chunkBegin[i], chunkEnd[i]);
    HIP_ERRCHECK(hipGetLastError());
    HIP_ERRCHECK(hipEventRecord(events[i], kernelStream));
  }

  // 5. Copy the buffer for r, g, b on the compute device to host
  // HIP_ERRCHECK(hipStreamSynchronize(kernelStream));
  for (int i = 0; i < numChunk; i++) {
    HIP_ERRCHECK(hipStreamWaitEvent(DtoHStream, events[i], 0));
    HIP_ERRCHECK(hipMemcpyAsync(&r[chunkBegin[i] * n], &d_r[chunkBegin[i] * n], (chunkEnd[i] - chunkBegin[i]) * n * sizeof(int), hipMemcpyDeviceToHost, DtoHStream));
    HIP_ERRCHECK(hipMemcpyAsync(&g[chunkBegin[i] * n], &d_g[chunkBegin[i] * n], (chunkEnd[i] - chunkBegin[i]) * n * sizeof(int), hipMemcpyDeviceToHost, DtoHStream));
    HIP_ERRCHECK(hipMemcpyAsync(&b[chunkBegin[i] * n], &d_b[chunkBegin[i] * n], (chunkEnd[i] - chunkBegin[i]) * n * sizeof(int), hipMemcpyDeviceToHost, DtoHStream));
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
  // Write data to an JPEG file.
  save_jpeg_image(output_filename, r, g, b, n, m);
#else
  // Write data to an ASCII PPM file.
  output_unit = fopen( output_filename, "wt" );

  fprintf( output_unit, "P3\n" );
  fprintf( output_unit, "%d  %d\n", n, m );
  fprintf( output_unit, "%d\n", 255 );
  for ( i = 0; i < m; i++ )
  {
    for ( jlo = 0; jlo < n; jlo = jlo + 4 )
    {
      jhi = MIN( jlo + 4, n );
      for ( j = jlo; j < jhi; j++ )
      {
        fprintf( output_unit, "  %d  %d  %d", r[i * n + j], g[i * n + j], b[i * n + j] );
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
