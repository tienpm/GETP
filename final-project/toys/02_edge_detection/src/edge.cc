#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <cstring>
#include <time.h>

#include <hip/hip_runtime.h>
#include <jpeglib.h>    
#include <jerror.h>

using namespace std;

#ifndef HIP_ERRORCHK
#define HIP_ERRORCHK(error)                       \
    if(error != hipSuccess)                       \
    {                                             \
        fprintf(stderr,                           \
                "Hip error: '%s'(%d) at %s:%d\n", \
                hipGetErrorString(error),         \
                error,                            \
                __FILE__,                         \
                __LINE__);                        \
        exit(EXIT_FAILURE);                       \
    }
#endif

#define MODE 0
#define BLOCK_SIZE 32

int timespec_subtract (struct timespec* result, struct timespec *x, struct timespec *y)
{
  /* Perform the carry for the later subtraction by updating y. */
  if (x->tv_nsec < y->tv_nsec) {
    int nsec = (y->tv_nsec - x->tv_nsec) / 1000000000 + 1;
    y->tv_nsec -= 1000000000 * nsec;
    y->tv_sec += nsec;
  }
  if (x->tv_nsec - y->tv_nsec > 1000000000) {
    int nsec = (x->tv_nsec - y->tv_nsec) / 1000000000;
    y->tv_nsec += 1000000000 * nsec;
    y->tv_sec -= nsec;
  }

  /* Compute the time remaining to wait.
     tv_nsec is certainly positive. */
  result->tv_sec = x->tv_sec - y->tv_sec;
  result->tv_nsec = x->tv_nsec - y->tv_nsec;

  /* Return 1 if result is negative. */
  return x->tv_sec < y->tv_sec;
}



struct Pixel {
  unsigned char r, g, b;
};

struct Image {
  Pixel* pixels;
  int width, height;

  Pixel& getPixel(int x, int y) {
    return pixels[y * width + x];
  }

  void copy(Image& out) {
    out.width = width;
    out.height = height;
    out.pixels = (Pixel*)malloc(width*height*sizeof(Pixel));
    memcpy(out.pixels, pixels, width * height * sizeof(Pixel));
  }
};

// Function to read an image as a JPG file
bool readJPEGImage(const string& filename, Image& img) {
  struct jpeg_decompress_struct cinfo;
  struct jpeg_error_mgr jerr;
  JSAMPROW row_pointer;
	FILE* fp;

  if ((fp = fopen(filename.c_str(), "rb")) == NULL) {
    fprintf(stderr, "can't open %s\n", filename.c_str());
    return false;
  }

  cinfo.err = jpeg_std_error(&jerr);

  jpeg_create_decompress(&cinfo);
  jpeg_stdio_src(&cinfo, fp);

  jpeg_read_header(&cinfo, TRUE);
  jpeg_start_decompress(&cinfo);

  if(cinfo.num_components != 3) {
    fprintf(stderr, "JPEG file with 3 channels is only supported\n");
    fprintf(stderr, "%s has %d channels\n", filename.c_str(), cinfo.num_components);
    return false;
  }

  img.width = cinfo.output_width;
  img.height = cinfo.output_height;

  img.pixels = (Pixel*)malloc(img.width * img.height * sizeof(Pixel));
	for(int i = 0; i < img.height; i++ ) {
    row_pointer = (JSAMPROW)&img.pixels[i * img.width];
    jpeg_read_scanlines(&cinfo, &row_pointer, 1);
  }

  jpeg_finish_decompress(&cinfo);
  jpeg_destroy_decompress(&cinfo);
  
  fclose(fp);
  return true;
}

// Function to save an image as a JPG file
bool saveJPEGImage(const string& filename, const Image& img) {
  struct jpeg_compress_struct cinfo;
  struct jpeg_error_mgr jerr;
  JSAMPROW row_pointer;
	FILE* fp;

  cinfo.err = jpeg_std_error(&jerr);

  if ((fp = fopen(filename.c_str(), "wb")) == NULL) {
    fprintf(stderr, "can't open %s\n", filename.c_str());
    return false;
  }

	jpeg_create_compress(&cinfo);
	jpeg_stdio_dest(&cinfo, fp);

	cinfo.image_width = img.width;
	cinfo.image_height = img.height;
	cinfo.input_components = 3;
	cinfo.in_color_space = JCS_RGB;

	jpeg_set_defaults(&cinfo);
	jpeg_start_compress(&cinfo, TRUE);

	for(int i = 0; i < img.height; i++ )
	{
		row_pointer = (JSAMPROW)&img.pixels[i * img.width];
		jpeg_write_scanlines(&cinfo, &row_pointer, 1);
	}

	jpeg_finish_compress(&cinfo);
	fclose(fp);

  return true;
}



// Function to apply the Sobel filter for edge detection
void applySobelFilter(Image& input, Image& output, const int filterX[3][3], const int filterY[3][3]) {
  for (int y = 1; y < input.height - 1; ++y) {
    for (int x = 1; x < input.width - 1; ++x) {
      int gx = 0, gy = 0;
      for (int i = -1; i <= 1; ++i) {
        for (int j = -1; j <= 1; ++j) {
          Pixel p = input.getPixel(x + j, y + i);
          gx += (p.r + p.g + p.b) / 3 * filterX[i + 1][j + 1];
          gy += (p.r + p.g + p.b) / 3 * filterY[i + 1][j + 1];
        }
      }
      int magnitude = static_cast<int>(sqrt(gx * gx + gy * gy));
      magnitude = min(max(magnitude, 0), 255);
      output.getPixel(x, y) = {static_cast<unsigned char>(magnitude),
        static_cast<unsigned char>(magnitude),
        static_cast<unsigned char>(magnitude)};
    }
  }
}

inline __device__ int getPixelPos(int x, int y, int offset) {
  return y * offset + x;
}

__global__ void convStaticKernel(Pixel *input, Pixel *output, const int h, const int w);


void applySobelFilter_hip(Image& input, Image& output, const int filterX[3][3], const int filterY[3][3]) {
  // TODO:
  int h = input.height, w = input.width;
  // 0. Alloc and copy pinned memory for input Image object on the host device
  Pixel *pinImage;
  HIP_ERRORCHK(hipHostMalloc(&pinImage,  h * w * sizeof(Pixel), hipHostMallocPortable));
  HIP_ERRORCHK(hipMemcpy(pinImage, input.pixels, h * w * sizeof(Pixel), hipMemcpyHostToHost));
  // // 1. Alloc and copy device memory for input Image object
  Pixel *d_input;
  HIP_ERRORCHK(hipMalloc(&d_input, h * w * sizeof(Pixel)));
  HIP_ERRORCHK(hipMemcpy(d_input, pinImage, h * w * sizeof(Pixel), hipMemcpyHostToDevice));
  Pixel *d_output;
  HIP_ERRORCHK(hipMalloc(&d_output, h * w * sizeof(Pixel)));

  // 2. Computing process
  dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, 1);
  dim3 gridDim((h + BLOCK_SIZE - 1) / BLOCK_SIZE, (w + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);

  #if MODE == 0
    /* Use 1 GPU and default stream */
    hipLaunchKernelGGL(convStaticKernel, 
                       gridDim, blockDim, 0, 0, 
                       d_input, d_output, h, w);
    HIP_ERRORCHK(hipDeviceSynchronize());
    HIP_ERRORCHK(hipGetLastError());
  #elif MODE == 1
    /* Use 1 GPU and 1 stream for H2D mem, 1 stream for computing, 1 stream for D2H mem */
    printf("I'm in mode 1\n");

  #elif MODE == 2 
    /* Use 2 GPU and defaul stream */
  #elif MODE == 3
    /* Use 2 GPU, each GPU have 1 stream for H2D mem, 1 stream for computing, 1 stream for D2H mem */
  #endif
  // 3. Copy memmory from device to host for output Image object
  HIP_ERRORCHK(hipMemcpy(output.pixels, d_output, h * w * sizeof(Pixel), hipMemcpyDeviceToHost));
  // 4. Free the declared memmory, destruct stream and event on the compute device
  HIP_ERRORCHK(hipHostFree(pinImage));
  HIP_ERRORCHK(hipFree(d_input));
  HIP_ERRORCHK(hipFree(d_output));
}

int main(int argc, char **argv) {
  string inputFilename;
  string outputFilename;
  string outputFilename_hip;
  int verify = 0;

  if(argc < 4) {
    fprintf(stderr, "$> edge <input_filename> <output_filename_seq> <output_filename_hip> <verification:0|1>\n");
    return 1;
  }
  else {
    inputFilename = argv[1];
    outputFilename = argv[2];
    outputFilename_hip = argv[3];
    if(argc >4) {
      verify = atoi(argv[4]);
    }
  }

  const int filterX[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
  const int filterY[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

  struct timespec start, end, spent;

  Image inputImage;
  Image outputImage;
  Image outputImage_hip;

  if (!readJPEGImage(inputFilename, inputImage)) {
    return -1;
  }

  inputImage.copy(outputImage);     // Copy input image properties to output image
  inputImage.copy(outputImage_hip); // Copy input image properties to output image

  clock_gettime(CLOCK_MONOTONIC, &start);
  applySobelFilter(inputImage, outputImage, filterX, filterY);
  clock_gettime(CLOCK_MONOTONIC, &end);
  timespec_subtract(&spent, &end, &start);
  printf("CPU Time spent: %ld.%09ld\n", spent.tv_sec, spent.tv_nsec);

  clock_gettime(CLOCK_MONOTONIC, &start);

  // You may modify this code part
  //{
  applySobelFilter_hip(inputImage, outputImage_hip, filterX, filterY);
  //}

  clock_gettime(CLOCK_MONOTONIC, &end);
  timespec_subtract(&spent, &end, &start);
  printf("GPU Time spent: %ld.%09ld\n", spent.tv_sec, spent.tv_nsec);
  //}


  // Save the output image
  saveJPEGImage(outputFilename, outputImage);
  saveJPEGImage(outputFilename_hip, outputImage_hip);

  //verfication (CPU vs GPU)
  if(verify == 1) { 
    //Verification
    bool pass = true;
    int count = 0;
    for(int i=0;i<outputImage.width*outputImage.height;i++) {
      if(outputImage.pixels[i].r != outputImage_hip.pixels[i].r) {
        // printf("[%d] r=%d vs %d : %d\n", i, outputImage.pixels[i].r, outputImage_hip.pixels[i].r, inputImage.pixels[i].r);
        pass = false;
        count++;
        // printf("[%d] ", i);
      }
      if(outputImage.pixels[i].g != outputImage_hip.pixels[i].g) {
        // printf("[%d] g=%d vs %d : %d\n", i, outputImage.pixels[i].g, outputImage_hip.pixels[i].g, inputImage.pixels[i].g);
        pass = false;
        count++;
      }
      if(outputImage.pixels[i].b != outputImage_hip.pixels[i].b) {
        // printf("[%d] b=%d vs %d : %d\n", i, outputImage.pixels[i].b, outputImage_hip.pixels[i].b, inputImage.pixels[i].b);
        pass = false;
        count++;
      }
    }
    if(pass) {
      printf("Verification Pass!\n");
    }
    else {
      printf("Verification Failed! (%d)\n", count);
    }
  }

  free(inputImage.pixels);
  free(outputImage.pixels);
  free(outputImage_hip.pixels);

  return 0;
}

__global__ void convStaticKernel(Pixel *input, Pixel *output, const int h, const int w) {
  const int filterX[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
  const int filterY[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
  // Calculate thread indexhttps://www.youtube.com/watch?v=mvWlHDl7QsQ
  int ty = threadIdx.x + blockIdx.x * blockDim.x;
  int tx = threadIdx.y + blockIdx.y * blockDim.y;

  // Define shared memmory for filter
  __shared__ int filterX_s[3][3];
  __shared__ int filterY_s[3][3];

  // Skip if output pixel is out of the valid range
  if (tx < 0 || tx >= w || ty < 0 || ty >= h) {
    return;
  }
  else if (tx == 0 || tx == w-1 || ty == 0 || ty == h-1) {
    output[getPixelPos(tx, ty, w)] = input[getPixelPos(tx, ty, w)];
    return;
  }

  // Load filter into shared memory with padding to handle boundaries
  int fx = threadIdx.x;
  int fy = threadIdx.y;

  if (fx < 3 && fy < 3) {
    filterX_s[fy][fx] = filterX[fy][fx];
    filterY_s[fy][fx] = filterY[fy][fx];
  } 
  // else {
  //   filterX_s[fy][fx] = 0; // Zero padding
  //   filterY_s[fy][fx] = 0; // Zero padding
  // }

  __syncthreads();  // Ensure all filter values are loaded into shared memory
  
  int gx = 0, gy = 0;
  for (int i = -1; i <= 1; ++i) {
    for (int j = -1; j <= 1; ++j) {
      Pixel p = input[getPixelPos(tx + j, ty + i, w)];
      gx += (p.r + p.g + p.b) / 3 * filterX_s[i + 1][j + 1]; // filterX_s[i + 1][j + 1];
      gy += (p.r + p.g + p.b) / 3 * filterY_s[i + 1][j + 1]; // filthttps://www.youtube.com/watch?v=mvWlHDl7QsQerY_s[i + 1][j + 1];
    }
  }

  // Store result in output
  int magnitude = static_cast<int>(sqrt(gx * gx + gy * gy));
  magnitude = min(max(magnitude, 0), 255);
  output[getPixelPos(tx, ty, w)] = {static_cast<unsigned char>(magnitude),
    static_cast<unsigned char>(magnitude),
    static_cast<unsigned char>(magnitude)};
}
