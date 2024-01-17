#include <cstdio>
#include <random>
#include <complex>
#include <sys/time.h>
#include <immintrin.h>

const double PI = 3.1415926535897932384626433832;

static double get_time() {
    struct timeval tv;
    gettimeofday(&tv, 0);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

struct Complex {
	double real = 0;
	double imag = 0;
};

void Shuffle(Complex* arr, int n) {
  Complex* buf = new Complex[n / 2];
  // Copy odd elements to buffer
	for (int i = 0; i < n / 2; ++i) {
		buf[i] = arr[i * 2 + 1];
  }
  // Move even elements to front
	for (int i = 0; i < n / 2; i++) {
    arr[i] = arr[i * 2];
  }
  // Copy odd elements back to end
	for (int i = 0; i < n / 2; i++) {
    arr[i + n / 2] = buf[i];
  }
	delete[] buf;
}

void FFT(Complex* arr, int n) {
	if (n < 2) return;
  Shuffle(arr, n);
  FFT(arr, n / 2);
  FFT(arr + n / 2, n / 2);
  for (int k = 0; k < n / 2; k++) {
    Complex e = arr[k];
    Complex o = arr[k + n / 2];
    Complex w;
    w.real = cos(-2.0 * PI * k / n);
    w.imag = sin(-2.0 * PI * k / n);
    Complex wo;
    wo.real = w.real * o.real - w.imag * o.imag;
    wo.imag = w.real * o.imag + w.imag * o.real;
    arr[k].real = e.real + wo.real;
    arr[k].imag = e.imag + wo.imag;
    arr[k + n / 2].real = e.real - wo.real;
    arr[k + n / 2].imag = e.imag - wo.imag;
  }
}

void FFT_SIMD(Complex* arr, int n) {
  // IMPLEMENT HERE
  if (n < 2) return;
  Shuffle(arr, n);
  FFT(arr, n / 2);
  FFT(arr + n / 2, n / 2);
  // FFT calculation
  for (int k = 0; k < n / 2; k++) {
    __m256d e_real = _mm256_load_pd((double*)&arr[k].real);    
    __m256d e_imag = _mm256_load_pd((double*)&arr[k].imag);
    __m256d o_real = _mm256_load_pd((double*)&arr[k+n/2].real);
    __m256d o_imag = _mm256_load_pd((double*)&arr[k+n/2].imag);
    __m256d w_real;
    __m256d w_imag;
    // Initial w value 
    double w_cos = cos(-2.0 * PI * k / n);
    double w_sin = sin(-2.0 * PI * k / n);
    w_real = _mm256_set1_pd(w_cos);
    w_imag = _mm256_set1_pd(w_sin);
    // Calculate wo value  
    __m256d wo_real = _mm256_mul_pd(w_imag, o_imag);
    wo_real = _mm256_fmsub_pd(w_real, o_real, wo_real);
    __m256d wo_imag = _mm256_mul_pd(w_imag, o_real);
    wo_imag = _mm256_fmadd_pd(w_real, o_imag, wo_imag);
    // Calculate final result
    __m256d k_real = _mm256_add_pd(e_real, wo_real);
    __m256d k_imag = _mm256_add_pd(e_imag, wo_imag);
    __m256d k_d2_real = _mm256_sub_pd(e_real, wo_real);
    __m256d k_d2_imag = _mm256_sub_pd(e_imag, wo_imag);
    // save result 
    _mm256_store_pd(&arr[k].real, k_real);
    _mm256_store_pd(&arr[k].imag, k_imag);
    _mm256_store_pd(&arr[k+n/2].real, k_d2_real);
    _mm256_store_pd(&arr[k+n/2].imag, k_d2_imag); 
  }
}

int main() {
  std::default_random_engine generator(42);
  std::uniform_real_distribution<double> distribution(-1.0, 1.0);

  const int NELEM = 8192;
  double* a = (double*)aligned_alloc(32, sizeof(double) * NELEM);
  double* b = (double*)aligned_alloc(32, sizeof(double) * NELEM);
  double* c = (double*)aligned_alloc(32, sizeof(double) * NELEM);
  Complex* ca = (Complex*)aligned_alloc(32, sizeof(Complex) * NELEM);
  Complex* cb = (Complex*)aligned_alloc(32, sizeof(Complex) * NELEM);
  Complex* cc = (Complex*)aligned_alloc(32, sizeof(Complex) * NELEM);

  for (int i = 0; i < NELEM; ++i) {
    a[i] = distribution(generator);
    b[i] = distribution(generator);
  }

  double st, et;

  // 1. Direct convolution
  st = get_time();
  for (int i = 0; i < NELEM; ++i) {
    c[i] = 0;
    for (int j = 0; j < NELEM; ++j) {
      c[i] += a[j] * b[(i - j + NELEM) % NELEM];
    }
  }
  et = get_time();
  printf("Direct convolution: %lf sec\n", et - st);

  // 2. Convolution with FFT
  st = get_time();
  for (int i = 0; i < NELEM; ++i) {
    ca[i].real = a[i];
    ca[i].imag = 0;
    cb[i].real = b[i];
    cb[i].imag = 0;
  }
  FFT(ca, NELEM);
  FFT(cb, NELEM);
  for (int i = 0; i < NELEM; ++i) {
    cc[i].real = ca[i].real * cb[i].real - ca[i].imag * cb[i].imag;
    cc[i].imag = ca[i].real * cb[i].imag + ca[i].imag * cb[i].real;
  }
  FFT(cc, NELEM);
  et = get_time();
  printf("FFT convolution: %lf sec\n", et - st);

  // 3. Convolution with FFT (SIMD)
  st = get_time();
  for (int i = 0; i < NELEM; ++i) {
    ca[i].real = a[i];
    ca[i].imag = 0;
    cb[i].real = b[i];
    cb[i].imag = 0;
  }
  FFT_SIMD(ca, NELEM);
  FFT_SIMD(cb, NELEM);
  for (int i = 0; i < NELEM; ++i) {
    cc[i].real = ca[i].real * cb[i].real - ca[i].imag * cb[i].imag;
    cc[i].imag = ca[i].real * cb[i].imag + ca[i].imag * cb[i].real;
  }
  FFT_SIMD(cc, NELEM);
  et = get_time();
  printf("FFT (SIMD) convolution: %lf sec\n", et - st);

  // 4. Compare
  int err_cnt = 0, err_threshold = 10;
  for (int i = 0; i < NELEM; ++i) {
    double expected = c[i];
    double actual = cc[(NELEM - i) % NELEM].real / NELEM;
    if (fabs(expected - actual) > 1e-6) {
      ++err_cnt;
      if (err_cnt <= err_threshold) {
        printf("Error at %d: expected %lf, actual %lf\n", i, expected, actual);
      }
      if (err_cnt == err_threshold + 1) {
        printf("Too many errors. Stop printing error messages.\n");
        exit(1);
      }
    }
  }
  printf("Result: VALID\n");

  free(a); free(b); free(c);
  free(ca); free(cb); free(cc);

  return 0;
}
