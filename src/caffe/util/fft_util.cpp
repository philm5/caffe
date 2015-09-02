#ifdef _OPENMP
#include <omp.h>
#endif
#include "caffe/util/fft_util.hpp"

namespace caffe {

double cpu_time(void)
{
  double value;
#ifdef _OPENMP
  value = omp_get_wtime();
#else
  value = (double) clock () / (double) CLOCKS_PER_SEC;
#endif
    return value;
}

template<>
void *fft_cpu_malloc<float>(size_t n) {
  return reinterpret_cast<void *>(fftwf_malloc(n));
}

template<>
void *fft_cpu_malloc<double>(size_t n) {
  return reinterpret_cast<void *>(fftw_malloc(n));
}

template<>
void *fft_plan_dft_r2c_2d<float>(int n0, int n1, float *in, std::complex<float> *out, unsigned flags) {
  return reinterpret_cast<void *>(fftwf_plan_dft_r2c_2d(n0, n1, in, reinterpret_cast<fftwf_complex *>(out), flags));
}

template<>
void *fft_plan_dft_r2c_2d<double>(int n0, int n1, double *in, std::complex<double> *out, unsigned flags) {
  return reinterpret_cast<void *>(fftw_plan_dft_r2c_2d(n0, n1, in, reinterpret_cast<fftw_complex *>(out), flags));
}

template<>
void *fft_plan_many_dft_r2c_2d<float>(int n0,
                                      int n1,
                                      int how_many,
                                      float *in,
                                      std::complex<float> *out,
                                      unsigned flags) {
  int rank = 2;
  int n[] = {n0, n1};
  int idist = n0 * n1; /* = 256*256, the distance in memory
                                          between the first element
                                          of the first array and the
                                          first element of the second array */
  int istride = 1; /* array is contiguous in memory */
  int *inembed = NULL;

  // out
  int odist = n0 * (n1 / 2 + 1);
  int ostride = 1;
  int *onembed = NULL;

  return reinterpret_cast<void *>(fftwf_plan_many_dft_r2c(rank, n, how_many, in, inembed, istride, idist,
                                                          reinterpret_cast<fftwf_complex *>(out), onembed, ostride,
                                                          odist, flags));
}

template<>
void *fft_plan_many_dft_r2c_2d<double>(int n0,
                                       int n1,
                                       int how_many,
                                       double *in,
                                       std::complex<double> *out,
                                       unsigned flags) {
  int rank = 2;
  int n[] = {n0, n1};
  int idist = n0 * n1; /* = 256*256, the distance in memory
                                          between the first element
                                          of the first array and the
                                          first element of the second array */
  int istride = 1; /* array is contiguous in memory */
  int *inembed = NULL;

  // out
  int odist = n0 * (n1 / 2 + 1);
  int ostride = 1;
  int *onembed = NULL;

  return reinterpret_cast<void *>(fftw_plan_many_dft_r2c(rank, n, how_many, in, inembed, istride, idist,
                                                          reinterpret_cast<fftw_complex *>(out), onembed, ostride,
                                                          odist, flags));
}

template<>
void *fft_plan_many_dft_c2r_2d<float>(int n0,
                                      int n1,
                                      int how_many,
                                      std::complex<float> *in,
                                      float *out,
                                      unsigned flags) {
  int rank = 2;
  int n[] = {n0, n1};
  int idist = n0 * (n1 / 2 + 1); /* = 256*129, the distance in memory
                                          between the first element
                                          of the first array and the
                                          first element of the second array */
  int istride = 1; /* array is contiguous in memory */
  int *inembed = NULL;

  // out
  int odist = n0 * n1;
  int ostride = 1;
  int *onembed = NULL;

  return reinterpret_cast<void *>(fftwf_plan_many_dft_c2r(rank,
                                                          n,
                                                          how_many,
                                                          reinterpret_cast<fftwf_complex *>(in),
                                                          inembed,
                                                          istride,
                                                          idist,
                                                          out,
                                                          onembed,
                                                          ostride,
                                                          odist,
                                                          flags));
}

template<>
void *fft_plan_many_dft_c2r_2d<double>(int n0,
                                       int n1,
                                       int how_many,
                                       std::complex<double> *in,
                                       double *out,
                                       unsigned flags) {

  int rank = 2;
  int n[] = {n0, n1};
  int idist = n0 * (n1 / 2 + 1); /* = 256*129, the distance in memory
                                          between the first element
                                          of the first array and the
                                          first element of the second array */
  int istride = 1; /* array is contiguous in memory */
  int *inembed = NULL;

  // out
  int odist = n0 * n1;
  int ostride = 1;
  int *onembed = NULL;


  return reinterpret_cast<void *>(fftw_plan_many_dft_c2r(rank,
                                                          n,
                                                          how_many,
                                                          reinterpret_cast<fftw_complex *>(in),
                                                          inembed,
                                                          istride,
                                                          idist,
                                                          out,
                                                          onembed,
                                                          ostride,
                                                          odist,
                                                          flags));
}


template<>
void fft_execute_plan<float>(const void *plan_handle) {
  fftwf_execute((const fftwf_plan) plan_handle);
}

template<>
void fft_execute_plan<double>(const void *plan_handle) {
  fftw_execute((const fftw_plan) plan_handle);
}

template<>
void fft_destroy_plan<float>(const void *plan_handle) {
  fftwf_destroy_plan((const fftwf_plan) plan_handle);
}

template<>
void fft_destroy_plan<double>(const void *plan_handle) {
  fftw_destroy_plan((const fftw_plan) plan_handle);
}

bool check_power_of_2(unsigned int n) {
  // source see: http://graphics.stanford.edu/~seander/bithacks.html#DetermineIfPowerOf2
  // zero will be recognized as power of 2 (incorrectly.) But no input dim should be zero...
  return (n & (n - 1)) == 0;
}

unsigned int next_power_of_2(unsigned int n) {
  // source see: http://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2

  //compute the next highest power of 2 of 32-bit v
  n--;
  n |= n >> 1;
  n |= n >> 2;
  n |= n >> 4;
  n |= n >> 8;
  n |= n >> 16;
  n++;

  return n;
}

template<>
void fft_cpu_free<float>(void *ptr) {
  fftwf_free(ptr);
  ptr = NULL;
}

template<>
void fft_cpu_free<double>(void *ptr) {
  fftw_free(ptr);
  ptr = NULL;
}
}