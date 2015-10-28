/* 
 * File:   fft_util.hpp
 * Author: Philipp Harzig <philipp at harzig.com>
 *
 * Created on July 16, 2015, 2:41 PM
 */

#ifndef FFT_UTIL_HPP
#define FFT_UTIL_HPP

#include <fftw3.h>
#include <complex>
#include <vector>
#include "caffe/util/mkl_alternate.hpp"
#include "caffe/util/math_functions.hpp"
#include <ipps.h>

namespace caffe {
double cpu_time(void);


#ifdef _OPENMP
template<typename Dtype>
void fft_cpu_init_threads();

template<typename Dtype>
void fft_cpu_plan_with_nthreads(int n);

template<typename Dtype>
void fft_cpu_cleanup_threads();
#endif


template <typename Dtype>
void caffe_complex_mul(const int N, const std::complex<Dtype> *a, const std::complex<Dtype> *b, std::complex<Dtype> *y);

template <typename Dtype>
void caffe_complex_add(const int N, const std::complex<Dtype> *a, const std::complex<Dtype> *b, std::complex<Dtype> *y);

template <typename Dtype>
void ipp_complex_add_product(const std::complex<Dtype> *src1, const std::complex<Dtype> *src2, std::complex<Dtype> *dst, int len);

template<typename Dtype>
void *fft_cpu_malloc(size_t n);

template<typename Dtype>
void fft_cpu_free(void *ptr);

template<typename Dtype>
void *fft_cpu_plan_dft_r2c_2d(int n0, int n1, Dtype *in, std::complex<Dtype> *out, unsigned flags);

template<typename Dtype>
void *fft_cpu_plan_many_dft_r2c_2d(int n0, int n1, int how_many, Dtype *in, std::complex<Dtype> *out, unsigned flags);

template<typename Dtype>
void *fft_cpu_plan_many_dft_c2r_2d(int n0, int n1, int how_many, std::complex<Dtype> *in, Dtype *out, unsigned flags);

template<typename Dtype>
void fft_cpu_execute_plan(const void *plan_handle);

template<typename Dtype>
void fft_cpu_destroy_plan(const void *plan_handle);


/* Helper methods for fft */
bool check_power_of_2(unsigned int n);

unsigned int next_power_of_2(unsigned int n);


#ifndef CPU_ONLY

template <typename Dtype>
void npp_complex_add_product(const std::complex<Dtype> *src1, const std::complex<Dtype> *src2,
                             std::complex<Dtype> *dst, int len);

template<typename Dtype>
void fft_gpu_plan_many_dft_r2c_2d(cufftHandle *plan, int n0, int n1, int how_many);

template<typename Dtype>
void fft_gpu_plan_many_dft_c2r_2d(cufftHandle *plan, int n0, int n1, int how_many);

template<typename Dtype>
void fft_gpu_execute_plan_r2c(cufftHandle plan, Dtype *in, std::complex<Dtype> *out);

template<typename Dtype>
void fft_gpu_execute_plan_c2r(cufftHandle plan, std::complex<Dtype> *in, Dtype *out);

void fft_gpu_destroy_plan(cufftHandle plan_handle);


template<typename Dtype>
void pad_real_blob_gpu(std::vector<int> shape, const int fft_height, const int fft_width,
                       const Dtype *blob_data, Dtype *padded_data, const int pad_h,
                       const int pad_w, const bool flip);

template<typename Dtype>
void fft_util_pointwise_multiply_gpu(std::vector<int> shape, int group, const std::complex<Dtype> *ffted_bottom_data,
                                     const std::complex<Dtype> *weight_complex, std::complex<Dtype> *ptwise_result);

template<typename Dtype>
void fft_util_pointwise_multiply_npp_gpu(std::vector<int> shape, int group, const std::complex<Dtype> *ffted_bottom_data,
                                         const std::complex<Dtype> *weight_complex, std::complex<Dtype> *ptwise_result);

template <typename Dtype>
void fft_util_normalize_gpu(std::vector<int> shape, const int kernel_h,
                            const int kernel_w, const int stride_h, const int stride_w,
                            float normalize_factor, int fft_height, int fft_width,
                            const Dtype *conv_result_real, Dtype *top_data);
#endif

}
#endif	/* FFT_UTIL_HPP */

