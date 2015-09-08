/* 
 * File:   fft_util.hpp
 * Author: Philipp Harzig <philipp at harzig.com>
 *
 * Created on July 16, 2015, 2:41 PM
 */

#ifndef FFT_UTIL_HPP
#define    FFT_UTIL_HPP

#include <fftw3.h>
#include <complex>
#include "caffe/util/mkl_alternate.hpp"

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
}


#endif	/* FFT_UTIL_HPP */

