/* 
 * File:   fft_util.hpp
 * Author: Philipp Harzig <philipp at harzig.com>
 *
 * Created on July 16, 2015, 2:41 PM
 */

#ifndef FFT_UTIL_HPP
#define	FFT_UTIL_HPP

#include <fftw3.h>
#include <complex>

namespace caffe
{
    template<typename Dtype> 
    void *fft_cpu_malloc(int n);
    
    template<typename Dtype>
    void *fft_plan_dft_r2c_2d(int n0, int n1, Dtype *in, std::complex<Dtype> *out, unsigned flags);
    
    template<typename Dtype>
    void fft_execute_plan(const void *plan_handle);
}


#endif	/* FFT_UTIL_HPP */

