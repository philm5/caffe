#include "caffe/util/fft_util.hpp"

namespace caffe 
{
    template<> 
    void *fft_cpu_malloc<float>(int n)
    {
        return reinterpret_cast<void *>(fftwf_malloc(n));
    }

    template<> 
    void *fft_cpu_malloc<double>(int n)
    {
        return reinterpret_cast<void *>(fftw_malloc(n));
    }
   
    template<>
    void *fft_plan_dft_r2c_2d<float>(int n0, int n1, float *in, std::complex<float> *out, unsigned flags)
    {
        return reinterpret_cast<void *>(fftwf_plan_dft_r2c_2d(n0, n1, in, reinterpret_cast<fftwf_complex *>(out), flags));
    }
   
    template<>
    void *fft_plan_dft_r2c_2d<double>(int n0, int n1, double *in, std::complex<double> *out, unsigned flags)
    {
        return reinterpret_cast<void *>(fftw_plan_dft_r2c_2d(n0, n1, in, reinterpret_cast<fftw_complex *>(out), flags));        
    }
    
    template<>
    void fft_execute_plan<float>(const void *plan_handle)
    {
        fftwf_execute((const fftwf_plan)plan_handle);
    }
    
    template<>
    void fft_execute_plan<double>(const void *plan_handle)
    {
        fftw_execute((const fftw_plan)plan_handle);
    }
}