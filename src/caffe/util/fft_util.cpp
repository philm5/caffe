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

    bool check_power_of_2(unsigned int n)
    {
        // source see: http://graphics.stanford.edu/~seander/bithacks.html#DetermineIfPowerOf2
        // zero will be recognized as power of 2 (incorrectly.) But no input dim should be zero...
        return (n & (n - 1)) == 0;
    }

    unsigned int next_power_of_2(unsigned int n)
    {
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
}