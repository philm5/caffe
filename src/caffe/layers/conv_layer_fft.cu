#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

// CUDA Header includes
#include <cuda.h>

namespace caffe {

template<typename Dtype>
void ConvolutionLayerFFT<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                             const vector<Blob<Dtype>*>& top) {
  if (this->layer_param().name() == "conv2") {
    this->fft_on_ = true;
  }

  if (this->fft_on_) {
    this->Forward_gpu_fft(bottom, top);
  } else {
    this->Forward_gpu_normal(bottom, top);
  }
}

template<typename Dtype>
void ConvolutionLayerFFT<Dtype>::Forward_gpu_fft(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* top_data = top[i]->mutable_gpu_data();

    for (int n = 0; n < this->num_; ++n) {
      this->Forward_gpu_fft_single(bottom_data + bottom[i]->offset(n),
                                   top_data + top[i]->offset(n));
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->cpu_data();
        this->forward_gpu_bias(top_data + top[i]->offset(n), bias);
      }
    }
  }
}

template<typename Dtype>
void ConvolutionLayerFFT<Dtype>::Forward_gpu_normal(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->gpu_data();
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* top_data = top[i]->mutable_gpu_data();
    for (int n = 0; n < this->num_; ++n) {
      this->forward_gpu_gemm(bottom_data + bottom[i]->offset(n), weight,
                             top_data + top[i]->offset(n));
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->gpu_data();
        this->forward_gpu_bias(top_data + top[i]->offset(n), bias);
      }
    }
  }

}

template<typename Dtype>
void ConvolutionLayerFFT<Dtype>::Forward_gpu_fft_single(const Dtype *bottom,
                                                        Dtype *top) {
  this->fft_set_up();

  // Allocate the complex memory for the bottom data
//  std::complex<Dtype> *ffted_bottom_data =
//      reinterpret_cast<std::complex<Dtype> *>(fft_cpu_malloc<Dtype>(
//          this->padded_bottom_complex_size_));
//  caffe_memset(this->padded_bottom_complex_size_, 0., ffted_bottom_data);

//  this->fft_bottom_cpu(bottom, ffted_bottom_data);
//  this->fft_convolve_cpu(ffted_bottom_data, top);

}

template<typename Dtype>
void ConvolutionLayerFFT<Dtype>::fft_set_up_gpu() {
//	  // Initialize OpenMP for the CPU.
//	  this->num_threads_ = 1;
//	#ifdef _OPENMP
//	  this->num_threads_ = omp_get_max_threads();
//	  if (this->num_threads_ < 1) {
//	    LOG(WARNING) << "FFT Convolution Layer: omp_get_max_threads() =" << this->num_threads_;
//	    this->num_threads_ = 1;
//	  }
//	  fft_cpu_init_threads<Dtype>();
//	  fft_cpu_plan_with_nthreads<Dtype>(this->num_threads_);
//	#endif

  this->mem_info_gpu();

  Dtype *padded_real_weights;

  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(padded_real_weights), this->padded_weights_real_size_));

  this->mem_info_gpu();

//  Dtype *padded_real_weights = reinterpret_cast<Dtype *>(fft_cpu_malloc<Dtype>(
//      this->padded_weights_real_size_));
//
//  // get the pointer to the weights
//  const Dtype *weight_data = this->blobs_[0]->cpu_data();
//  vector<int> shape;
//  shape.push_back(this->num_output_);
//  shape.push_back((this->channels_ / this->group_));
//  shape.push_back(this->kernel_h_);
//  shape.push_back(this->kernel_w_);
//
//  // weights do not have to be padded. But the weights have to be flipped, since the convolution is actually a
//  // cross-correlation.
//  this->pad_real_blob(shape, weight_data, padded_real_weights, 0, 0, true);
//
//  // The plan for fft of the weights
//  // TODO: Do inplace fft to save memory???
//  this->ffted_weights_ = reinterpret_cast<std::complex<Dtype> *>(fft_cpu_malloc<
//      Dtype>(this->padded_weights_complex_size_));
//
//  const void *fft_weight_plan = fft_cpu_plan_many_dft_r2c_2d<Dtype>(
//      this->fft_height_, this->fft_width_, this->num_weights_,
//      padded_real_weights, this->ffted_weights_, FFTW_ESTIMATE);
//
//  // set the complex data to 0.
//  caffe_memset(this->padded_weights_complex_size_, 0., this->ffted_weights_);
//
//  // Do the FFT of the padded weights:
//  fft_cpu_execute_plan<Dtype>(fft_weight_plan);
//
//  // Destroy the weight plan:
//  fft_cpu_destroy_plan<Dtype>(fft_weight_plan);
//  // free the padded real weights. There is no need for them anymore. Also free weights in blobs_[0] ???
//  fft_cpu_free<Dtype>(padded_real_weights);
//
//#if FFT_CONVOLUTION_KIND == FFT_CONVOLUTION_KIND_CGEMM
//  // transpose weights if cgemm pt-wise product should be done
//  std::complex<Dtype> *transposed_weights =
//      reinterpret_cast<std::complex<Dtype> *>(fft_cpu_malloc<Dtype>(
//          this->padded_weights_complex_size_));
//  const int weight_shape[] = { this->num_output_, this->channels_
//      / this->group_, this->fft_height_, (this->fft_width_ / 2) + 1 };
//  const int permutation[] = { 2, 3, 0, 1 };
//  this->fft_permute_4d_cpu(this->ffted_weights_, transposed_weights,
//                           weight_shape, permutation);
//  fft_cpu_free<Dtype>(this->ffted_weights_);
//  this->ffted_weights_ = transposed_weights;
//#endif

}

template<typename Dtype>
void ConvolutionLayerFFT<Dtype>::mem_info_gpu() {
  size_t totalGlobalMem;
  size_t freeMem;

  cuMemGetInfo(&freeMem,&totalGlobalMem);
  printf("Free memory/Total memory:     %4.2f MB/%4.2f MB\n", (float)freeMem/(1024*1024), (float)totalGlobalMem/(1024*1024));
}

template<typename Dtype>
void ConvolutionLayerFFT<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

}

INSTANTIATE_LAYER_GPU_FUNCS(ConvolutionLayerFFT);

// float instatiation

template
void ConvolutionLayerFFT<float>::Forward_gpu_fft(const vector<Blob<float>*>& bottom,
                                                 const vector<Blob<float>*>& top);

template
void ConvolutionLayerFFT<float>::Forward_gpu_normal(const vector<Blob<float>*>& bottom,
                                                    const vector<Blob<float>*>& top);

template
void ConvolutionLayerFFT<float>::Forward_gpu_fft_single(const float *bottom, float *top);

template
void ConvolutionLayerFFT<float>::fft_set_up_gpu();

template
void ConvolutionLayerFFT<float>::mem_info_gpu();

// double instatiation

template
void ConvolutionLayerFFT<double>::Forward_gpu_fft(const vector<Blob<double>*>& bottom,
                                                 const vector<Blob<double>*>& top);

template
void ConvolutionLayerFFT<double>::Forward_gpu_normal(const vector<Blob<double>*>& bottom,
                                                    const vector<Blob<double>*>& top);

template
void ConvolutionLayerFFT<double>::Forward_gpu_fft_single(const double *bottom, double *top);

template
void ConvolutionLayerFFT<double>::fft_set_up_gpu();

template
void ConvolutionLayerFFT<double>::mem_info_gpu();


}  // namespace caffe
