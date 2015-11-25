#ifdef USE_FFT
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/fft_util.hpp"

// CUDA Header includes
#include <cuda.h>

namespace caffe {

// #define DBG_OUTPUT 1
//#define WRITE_TOP_RES

template<typename Dtype>
void ConvolutionLayerFFT<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                             const vector<Blob<Dtype>*>& top) {
  this->fft_on_ = true;

  if (this->fft_on_) {
    this->Forward_gpu_fft(bottom, top);
  } else {
    this->Forward_gpu_normal(bottom, top);
  }
}

template <typename Dtype>
void ConvolutionLayerFFT<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                              const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  this->fft_on_ = true;

  const Dtype* weight = this->blobs_[0]->gpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->gpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_gpu_bias(bias_diff, top_diff + top[i]->offset(n));
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      const Dtype* bottom_data = bottom[i]->gpu_data();
      Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
      if (!this->fft_on_) {
        for (int n = 0; n < this->num_; ++n) {
          // gradient w.r.t. weight. Note that we will accumulate diffs.
          if (this->param_propagate_down_[0]) {
            this->weight_gpu_gemm(bottom_data + bottom[i]->offset(n),
                                  top_diff + top[i]->offset(n), weight_diff);
          }
          // gradient w.r.t. bottom data, if necessary.
          if (propagate_down[i]) {
            this->backward_gpu_gemm(top_diff + top[i]->offset(n), weight,
                                    bottom_diff + bottom[i]->offset(n));
          }
        }
      } else {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->Weight_gpu_fft(bottom_data + bottom[i]->offset(0),
                               top_diff + top[i]->offset(0), weight_diff);
        }

        this->fft_update_weights_gpu();

        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->Backward_gpu_fft(top_diff + top[i]->offset(0),
                                 bottom_diff + bottom[i]->offset(0));
        }
      }
    }
  }
}

template <typename Dtype>
void ConvolutionLayerFFT<Dtype>::Weight_gpu_fft(const Dtype* input, const Dtype* output, Dtype* weight) {
  // fft the bottom data...
  this->fft_bottom_gpu(input);

  // fft the top diff data...
  this->fft_top_gpu(output);

  this->fft_convolve_weight_gpu(weight);
}

template <typename Dtype>
void ConvolutionLayerFFT<Dtype>::Backward_gpu_fft(const Dtype* top_blob, Dtype* bottom_blob) {
  // fft the top diff data...
  this->fft_top_gpu(top_blob);

  this->fft_convolve_backward_gpu(bottom_blob);
}


template<typename Dtype>
void ConvolutionLayerFFT<Dtype>::Forward_gpu_fft(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  this->fft_set_up();

  // the unit tests modify weights between twno Forward ops by step_size to check if backward calculates the gradient correctly.
  // But since the weights are only ffted once to save compute power, changes arent reflected in the complex values (ffted ones).
  // If fft_update_weights_each_batch_ mode is on, the weights are ffted every pass!!! Costs extra computing effort if done.
  if (this->fft_update_weights_each_batch_) {
    this->fft_update_weights_gpu();
  }

  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* top_data = top[i]->mutable_gpu_data();

    this->Forward_gpu_fft_single(bottom_data + bottom[i]->offset(0),
                                 top_data + top[i]->offset(0));

    for (int n = 0; n < this->num_; ++n) {
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->gpu_data();
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
  this->fft_bottom_gpu(bottom);

  this->fft_convolve_gpu(top);
}

template<typename Dtype>
void ConvolutionLayerFFT<Dtype>::fft_set_up_gpu() {
  CUDA_CHECK(cudaMalloc(&this->ffted_weights_, this->padded_weights_complex_size_));

  fft_gpu_plan_many_dft_r2c_2d<Dtype>(&fft_weight_plan_, this->fft_height_, this->fft_width_, this->num_weights_);

  // init the weights...
  this->fft_update_weights_gpu();

  // Set-up bottom plan once
  // Create FFT plan for the bottom data
  fft_gpu_plan_many_dft_r2c_2d<Dtype>(&fft_bottom_plan_gpu_, this->fft_height_, this->fft_width_, this->channels_ * this->num_);
  // Create FFT plan for the top data
  fft_gpu_plan_many_dft_r2c_2d<Dtype>(&fft_top_plan_gpu_, this->fft_height_, this->fft_width_, this->num_output_ * this->num_);

  // Allocate the real and complex memory for the bottom data
  CUDA_CHECK(cudaMalloc(&this->padded_real_bottom_gpu_, this->padded_bottom_real_size_));
  CUDA_CHECK(cudaMalloc(&this->ffted_bottom_data_gpu_, this->padded_bottom_complex_size_));

  CUDA_CHECK(cudaMalloc(&this->fft_convolution_result_real_gpu_, this->convolution_result_real_size_));
  CUDA_CHECK(cudaMalloc(&this->ptwise_result_gpu_, this->convolution_result_complex_size_));

  // Init other plans
  fft_gpu_plan_many_dft_c2r_2d<Dtype>(&ifft_plan_gpu_, this->fft_height_, this->fft_width_, this->num_output_ * this->num_);
  fft_gpu_plan_many_dft_c2r_2d<Dtype>(&ifft_backward_plan_gpu_, this->fft_height_, this->fft_width_, this->channels_ * this->num_);
  fft_gpu_plan_many_dft_c2r_2d<Dtype>(&ifft_weight_plan_gpu_, this->fft_height_, this->fft_width_, this->num_weights_);
}

template<typename Dtype>
void ConvolutionLayerFFT<Dtype>::fft_update_weights_gpu() {
  Dtype *padded_real_weights_gpu;

  CUDA_CHECK(cudaMalloc(&padded_real_weights_gpu, this->padded_weights_real_size_));
  const Dtype *weight_data = this->blobs_[0]->gpu_data();
  vector<int> shape;
  shape.push_back(this->num_output_);
  shape.push_back((this->channels_ / this->group_));
  shape.push_back(this->kernel_shape_.cpu_data()[0]);
  shape.push_back(this->kernel_shape_.cpu_data()[1]);

  // weights do not have to be padded (only 0-padded). But the weights have to be flipped, since the convolution is actually a
  // cross-correlation.
  pad_real_blob_gpu<Dtype>(shape, this->fft_height_, this->fft_width_, weight_data, padded_real_weights_gpu,
                           0, 0, 1, 1);

  fft_gpu_execute_plan_r2c<Dtype>(fft_weight_plan_, padded_real_weights_gpu, this->ffted_weights_);

  // free the padded real data... (no more need for it)
  CUDA_CHECK(cudaFree(padded_real_weights_gpu));

  // transpose weights if cgemm pt-wise product should be done
#if FFT_CONVOLUTION_KIND == FFT_CONVOLUTION_KIND_CGEMM
  std::complex<Dtype> *transposed_weights;
  CUDA_CHECK(cudaMalloc(&transposed_weights, this->padded_weights_complex_size_));

  const int weight_shape[] = { this->num_output_, this->channels_
      / this->group_, this->fft_height_, (this->fft_width_ / 2) + 1 };

  fft_util_geam_transpose_gpu<Dtype>(this->ffted_weights_, transposed_weights, weight_shape, 2);
  CUDA_CHECK(cudaFree(this->ffted_weights_));
  this->ffted_weights_ = transposed_weights;
#endif
}

template<typename Dtype>
void ConvolutionLayerFFT<Dtype>::fft_free_weights_gpu() {
  if (this->fft_initialized_) {
    CUDA_CHECK(cudaFree(this->padded_real_bottom_gpu_));
    CUDA_CHECK(cudaFree(this->ffted_bottom_data_gpu_));
    CUDA_CHECK(cudaFree(this->ffted_weights_));
    CUDA_CHECK(cudaFree(this->ptwise_result_gpu_));
    CUDA_CHECK(cudaFree(this->fft_convolution_result_real_gpu_));

    fft_gpu_destroy_plan(this->fft_weight_plan_);
    fft_gpu_destroy_plan(this->fft_bottom_plan_gpu_);
    fft_gpu_destroy_plan(this->fft_top_plan_gpu_);
    fft_gpu_destroy_plan(this->ifft_plan_gpu_);
    fft_gpu_destroy_plan(this->ifft_backward_plan_gpu_);
    fft_gpu_destroy_plan(this->ifft_weight_plan_gpu_);

    this->fft_initialized_ = false;
  }
}

template<typename Dtype>
void ConvolutionLayerFFT<Dtype>::fft_bottom_gpu(const Dtype *bottom) {
#ifdef DBG_OUTPUT
  float time;
  cudaEvent_t start, pad_bottom, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&pad_bottom);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
#endif
  const Dtype *bottom_blob = bottom;

  pad_real_blob_gpu(*this->bottom_shape_, this->fft_height_, this->fft_width_, bottom_blob, this->padded_real_bottom_gpu_, this->pad_.cpu_data()[0], this->pad_.cpu_data()[1], 1, 1);

#ifdef DBG_OUTPUT
  cudaEventRecord(pad_bottom, 0);
  cudaEventSynchronize(pad_bottom);
#endif

  // Execute fft bottom plan
  fft_gpu_execute_plan_r2c<Dtype>(this->fft_bottom_plan_gpu_, this->padded_real_bottom_gpu_, this->ffted_bottom_data_gpu_);

#if FFT_CONVOLUTION_KIND == FFT_CONVOLUTION_KIND_CGEMM
  // transpose input if cgemm pt-wise product should be done
  std::complex<Dtype> *fft_transposed_bottom;
  CUDA_CHECK(cudaMalloc(&fft_transposed_bottom, this->padded_bottom_complex_size_));
  const int shape_bottom[] = {this->num_, this->channels_, this->fft_height_, (this->fft_width_ / 2) + 1};
  // const int permutation_bottom[] = {2, 3, 0, 1};
  fft_util_geam_transpose_gpu(this->ffted_bottom_data_gpu_, fft_transposed_bottom, shape_bottom, 2);
  CUDA_CHECK(cudaFree(this->ffted_bottom_data_gpu_));
  this->ffted_bottom_data_gpu_ = fft_transposed_bottom;
#endif

#ifdef DBG_OUTPUT
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, pad_bottom);
  LOG(INFO) << this->layer_param_.name() << "| fft_bottom_gpu | padding of bottom: " << time << "ms..";
  cudaEventElapsedTime(&time, pad_bottom, stop);
  LOG(INFO) << this->layer_param_.name() << "| fft_bottom_gpu | fft of bottom (with permute4d if cgemm): " << time << "ms.";

  cudaEventDestroy(start);
  cudaEventDestroy(pad_bottom);
  cudaEventDestroy(stop);
#endif
}


template<typename Dtype>
void ConvolutionLayerFFT<Dtype>::fft_top_gpu(const Dtype *top) {
  const Dtype *top_blob = top;

  pad_real_blob_gpu(this->top_shape_, this->fft_height_, this->fft_width_, top_blob, this->fft_convolution_result_real_gpu_, 0, 0,
                    this->stride_.cpu_data()[0], this->stride_.cpu_data()[1]);

  // Execute fft bottom plan
  fft_gpu_execute_plan_r2c<Dtype>(this->fft_top_plan_gpu_, this->fft_convolution_result_real_gpu_, this->ptwise_result_gpu_);

#if FFT_CONVOLUTION_KIND == FFT_CONVOLUTION_KIND_CGEMM
  // transpose input if cgemm pt-wise product should be done
  std::complex<Dtype> *fft_transposed_top;
  CUDA_CHECK(cudaMalloc(&fft_transposed_top, this->padded_top_complex_size_));
  const int shape_top[] = {this->num_, this->num_output_, this->fft_height_, (this->fft_width_ / 2) + 1};
  // const int permutation_bottom[] = {2, 3, 0, 1};
  fft_util_geam_transpose_gpu(this->ptwise_result_gpu_, fft_transposed_top, shape_top, 2);
  CUDA_CHECK(cudaFree(this->ptwise_result_gpu_));
  this->ptwise_result_gpu_ = fft_transposed_top;
#endif
}

template<typename Dtype>
void ConvolutionLayerFFT<Dtype>::fft_convolve_gpu(Dtype *top) {
  // alloc data for pointwise multiplication result (with channels added up) and set memory to 0
  caffe_gpu_memset(this->convolution_result_complex_size_, 0., this->ptwise_result_gpu_);

#if FFT_CONVOLUTION_KIND == FFT_CONVOLUTION_KIND_POINTWISE_SIMPLE
  this->fft_pointwise_multiply_gpu();
#elif FFT_CONVOLUTION_KIND == FFT_CONVOLUTION_KIND_POINTWISE_IPP
  this->fft_pointwise_multiply_gpu();
#elif FFT_CONVOLUTION_KIND == FFT_CONVOLUTION_KIND_CGEMM
  this->fft_pointwise_multiply_gemm_gpu();
#endif

  this->fft_normalize_gpu(top);
}

template<typename Dtype>
void ConvolutionLayerFFT<Dtype>::fft_convolve_backward_gpu(Dtype *bottom) {
  // alloc data for pointwise multiplication result (with channels added up) and set memory to 0
  caffe_gpu_memset(this->padded_bottom_complex_size_, 0., this->ffted_bottom_data_gpu_);
#if FFT_CONVOLUTION_KIND == FFT_CONVOLUTION_KIND_POINTWISE_SIMPLE
  this->fft_pointwise_multiply_backward_gpu();
#elif FFT_CONVOLUTION_KIND == FFT_CONVOLUTION_KIND_POINTWISE_IPP
  this->fft_pointwise_multiply_backward_gpu();
#elif FFT_CONVOLUTION_KIND == FFT_CONVOLUTION_KIND_CGEMM
  this->fft_pointwise_multiply_gemm_backward_gpu();
#endif

  this->fft_normalize_backward_gpu(bottom);
}

template<typename Dtype>
void ConvolutionLayerFFT<Dtype>::fft_convolve_weight_gpu(Dtype *weight) {
  // alloc data for pointwise multiplication result (with channels added up) and set memory to 0
  caffe_gpu_memset(this->padded_weights_complex_size_, 0., this->ffted_weights_);

#if FFT_CONVOLUTION_KIND == FFT_CONVOLUTION_KIND_POINTWISE_SIMPLE
  this->fft_pointwise_multiply_weight_gpu();
#elif FFT_CONVOLUTION_KIND == FFT_CONVOLUTION_KIND_POINTWISE_IPP
  this->fft_pointwise_multiply_weight_gpu();
#elif FFT_CONVOLUTION_KIND == FFT_CONVOLUTION_KIND_CGEMM
  this->fft_pointwise_multiply_gemm_weight_gpu();
#endif

  this->fft_normalize_weight_gpu(weight);
}

template<typename Dtype>
void ConvolutionLayerFFT<Dtype>::fft_pointwise_multiply_gpu() {
  vector<int> shape;
  shape.push_back(this->num_);                        //              10 (batch size)
  shape.push_back(this->num_output_);                 //              256
  shape.push_back((this->channels_ / this->group_));  // 96 / 2     = 48
  shape.push_back(this->fft_height_);                 //              32
  shape.push_back((this->fft_width_ / 2) + 1);        // 32 / 2 + 1 = 17

  fft_util_pointwise_multiply_gpu<Dtype>(shape, this->group_, this->ffted_bottom_data_gpu_, this->ffted_weights_, this->ptwise_result_gpu_);
}

template<typename Dtype>
void ConvolutionLayerFFT<Dtype>::fft_pointwise_multiply_backward_gpu() {
  vector<int> shape;
  shape.push_back(this->num_);                        //              10 (batch size)
  shape.push_back(this->num_output_);                 //              256
  shape.push_back((this->channels_ / this->group_));  // 96 / 2     = 48
  shape.push_back(this->fft_height_);                 //              32
  shape.push_back((this->fft_width_ / 2) + 1);        // 32 / 2 + 1 = 17

  fft_util_pointwise_multiply_backward_gpu<Dtype>(shape, this->group_, this->ffted_bottom_data_gpu_, this->ffted_weights_, this->ptwise_result_gpu_);
}

template<typename Dtype>
void ConvolutionLayerFFT<Dtype>::fft_pointwise_multiply_weight_gpu() {
  vector<int> shape;
  shape.push_back(this->num_);                        //              10 (batch size)
  shape.push_back(this->num_output_);                 //              256
  shape.push_back((this->channels_ / this->group_));  // 96 / 2     = 48
  shape.push_back(this->fft_height_);                 //              32
  shape.push_back((this->fft_width_ / 2) + 1);        // 32 / 2 + 1 = 17

  fft_util_pointwise_multiply_weight_gpu<Dtype>(shape, this->group_, this->ffted_bottom_data_gpu_, this->ffted_weights_, this->ptwise_result_gpu_);
}

template<typename Dtype>
void ConvolutionLayerFFT<Dtype>::fft_pointwise_multiply_gemm_gpu() {
  cgemm_sizes sizes = this->fft_pointwise_multiply_gemm_init_cpu(FORWARD);

  fft_util_pointwise_multiply_gemm_gpu<Dtype>(sizes, this->ffted_bottom_data_gpu_,
                                              this->ffted_weights_, this->ptwise_result_gpu_);
}


template<typename Dtype>
void ConvolutionLayerFFT<Dtype>::fft_pointwise_multiply_gemm_backward_gpu() {
  cgemm_sizes sizes = this->fft_pointwise_multiply_gemm_init_cpu(BACKWARD);

  fft_util_pointwise_multiply_gemm_backward_gpu<Dtype>(sizes, this->ffted_bottom_data_gpu_,
                                                       this->ffted_weights_, this->ptwise_result_gpu_);
}

template<typename Dtype>
void ConvolutionLayerFFT<Dtype>::fft_pointwise_multiply_gemm_weight_gpu() {
  cgemm_sizes sizes = this->fft_pointwise_multiply_gemm_init_cpu(WEIGHT);

  fft_util_pointwise_multiply_gemm_weight_gpu<Dtype>(sizes, this->ffted_bottom_data_gpu_,
                                                     this->ffted_weights_, this->ptwise_result_gpu_);
}

template<typename Dtype>
void ConvolutionLayerFFT<Dtype>::fft_normalize_gpu(Dtype *top_data) {
  // caffe_gpu_memset(this->convolution_result_real_size_, 0., this->fft_convolution_result_real_gpu_);
  fft_gpu_execute_plan_c2r(this->ifft_plan_gpu_, this->ptwise_result_gpu_, this->fft_convolution_result_real_gpu_);

  Dtype ifft_normalize_factor = 1. / (this->fft_width_ * this->fft_height_);

  vector<int> shape;
  shape.push_back(this->num_);
  shape.push_back(this->num_output_);
  shape.push_back(this->output_shape_[0]);
  shape.push_back(this->output_shape_[1]);

  fft_util_normalize_gpu(shape, this->fft_height_, this->fft_width_, this->stride_.cpu_data()[0], this->stride_.cpu_data()[1], 0, 0,
                         ifft_normalize_factor, this->fft_convolution_result_real_gpu_, top_data, false);
}

template<typename Dtype>
void ConvolutionLayerFFT<Dtype>::fft_normalize_backward_gpu(Dtype *bottom) {
  fft_gpu_execute_plan_c2r(this->ifft_backward_plan_gpu_, this->ffted_bottom_data_gpu_, this->padded_real_bottom_gpu_);

  Dtype ifft_normalize_factor = 1. / (this->fft_width_ * this->fft_height_);

  vector<int> shape;
  shape.push_back(this->num_);
  shape.push_back(this->channels_);
  shape.push_back(this->input_shape(1));
  shape.push_back(this->input_shape(2));

  fft_util_normalize_gpu(shape, this->fft_height_, this->fft_width_, 1, 1, this->pad_.cpu_data()[0], this->pad_.cpu_data()[1], ifft_normalize_factor,
                         this->padded_real_bottom_gpu_, bottom, false);
}

template<typename Dtype>
void ConvolutionLayerFFT<Dtype>::fft_normalize_weight_gpu(Dtype *weight) {
  Dtype *padded_real_weights_gpu;
  CUDA_CHECK(cudaMalloc(&padded_real_weights_gpu, this->padded_weights_real_size_));

  fft_gpu_execute_plan_c2r(this->ifft_weight_plan_gpu_, this->ffted_weights_, padded_real_weights_gpu);

  Dtype ifft_normalize_factor = 1. / (this->fft_width_ * this->fft_height_);

  vector<int> shape;
  shape.push_back(this->num_output_);
  shape.push_back(this->channels_ / this->group_);
  shape.push_back(this->kernel_shape_.cpu_data()[0]);
  shape.push_back(this->kernel_shape_.cpu_data()[1]);


  fft_util_normalize_gpu(shape, this->fft_height_, this->fft_width_, 1, 1, 0, 0, ifft_normalize_factor,
                         padded_real_weights_gpu, weight, true);
  CUDA_CHECK(cudaFree(padded_real_weights_gpu));
}

template<typename Dtype>
void ConvolutionLayerFFT<Dtype>::mem_info_gpu() {
  size_t totalGlobalMem;
  size_t freeMem;

  cuMemGetInfo(&freeMem,&totalGlobalMem);
  printf("Free memory/Total memory:     %4.2f MB/%4.2f MB\n", (float)freeMem/(1024*1024), (float)totalGlobalMem/(1024*1024));
}

INSTANTIATE_LAYER_GPU_FUNCS(ConvolutionLayerFFT);

// float instantiation

template
void ConvolutionLayerFFT<float>::Backward_gpu_fft(const float* input, float* output);

template
void ConvolutionLayerFFT<float>::Weight_gpu_fft(const float* input, const float* output, float* weight);

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
void ConvolutionLayerFFT<float>::fft_update_weights_gpu();

template
void ConvolutionLayerFFT<float>::fft_free_weights_gpu();

template
void ConvolutionLayerFFT<float>::fft_bottom_gpu(const float *bottom);

template
void ConvolutionLayerFFT<float>::fft_top_gpu(const float *top);

template
void ConvolutionLayerFFT<float>::fft_convolve_gpu(float *top);

template
void ConvolutionLayerFFT<float>::fft_convolve_backward_gpu(float *bottom);

template
void ConvolutionLayerFFT<float>::fft_convolve_weight_gpu(float *weight);

template
void ConvolutionLayerFFT<float>::fft_pointwise_multiply_gpu();

template
void ConvolutionLayerFFT<float>::fft_pointwise_multiply_backward_gpu();

template
void ConvolutionLayerFFT<float>::fft_pointwise_multiply_weight_gpu();

template
void ConvolutionLayerFFT<float>::fft_pointwise_multiply_gemm_gpu();

template
void ConvolutionLayerFFT<float>::fft_pointwise_multiply_gemm_backward_gpu();

template
void ConvolutionLayerFFT<float>::fft_pointwise_multiply_gemm_weight_gpu();

template
void ConvolutionLayerFFT<float>::fft_normalize_gpu(float *top_data);

template
void ConvolutionLayerFFT<float>::fft_normalize_backward_gpu(float *bottom);

template
void ConvolutionLayerFFT<float>::fft_normalize_weight_gpu(float *weight);

template
void ConvolutionLayerFFT<float>::mem_info_gpu();

// double instantiation


template
void ConvolutionLayerFFT<double>::Backward_gpu_fft(const double* input, double* output);

template
void ConvolutionLayerFFT<double>::Weight_gpu_fft(const double* input, const double* output, double* weight);

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
void ConvolutionLayerFFT<double>::fft_update_weights_gpu();

template
void ConvolutionLayerFFT<double>::fft_free_weights_gpu();

template
void ConvolutionLayerFFT<double>::fft_bottom_gpu(const double *bottom);

template
void ConvolutionLayerFFT<double>::fft_top_gpu(const double *top);

template
void ConvolutionLayerFFT<double>::fft_convolve_gpu(double *top);

template
void ConvolutionLayerFFT<double>::fft_convolve_backward_gpu(double *bottom);

template
void ConvolutionLayerFFT<double>::fft_convolve_weight_gpu(double *weight);

template
void ConvolutionLayerFFT<double>::fft_pointwise_multiply_gpu();

template
void ConvolutionLayerFFT<double>::fft_pointwise_multiply_backward_gpu();

template
void ConvolutionLayerFFT<double>::fft_pointwise_multiply_weight_gpu();

template
void ConvolutionLayerFFT<double>::fft_pointwise_multiply_gemm_gpu();

template
void ConvolutionLayerFFT<double>::fft_pointwise_multiply_gemm_backward_gpu();

template
void ConvolutionLayerFFT<double>::fft_pointwise_multiply_gemm_weight_gpu();

template
void ConvolutionLayerFFT<double>::fft_normalize_gpu(double *top_data);

template
void ConvolutionLayerFFT<double>::fft_normalize_backward_gpu(double *bottom);

template
void ConvolutionLayerFFT<double>::fft_normalize_weight_gpu(double *weight);

template
void ConvolutionLayerFFT<double>::mem_info_gpu();


}  // namespace caffe
#endif /* USE_FFT */
