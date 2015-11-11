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

template<typename Dtype>
void ConvolutionLayerFFT<Dtype>::Forward_gpu_fft(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
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

#ifdef DBG_OUTPUT
  float time;
  cudaEvent_t start, fft_set_up, alloc_bottom, fft_bottom, convolve;
  cudaEventCreate(&start);
  cudaEventCreate(&fft_set_up);
  cudaEventCreate(&alloc_bottom);
  cudaEventCreate(&fft_bottom);
  cudaEventCreate(&convolve);
  cudaEventRecord(start, 0);
#endif
  this->fft_set_up();
#ifdef DBG_OUTPUT
  cudaEventRecord(fft_set_up, 0);
  cudaEventSynchronize(fft_set_up);
#endif

    // caffe_gpu_memset(this->padded_bottom_complex_size_, 0., this->ffted_bottom_data_gpu_);

#ifdef DBG_OUTPUT
    cudaEventRecord(alloc_bottom, 0);
    cudaEventSynchronize(alloc_bottom);
#endif
    this->fft_bottom_gpu(bottom);

#ifdef DBG_OUTPUT
    cudaEventRecord(fft_bottom, 0);
    cudaEventSynchronize(fft_bottom);
#endif
    this->fft_convolve_gpu(top);
#ifdef DBG_OUTPUT
    cudaEventRecord(convolve, 0);
    cudaEventSynchronize(convolve);
#endif

#ifdef DBG_OUTPUT
    cudaEventElapsedTime(&time, start, fft_set_up);
    LOG(INFO) << this->layer_param_.name() << "| fft_set_up: " << time << "ms..";
    cudaEventElapsedTime(&time, fft_set_up, alloc_bottom);
    LOG(INFO) << this->layer_param_.name() << "| caffe_memset bottom: " << time << "ms.";
    cudaEventElapsedTime(&time, alloc_bottom, fft_bottom);
    LOG(INFO) << this->layer_param_.name() << "| fft_bottom_gpu: " << time << "ms.";
    cudaEventElapsedTime(&time, fft_bottom, convolve);
    LOG(INFO) << this->layer_param_.name() << "| fft_convolve_gpu: " << time << "ms.";
    cudaEventElapsedTime(&time, start, convolve);
    LOG(INFO) << this->layer_param_.name() << "| total pass gpu: " << time << "ms.";

    cudaEventDestroy(start);
    cudaEventDestroy(fft_set_up);
    cudaEventDestroy(alloc_bottom);
    cudaEventDestroy(fft_bottom);
    cudaEventDestroy(convolve);
#endif

}

template<typename Dtype>
void ConvolutionLayerFFT<Dtype>::fft_set_up_gpu() {
//  this->mem_info_gpu();

  Dtype *padded_real_weights_gpu;

  CUDA_CHECK(cudaMalloc(&padded_real_weights_gpu, this->padded_weights_real_size_));
  const Dtype *weight_data = this->blobs_[0]->gpu_data();
	vector<int> shape;
	shape.push_back(this->num_output_);
	shape.push_back((this->channels_ / this->group_));
	shape.push_back(this->kernel_h_);
	shape.push_back(this->kernel_w_);

  // weights do not have to be padded (only 0-padded). But the weights have to be flipped, since the convolution is actually a
  // cross-correlation.
	pad_real_blob_gpu<Dtype>(shape, this->fft_height_, this->fft_width_, weight_data, padded_real_weights_gpu,
	                         0, 0, true);

//	this->mem_info_gpu();
	CUDA_CHECK(cudaMalloc(&this->ffted_weights_, this->padded_weights_complex_size_));

	cufftHandle plan;
	fft_gpu_plan_many_dft_r2c_2d<Dtype>(&plan, this->fft_height_, this->fft_width_, this->num_weights_);
	// caffe_gpu_memset(this->padded_weights_complex_size_, 0., this->ffted_weights_);

	fft_gpu_execute_plan_r2c<Dtype>(plan, padded_real_weights_gpu, this->ffted_weights_);
//	this->mem_info_gpu();

	// destroy the plan
	fft_gpu_destroy_plan(plan);

	// free the padded real data... (no more need for it)
	CUDA_CHECK(cudaFree(padded_real_weights_gpu));

	// Set-up bottom plan once
  // Create FFT plan for the bottom data and alloc memory
  fft_gpu_plan_many_dft_r2c_2d<Dtype>(&fft_bottom_plan_gpu_, this->fft_height_, this->fft_width_, this->channels_ * this->num_);

  // Allocate the real and complex memory for the bottom data

  CUDA_CHECK(cudaMalloc(&this->padded_real_bottom_gpu_, this->padded_bottom_real_size_));
  CUDA_CHECK(cudaMalloc(&this->ffted_bottom_data_gpu_, this->padded_bottom_complex_size_));

  CUDA_CHECK(cudaMalloc(&this->fft_convolution_result_real_gpu_, this->convolution_result_real_size_));
  CUDA_CHECK(cudaMalloc(&this->ptwise_result_gpu_, this->convolution_result_complex_size_));


  fft_gpu_plan_many_dft_c2r_2d<Dtype>(&ifft_plan_gpu_, this->fft_height_, this->fft_width_, this->num_output_ * this->num_);

//  this->mem_info_gpu();

//
#if FFT_CONVOLUTION_KIND == FFT_CONVOLUTION_KIND_CGEMM
  // transpose weights if cgemm pt-wise product should be done
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
  CUDA_CHECK(cudaFree(this->ffted_weights_));
  fft_gpu_destroy_plan(this->fft_bottom_plan_gpu_);
  CUDA_CHECK(cudaFree(this->padded_real_bottom_gpu_));
  CUDA_CHECK(cudaFree(this->ffted_bottom_data_gpu_));
  CUDA_CHECK(cudaFree(this->ptwise_result_gpu_));
  CUDA_CHECK(cudaFree(this->fft_convolution_result_real_gpu_));

  fft_gpu_destroy_plan(this->ifft_plan_gpu_);
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

  pad_real_blob_gpu(this->bottom_shape_, this->fft_height_, this->fft_width_, bottom_blob, this->padded_real_bottom_gpu_, this->pad_h_, this->pad_w_, false);

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
void ConvolutionLayerFFT<Dtype>::fft_convolve_gpu(Dtype *top) {
#ifdef DBG_OUTPUT
  float time;
  cudaEvent_t start, memset, conv, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&memset);
  cudaEventCreate(&conv);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
#endif
  // alloc data for pointwise multiplication result (with channels added up) and set memory to 0
  // caffe_gpu_memset(this->convolution_result_complex_size_, 0., this->ptwise_result_gpu_);

#ifdef DBG_OUTPUT
  cudaEventRecord(memset, 0);
  cudaEventSynchronize(memset);
#endif

#if FFT_CONVOLUTION_KIND == FFT_CONVOLUTION_KIND_POINTWISE_SIMPLE
  this->fft_pointwise_multiply_gpu();
#elif FFT_CONVOLUTION_KIND == FFT_CONVOLUTION_KIND_POINTWISE_IPP
  this->fft_pointwise_multiply_npp_gpu();
#elif FFT_CONVOLUTION_KIND == FFT_CONVOLUTION_KIND_CGEMM
  this->fft_pointwise_multiply_gemm_gpu();
#endif

#ifdef DBG_OUTPUT
  cudaEventRecord(conv, 0);
  cudaEventSynchronize(conv);
#endif

//  std::complex<Dtype> *res = reinterpret_cast<std::complex<Dtype> *>(fft_cpu_malloc<Dtype>(this->convolution_result_complex_size_));
//  CUDA_CHECK(cudaMemcpy(res, this->ptwise_result_gpu_, this->convolution_result_complex_size_, cudaMemcpyDeviceToHost));
//  std::stringstream ss;
//  ss << "convolved_fft_" << this->layer_param_.name() << ".txt";
//  const char *s = ss.str().c_str();
//  this->write_arr_to_disk(s, this->num_output_, res, true);

  this->fft_normalize_gpu(top);

#ifdef DBG_OUTPUT
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, memset);
    LOG(INFO) << this->layer_param_.name() << "| fft_convolve_gpu | memset: " << time << "ms.";
    cudaEventElapsedTime(&time, memset, conv);
    LOG(INFO) << this->layer_param_.name() << "| fft_convolve_gpu | conv:" << time << "ms.";
    cudaEventElapsedTime(&time, conv, stop);
    LOG(INFO) << this->layer_param_.name() << "| fft_convolve_gpu | normalize:" << time << "ms.";

    cudaEventDestroy(start);
    cudaEventDestroy(memset);
    cudaEventDestroy(conv);
    cudaEventDestroy(stop);
#endif

//  std::stringstream ss;
//  ss << "convolved_fft_" << this->layer_param_.name() << ".txt";
//  const char *s = ss.str().c_str();
//  this->write_arr_to_disk(s, this->num_output_, res, true);
}

template<typename Dtype>
void ConvolutionLayerFFT<Dtype>::fft_pointwise_multiply_gpu() {
  vector<int> shape;
  shape.push_back(this->num_);                        //              10 (batch size)
  shape.push_back(this->num_output_);                 //              256
  shape.push_back((this->channels_ / this->group_));  // 96 / 2     = 48
  shape.push_back(this->fft_height_);                 //              32
  shape.push_back((this->fft_width_ / 2) + 1);        // 32 / 2 + 1 = 17

#ifdef DBG_OUTPUT
  float time;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
#endif
  fft_util_pointwise_multiply_gpu<Dtype>(shape, this->group_, this->ffted_bottom_data_gpu_, this->ffted_weights_, this->ptwise_result_gpu_);
#ifdef DBG_OUTPUT
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  LOG(INFO) << this->layer_param_.name() << "| fft_pointwise_multiply_gpu: " << time << "ms.";
#endif
}

template<typename Dtype>
void ConvolutionLayerFFT<Dtype>::fft_pointwise_multiply_npp_gpu() {
  vector<int> shape;
  shape.push_back(this->num_output_);                 //              256
  shape.push_back((this->channels_ / this->group_));  // 96 / 2     = 48
  shape.push_back(this->fft_height_);                 //              32
  shape.push_back((this->fft_width_ / 2) + 1);        // 32 / 2 + 1 = 17

#ifdef DBG_OUTPUT
  float time;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
#endif
  fft_util_pointwise_multiply_npp_gpu<Dtype>(shape, this->group_, this->ffted_bottom_data_gpu_,
                                             this->ffted_weights_, this->ptwise_result_gpu_);
#ifdef DBG_OUTPUT
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  LOG(INFO) << this->layer_param_.name() << "| fft_pointwise_multiply_npp_gpu: " << time << "ms.";
#endif
}

template<typename Dtype>
void ConvolutionLayerFFT<Dtype>::fft_pointwise_multiply_gemm_gpu() {
  vector<int> shape;
  shape.push_back(this->num_);                        //              10 (batch size)
  shape.push_back(this->num_output_);                 //              256
  shape.push_back((this->channels_ / this->group_));  // 96 / 2     = 48
  shape.push_back(this->fft_height_);                 //              32
  shape.push_back((this->fft_width_ / 2) + 1);        // 32 / 2 + 1 = 17

#ifdef DBG_OUTPUT
  float time;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
#endif
  fft_util_pointwise_multiply_gemm_gpu<Dtype>(shape, this->group_, this->ffted_bottom_data_gpu_,
                                              this->ffted_weights_, this->ptwise_result_gpu_);
#ifdef DBG_OUTPUT
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  LOG(INFO) << this->layer_param_.name() << "| fft_pointwise_multiply_gemm_gpu: " << time << "ms.";
#endif
}

template<typename Dtype>
void ConvolutionLayerFFT<Dtype>::fft_normalize_gpu(Dtype *top_data) {
  // caffe_gpu_memset(this->convolution_result_real_size_, 0., this->fft_convolution_result_real_gpu_);
  fft_gpu_execute_plan_c2r(this->ifft_plan_gpu_, this->ptwise_result_gpu_, this->fft_convolution_result_real_gpu_);

  Dtype ifft_normalize_factor = 1. / (this->fft_width_ * this->fft_height_);

  vector<int> shape;
  shape.push_back(this->num_);
  shape.push_back(this->num_output_);
  shape.push_back(this->height_out_);
  shape.push_back(this->width_out_);

  fft_util_normalize_gpu(shape, this->kernel_h_, this->kernel_w_, this->stride_h_, this->stride_w_, ifft_normalize_factor,
                         this->fft_height_, this->fft_width_, this->fft_convolution_result_real_gpu_, top_data);

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

// float instantiation

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
void ConvolutionLayerFFT<float>::fft_free_weights_gpu();

template
void ConvolutionLayerFFT<float>::fft_bottom_gpu(const float *bottom);

template
void ConvolutionLayerFFT<float>::fft_convolve_gpu(float *top);

template
void ConvolutionLayerFFT<float>::fft_pointwise_multiply_gpu();

template
void ConvolutionLayerFFT<float>::fft_pointwise_multiply_npp_gpu();

template
void ConvolutionLayerFFT<float>::fft_pointwise_multiply_gemm_gpu();

template
void ConvolutionLayerFFT<float>::fft_normalize_gpu(float *top_data);

template
void ConvolutionLayerFFT<float>::mem_info_gpu();

// double instantiation

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
void ConvolutionLayerFFT<double>::fft_free_weights_gpu();

template
void ConvolutionLayerFFT<double>::fft_bottom_gpu(const double *bottom);

template
void ConvolutionLayerFFT<double>::fft_convolve_gpu(double *top);

template
void ConvolutionLayerFFT<double>::fft_pointwise_multiply_gpu();

template
void ConvolutionLayerFFT<double>::fft_pointwise_multiply_npp_gpu();

template
void ConvolutionLayerFFT<double>::fft_pointwise_multiply_gemm_gpu();

template
void ConvolutionLayerFFT<double>::fft_normalize_gpu(double *top_data);

template
void ConvolutionLayerFFT<double>::mem_info_gpu();


}  // namespace caffe
