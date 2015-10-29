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

//#define DBG_OUTPUT 1
//#define WRITE_TOP_RES

template<typename Dtype>
void ConvolutionLayerFFT<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                             const vector<Blob<Dtype>*>& top) {
  if (this->layer_param().name() != "conv1") {
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
  printf("hallo!!!cu");
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* top_data = top[i]->mutable_gpu_data();

    for (int n = 0; n < this->num_; ++n) {
      this->Forward_gpu_fft_single(bottom_data + bottom[i]->offset(n),
                                   top_data + top[i]->offset(n));
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

    // Allocate the complex memory for the bottom data
    std::complex<Dtype> *ffted_bottom_data;
    CUDA_CHECK(cudaMalloc(&ffted_bottom_data, this->padded_bottom_complex_size_));
    caffe_gpu_memset(this->padded_bottom_complex_size_, 0., ffted_bottom_data);

#ifdef DBG_OUTPUT
    cudaEventRecord(alloc_bottom, 0);
    cudaEventSynchronize(alloc_bottom);
#endif
    this->fft_bottom_gpu(bottom, ffted_bottom_data);

#ifdef DBG_OUTPUT
    cudaEventRecord(fft_bottom, 0);
    cudaEventSynchronize(fft_bottom);
#endif
    this->fft_convolve_gpu(ffted_bottom_data, top);
#ifdef DBG_OUTPUT
    cudaEventRecord(convolve, 0);
    cudaEventSynchronize(convolve);
#endif

#ifdef DBG_OUTPUT
    cudaEventElapsedTime(&time, start, fft_set_up);
    LOG(INFO) << "fft_set_up: " << time << "ms..";
    cudaEventElapsedTime(&time, fft_set_up, alloc_bottom);
    LOG(INFO) << "caffe_memset bottom: " << time << "ms.";
    cudaEventElapsedTime(&time, alloc_bottom, fft_bottom);
    LOG(INFO) << "fft_bottom_cpu: " << time << "ms.";
    cudaEventElapsedTime(&time, fft_bottom, convolve);
    LOG(INFO) << "fft_convolve_cpu: " << time << "ms.";
    cudaEventElapsedTime(&time, start, convolve);
    LOG(INFO) << "total pass: " << time << "ms.";

    cudaEventDestroy(start);
    cudaEventDestroy(fft_set_up);
    cudaEventDestroy(alloc_bottom);
    cudaEventDestroy(fft_bottom);
    cudaEventDestroy(convolve);
#endif

}

template<typename Dtype>
void ConvolutionLayerFFT<Dtype>::fft_set_up_gpu() {
  this->mem_info_gpu();

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

	this->mem_info_gpu();
	CUDA_CHECK(cudaMalloc(&this->ffted_weights_, this->padded_weights_complex_size_));

	cufftHandle plan;
	fft_gpu_plan_many_dft_r2c_2d<Dtype>(&plan, this->fft_height_, this->fft_width_, this->num_weights_);
	caffe_gpu_memset(this->padded_weights_complex_size_, 0., this->ffted_weights_);

	fft_gpu_execute_plan_r2c<Dtype>(plan, padded_real_weights_gpu, this->ffted_weights_);
	this->mem_info_gpu();

	// destroy the plan
	fft_gpu_destroy_plan(plan);

	// free the padded real data... (no more need for it)
	CUDA_CHECK(cudaFree(padded_real_weights_gpu));
  this->mem_info_gpu();

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
void ConvolutionLayerFFT<Dtype>::fft_free_weights_gpu() {
  CUDA_CHECK(cudaFree(this->ffted_weights_));
}

template<typename Dtype>
void ConvolutionLayerFFT<Dtype>::fft_bottom_gpu(const Dtype *bottom, std::complex<Dtype> *&ffted_bottom_data) {
  const Dtype *bottom_blob = bottom;
  Dtype *padded_real_bottom;

  CUDA_CHECK(cudaMalloc(&padded_real_bottom, this->padded_bottom_real_size_));
  // now pad the bottom data (it should have its origin in (h,w)), but don't flip it.
  pad_real_blob_gpu(this->bottom_shape_, this->fft_height_, this->fft_width_, bottom_blob, padded_real_bottom, this->pad_h_, this->pad_w_, false);

  // Create FFT plan for the bottom data and execute it
  cufftHandle fft_bottom_plan;
  fft_gpu_plan_many_dft_r2c_2d<Dtype>(&fft_bottom_plan, this->fft_height_, fft_width_, this->channels_);
  fft_gpu_execute_plan_r2c<Dtype>(fft_bottom_plan, padded_real_bottom, ffted_bottom_data);
  fft_gpu_destroy_plan(fft_bottom_plan);
  CUDA_CHECK(cudaFree(padded_real_bottom));

//#if FFT_CONVOLUTION_KIND == FFT_CONVOLUTION_KIND_CGEMM
//  // transpose input if cgemm pt-wise product should be done
//  std::complex<Dtype> *fft_transposed_bottom =
//      reinterpret_cast<std::complex<Dtype> *>(fft_cpu_malloc<Dtype>(this->padded_bottom_complex_size_));
//  const int shape_bottom[] = {1, this->channels_, this->fft_height_, (this->fft_width_ / 2) + 1};
//  const int permutation_bottom[] = {2, 3, 0, 1};
//  this->fft_permute_4d_cpu(ffted_bottom_data, fft_transposed_bottom, shape_bottom, permutation_bottom);
//
//  fft_cpu_free<Dtype>(ffted_bottom_data);
//  ffted_bottom_data = fft_transposed_bottom;
//#endif
}

template<typename Dtype>
void ConvolutionLayerFFT<Dtype>::fft_convolve_gpu(std::complex<Dtype> *ffted_bottom_data, Dtype *top) {
  // alloc data for pointwise multiplication result (with channels added up) and set memory to 0
  std::complex<Dtype> *ptwise_result;
  CUDA_CHECK(cudaMalloc(&ptwise_result, this->convolution_result_complex_size_));
  caffe_gpu_memset(this->convolution_result_complex_size_, 0., ptwise_result);

#if FFT_CONVOLUTION_KIND == FFT_CONVOLUTION_KIND_POINTWISE_SIMPLE
  this->fft_pointwise_multiply_gpu(ffted_bottom_data, ptwise_result);
#elif FFT_CONVOLUTION_KIND == FFT_CONVOLUTION_KIND_POINTWISE_IPP
  this->fft_pointwise_multiply_npp_gpu(ffted_bottom_data, ptwise_result);
#elif FFT_CONVOLUTION_KIND == FFT_CONVOLUTION_KIND_CGEMM
  this->fft_pointwise_multiply_gemm_cpu(ffted_bottom_data, ptwise_result);
#endif

//  std::complex<Dtype> *res = reinterpret_cast<std::complex<Dtype> *>(fft_cpu_malloc<Dtype>(this->convolution_result_complex_size_));
//  CUDA_CHECK(cudaMemcpy(res, ptwise_result, this->convolution_result_complex_size_, cudaMemcpyDeviceToHost));
//  std::stringstream ss;
//  ss << "convolved_fft_" << this->layer_param_.name() << ".txt";
//  const char *s = ss.str().c_str();
//  this->write_arr_to_disk(s, this->num_output_, res, true);

  CUDA_CHECK(cudaFree(ffted_bottom_data));
  this->fft_normalize_gpu(ptwise_result, top);

//  std::stringstream ss;
//  ss << "convolved_fft_" << this->layer_param_.name() << ".txt";
//  const char *s = ss.str().c_str();
//  this->write_arr_to_disk(s, this->num_output_, res, true);
}

template<typename Dtype>
void ConvolutionLayerFFT<Dtype>::fft_pointwise_multiply_gpu(const std::complex<Dtype> *ffted_bottom_data,
                                                            std::complex<Dtype> *ptwise_result) {
  vector<int> shape;
  shape.push_back(this->num_output_);                 //              256
  shape.push_back((this->channels_ / this->group_));  // 96 / 2     = 48
  shape.push_back(this->fft_height_);                 //              32
  shape.push_back((this->fft_width_ / 2) + 1);        // 32 / 2 + 1 = 17

//  float time;
//  cudaEvent_t start, stop;
//  cudaEventCreate(&start);
//  cudaEventCreate(&stop);
//  cudaEventRecord(start, 0);
  fft_util_pointwise_multiply_gpu<Dtype>(shape, this->group_, ffted_bottom_data, this->ffted_weights_, ptwise_result);
//  cudaEventRecord(stop, 0);
//  cudaEventSynchronize(stop);
//  cudaEventElapsedTime(&time, start, stop);
//  cudaEventDestroy(start);
//  cudaEventDestroy(stop);
//
//  printf("Elapsed time on gpu: %f ms\n", time);

//  std::complex<Dtype> *res = reinterpret_cast<std::complex<Dtype> *>(fft_cpu_malloc<Dtype>(this->convolution_result_complex_size_));
//  CUDA_CHECK(cudaMemcpy(res, ptwise_result, this->convolution_result_complex_size_, cudaMemcpyDeviceToHost));
//
//  this->write_arr_to_disk("/home/harzigph/res_gpu.txt", this->num_output_, res, true);
//
//
//  printf("copied...");
}

template<typename Dtype>
void ConvolutionLayerFFT<Dtype>::fft_pointwise_multiply_npp_gpu(const std::complex<Dtype> *ffted_bottom_data,
                                                                std::complex<Dtype> *ptwise_result) {
  vector<int> shape;
  shape.push_back(this->num_output_);                 //              256
  shape.push_back((this->channels_ / this->group_));  // 96 / 2     = 48
  shape.push_back(this->fft_height_);                 //              32
  shape.push_back((this->fft_width_ / 2) + 1);        // 32 / 2 + 1 = 17

  float time;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  fft_util_pointwise_multiply_npp_gpu<Dtype>(shape, this->group_, ffted_bottom_data,
                                             this->ffted_weights_, ptwise_result);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  printf("Elapsed time on gpu npp: %f ms\n", time);
}

template<typename Dtype>
void ConvolutionLayerFFT<Dtype>::fft_normalize_gpu(std::complex<Dtype> *ptwise_result, Dtype *top_data) {
  Dtype *fft_convolution_result_real;
  CUDA_CHECK(cudaMalloc(&fft_convolution_result_real, this->convolution_result_real_size_));
  caffe_gpu_memset(this->convolution_result_real_size_, 0., fft_convolution_result_real);
  cufftHandle ifft_plan;
  fft_gpu_plan_many_dft_c2r_2d<Dtype>(&ifft_plan, this->fft_height_, this->fft_width_, this->num_output_);
  fft_gpu_execute_plan_c2r(ifft_plan, ptwise_result, fft_convolution_result_real);
  fft_gpu_destroy_plan(ifft_plan);

  // free the ptwise-result
  CUDA_CHECK(cudaFree(ptwise_result));
  Dtype ifft_normalize_factor = 1. / (this->fft_width_ * this->fft_height_);


  vector<int> shape;
  shape.push_back(1);
  shape.push_back(this->num_output_);
  shape.push_back(this->height_out_);
  shape.push_back(this->width_out_);


  fft_util_normalize_gpu(shape, this->kernel_h_, this->kernel_w_, this->stride_h_, this->stride_w_, ifft_normalize_factor,
                         this->fft_height_, this->fft_width_, fft_convolution_result_real, top_data);


  // free the intermediate convolution result, because data has already been forwarded to top_data.
  CUDA_CHECK(cudaFree(fft_convolution_result_real));
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
void ConvolutionLayerFFT<float>::fft_bottom_gpu(const float *bottom, std::complex<float> *&ffted_bottom_data);

template
void ConvolutionLayerFFT<float>::fft_convolve_gpu(std::complex<float> *ffted_bottom_data, float *top);

template
void ConvolutionLayerFFT<float>::fft_pointwise_multiply_gpu(const std::complex<float> *ffted_bottom_data,
                                                            std::complex<float> *ptwise_result);

template
void ConvolutionLayerFFT<float>::fft_pointwise_multiply_npp_gpu(const std::complex<float> *ffted_bottom_data,
                                                                std::complex<float> *ptwise_result);

template
void ConvolutionLayerFFT<float>::fft_normalize_gpu(std::complex<float> *ptwise_result, float *top_data);

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
void ConvolutionLayerFFT<double>::fft_bottom_gpu(const double *bottom, std::complex<double> *&ffted_bottom_data);

template
void ConvolutionLayerFFT<double>::fft_convolve_gpu(std::complex<double> *ffted_bottom_data, double *top);

template
void ConvolutionLayerFFT<double>::fft_pointwise_multiply_gpu(const std::complex<double> *ffted_bottom_data,
                                                            std::complex<double> *ptwise_result);

template
void ConvolutionLayerFFT<double>::fft_pointwise_multiply_npp_gpu(const std::complex<double> *ffted_bottom_data,
                                                                std::complex<double> *ptwise_result);

template
void ConvolutionLayerFFT<double>::fft_normalize_gpu(std::complex<double> *ptwise_result, double *top_data);

template
void ConvolutionLayerFFT<double>::mem_info_gpu();


}  // namespace caffe
