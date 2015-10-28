#include <vector>
#include <caffe/util/fft_util.hpp>
#include <omp.h>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

#define DBG_OUTPUT

template <typename Dtype>
ConvolutionLayerFFT<Dtype>::~ConvolutionLayerFFT() {
  if (this->fft_on_) {
    fft_cpu_free<Dtype>(this->ffted_weights_);
  }
}

template <typename Dtype>
void ConvolutionLayerFFT<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                             const vector<Blob<Dtype>*>& top) {
  if (this->layer_param().name() == "conv2") {
    this->fft_on_ = true;
  }

  if (this->fft_on_) {
    this->Forward_cpu_fft(bottom, top);
  } else {
    this->Forward_cpu_normal(bottom, top);
  }
}

template <typename Dtype>
void ConvolutionLayerFFT<Dtype>::Forward_cpu_normal(const vector<Blob<Dtype>*>& bottom,
                                                 const vector<Blob<Dtype>*>& top) {

  const Dtype *weight = this->blobs_[0]->cpu_data();

  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype *bottom_data = bottom[i]->cpu_data();
    Dtype *top_data = top[i]->mutable_cpu_data();

    for (int n = 0; n < this->num_; ++n) {
      this->forward_cpu_gemm(bottom_data + bottom[i]->offset(n), weight,
                             top_data + top[i]->offset(n));
      if (this->bias_term_) {
        const Dtype *bias = this->blobs_[1]->cpu_data();
        this->forward_cpu_bias(top_data + top[i]->offset(n), bias);
      }
    }
  }
}

template <typename Dtype>
void ConvolutionLayerFFT<Dtype>::Forward_cpu_fft(const vector<Blob<Dtype>*>& bottom,
                                                      const vector<Blob<Dtype>*>& top) {
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = top[i]->mutable_cpu_data();

    for (int n = 0; n < this->num_; ++n) {
#ifdef DBG_OUTPUT
      start_time_ = cpu_time();
#endif
      this->Forward_cpu_fft_single(bottom_data + bottom[i]->offset(n), top_data + top[i]->offset(n));
#ifdef DBG_OUTPUT
      double fft_time = cpu_time();
#endif
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->cpu_data();
        this->forward_cpu_bias(top_data + top[i]->offset(n), bias);
      }
#ifdef DBG_OUTPUT
      double pass_time = cpu_time();
      LOG(INFO) << "Forward_cpu_fft_single: " << (pass_time - start_time_) * 1000 << "ms.";
#endif
    }
  }

}

  template <typename Dtype>
  void ConvolutionLayerFFT<Dtype>::Forward_cpu_fft_single(const Dtype *bottom,
                                                          Dtype *top) {
    this->fft_set_up();
#ifdef DBG_OUTPUT
    double set_up_time = cpu_time();
#endif

    // Allocate the complex memory for the bottom data
    std::complex<Dtype> *ffted_bottom_data =
        reinterpret_cast<std::complex<Dtype> *>(fft_cpu_malloc<Dtype>(this->padded_bottom_complex_size_));
    caffe_memset(this->padded_bottom_complex_size_, 0., ffted_bottom_data);
#ifdef DBG_OUTPUT
    double memset_bottom_time = cpu_time();
#endif

    this->fft_bottom_cpu(bottom, ffted_bottom_data);
    this->write_arr_to_disk("/home/harzigph/bottom_cpu.txt", this->channels_, ffted_bottom_data, true);
#ifdef DBG_OUTPUT
    double fft_bottom_cpu_time = cpu_time();
#endif
    this->fft_convolve_cpu(ffted_bottom_data, top);
#ifdef DBG_OUTPUT
    double fft_convolve_cpu_time = cpu_time();
#endif


#ifdef DBG_OUTPUT
    LOG(INFO) << "fft_set_up: " << (set_up_time - start_time_) * 1000 << "ms.";
    LOG(INFO) << "caffe_memset bottom: " << (memset_bottom_time - set_up_time) * 1000 << "ms.";
    LOG(INFO) << "fft_bottom_cpu: " << (fft_bottom_cpu_time - memset_bottom_time) * 1000 << "ms.";
    LOG(INFO) << "fft_convolve_cpu: " << (fft_convolve_cpu_time - fft_bottom_cpu_time) * 1000 << "ms.";
#endif
  }



/**
 * Generic FFT Stuff:
 */

template <typename Dtype>
void ConvolutionLayerFFT<Dtype>::fft_set_up() {
  // Only do the set-up if FFT mode is enabled.
  if(this->fft_on_ && !this->fft_initialized_) {
    // set fft width and height to be the image width and height (Padding and kernel size are considered!). These
    // sizes are checked if they are a power of 2. If not, round up to the next power of 2.
    int w_to_check = this->width_ + std::max(2 * this->pad_w_, (this->kernel_w_ - 1));
    int h_to_check = this->height_ + std::max(2 * this->pad_h_, (this->kernel_h_ - 1));

    if (!check_power_of_2(w_to_check)) {
      this->fft_width_ = next_power_of_2(w_to_check);
    }
    else {
      this->fft_width_ = w_to_check;
    }

    if (!check_power_of_2(h_to_check)) {
      this->fft_height_ = next_power_of_2(h_to_check);
    }
    else {
      this->fft_height_ = h_to_check;
    }

    // Calculate the size of the fft 2D matrices.
    // for sizes see: http://www.fftw.org/doc/Multi_002dDimensional-DFTs-of-Real-Data.html
    this->fft_real_size_ = this->fft_height_ * this->fft_width_;
    this->fft_complex_size_ = this->fft_height_ * (this->fft_width_ / 2 + 1);

    // Calculate the number of weights / filters used in this layer:
    this->num_weights_ = this->num_output_ * (this->channels_ / this->group_);

    // Set the sizes needed for allocation:
    this->padded_weights_real_size_ = this->fft_real_size_ * this->num_weights_ * sizeof(Dtype);
    this->padded_weights_complex_size_ = this->fft_complex_size_ * this->num_weights_ * sizeof(std::complex<Dtype>);
    // now the sizes for the bottom data:
    this->padded_bottom_real_size_ = this->fft_real_size_ * this->channels_ * 1 * sizeof(Dtype);
    this->padded_bottom_complex_size_ = this->fft_complex_size_ * this->channels_ * 1 * sizeof(std::complex<Dtype>);
    // and the sizes for the result (before normalizing and applying the stride):
    this->convolution_result_real_size_ = this->fft_real_size_ * this->num_output_ * sizeof(Dtype);
    this->convolution_result_complex_size_ = this->fft_complex_size_ * this->num_output_ * sizeof(std::complex<Dtype>);

    // set the shape of the input bottom shape. In the Forward_cpu the first dim of bottom can be more than 1 if
    // more than one image is passed (batching). But every input image is processed separately.
    std::vector<int> bot_shape;
    bot_shape.push_back(1);
    bot_shape.push_back(this->channels_);
    bot_shape.push_back(this->height_out_);
    bot_shape.push_back(this->width_out_);
    this->bottom_shape_ =  bot_shape;

    // TODO: clean fft???
    // Do specific handling for allocation on cpu/gpu:
    switch (Caffe::mode()) {
      case Caffe::CPU:
        this->fft_set_up_cpu();
        this->fft_cpu_initialized_ = true;
        break;
      case Caffe::GPU:
#ifndef CPU_ONLY
        this->fft_set_up_gpu();
        this->fft_gpu_initialized_ = true;
#endif
        break;
    }

    this->fft_initialized_ = true;
  }
}

/**
 * FFT CPU Stuff:
 */

template <typename Dtype>
void ConvolutionLayerFFT<Dtype>::fft_set_up_cpu() {
  // Initialize OpenMP for the CPU.
  this->num_threads_ = 1;
#ifdef _OPENMP
  this->num_threads_ = omp_get_max_threads();
  if (this->num_threads_ < 1) {
    LOG(WARNING) << "FFT Convolution Layer: omp_get_max_threads() =" << this->num_threads_;
    this->num_threads_ = 1;
  }
  fft_cpu_init_threads<Dtype>();
  fft_cpu_plan_with_nthreads<Dtype>(this->num_threads_);
#endif

  Dtype *padded_real_weights = reinterpret_cast<Dtype *>(fft_cpu_malloc<Dtype>(this->padded_weights_real_size_));

  // get the pointer to the weights
  const Dtype *weight_data = this->blobs_[0]->cpu_data();
  vector<int> shape;
  shape.push_back(this->num_output_);
  shape.push_back((this->channels_ / this->group_));
  shape.push_back(this->kernel_h_);
  shape.push_back(this->kernel_w_);

  // weights do not have to be padded (only 0-padded). But the weights have to be flipped, since the convolution is actually a
  // cross-correlation.
  this->pad_real_blob(shape, weight_data, padded_real_weights, 0, 0, true);


  this->write_arr_to_disk("/home/harzigph/pad_cpu.txt", this->num_output_ * (this->channels_ / this->group_), padded_real_weights, false);

  // The plan for fft of the weights
  // TODO: Do inplace fft to save memory???
  this->ffted_weights_ = reinterpret_cast<std::complex<Dtype> *>(fft_cpu_malloc<Dtype>(this->padded_weights_complex_size_));

  const void *fft_weight_plan = fft_cpu_plan_many_dft_r2c_2d<Dtype>(this->fft_height_,
                                                                    this->fft_width_,
                                                                    this->num_weights_,
                                                                    padded_real_weights,
                                                                    this->ffted_weights_,
                                                                    FFTW_ESTIMATE);

  // set the complex data to 0.
  caffe_memset(this->padded_weights_complex_size_, 0., this->ffted_weights_);

  // Do the FFT of the padded weights:
  fft_cpu_execute_plan<Dtype>(fft_weight_plan);



  this->write_arr_to_disk("/home/harzigph/fft_cpu.txt", this->num_output_ * (this->channels_ / this->group_), this->ffted_weights_, true);

  // Destroy the weight plan:
  fft_cpu_destroy_plan<Dtype>(fft_weight_plan);
  // free the padded real weights. There is no need for them anymore. Also free weights in blobs_[0] ???
  fft_cpu_free<Dtype>(padded_real_weights);

#if FFT_CONVOLUTION_KIND == FFT_CONVOLUTION_KIND_CGEMM
  // transpose weights if cgemm pt-wise product should be done
  std::complex<Dtype> *transposed_weights =
      reinterpret_cast<std::complex<Dtype> *>(fft_cpu_malloc<Dtype>(this->padded_weights_complex_size_));
  const int weight_shape[] = {this->num_output_, this->channels_ / this->group_,
                              this->fft_height_, (this->fft_width_ / 2) + 1};
  const int permutation[] = {2, 3, 0, 1};
  this->fft_permute_4d_cpu(this->ffted_weights_, transposed_weights, weight_shape, permutation);
  fft_cpu_free<Dtype>(this->ffted_weights_);
  this->ffted_weights_ = transposed_weights;
#endif
}

template<typename Dtype>
void ConvolutionLayerFFT<Dtype>::fft_permute_4d_cpu(const std::complex<Dtype> *in, std::complex<Dtype> *out,
                                                    const int shape[4], const int permutation[4]) {
  const int N = shape[0];              // 96
  const int K = shape[1];              // 3
  const int H = shape[2];              // 256
  const int W = shape[3];              // 129

  // define the indexes for the transposed version, so the data can be transposed very easily:
  const int K_T = shape[permutation[1]];
  const int H_T = shape[permutation[2]];
  const int W_T = shape[permutation[3]];

  int n = 0;
  int k = 0;
  int h = 0;
  int w = 0;

  int *vars[] = {&n, &k, &h, &w};

  int *n_t = vars[permutation[0]];
  int *k_t = vars[permutation[1]];
  int *h_t = vars[permutation[2]];
  int *w_t = vars[permutation[3]];

  // the indexing for the non-transposed (nt) input is split up into the loops, so some less ops are required:
  // idx_nt equals: ((n * K + k) * H + h) * W + w; idx_t is the index of the transposed output.
  for (n = 0; n < N; ++n) {
    for (k = 0; k < K; ++k) {
      const int nk_idx_nt = (n * K + k) * H;
      for (h = 0; h < H; ++h) {
        const int nkh_idx_nt = (nk_idx_nt + h ) * W;
        for (w = 0; w < W; ++w) {
          const int idx_nt = nkh_idx_nt + w;
          // alter indexing for t_idx
          const int idx_t = ((*n_t * K_T + *k_t) * H_T + *h_t) * W_T + *w_t;
          out[idx_t] = in[idx_nt];
        }
      }
    }
  }
}

template <typename Dtype>
void ConvolutionLayerFFT<Dtype>::fft_bottom_cpu(const Dtype *bottom, std::complex<Dtype> *&ffted_bottom_data) {
  const Dtype *bottom_blob = bottom;
  Dtype *padded_real_bottom = reinterpret_cast<Dtype *>(fft_cpu_malloc<Dtype>(this->padded_bottom_real_size_));

  // now pad the bottom data (it should have its origin in (h,w)), but don't flip it.1
  this->pad_real_blob(this->bottom_shape_, bottom_blob, padded_real_bottom, this->pad_h_, this->pad_w_, false);

  // Create FFT plan for the bottom data and execute it
  const void *fft_bottom_plan = fft_cpu_plan_many_dft_r2c_2d<Dtype>(this->fft_height_,
                                                                    this->fft_width_,
                                                                    this->channels_,
                                                                    padded_real_bottom,
                                                                    ffted_bottom_data,
                                                                    FFTW_ESTIMATE);
  fft_cpu_execute_plan<Dtype>(fft_bottom_plan);
  fft_cpu_destroy_plan<Dtype>(fft_bottom_plan);

#if FFT_CONVOLUTION_KIND == FFT_CONVOLUTION_KIND_CGEMM
  // transpose input if cgemm pt-wise product should be done
  std::complex<Dtype> *fft_transposed_bottom =
      reinterpret_cast<std::complex<Dtype> *>(fft_cpu_malloc<Dtype>(this->padded_bottom_complex_size_));
  const int shape_bottom[] = {1, this->channels_, this->fft_height_, (this->fft_width_ / 2) + 1};
  const int permutation_bottom[] = {2, 3, 0, 1};
  this->fft_permute_4d_cpu(ffted_bottom_data, fft_transposed_bottom, shape_bottom, permutation_bottom);

  fft_cpu_free<Dtype>(ffted_bottom_data);
  ffted_bottom_data = fft_transposed_bottom;
#endif
}

template <typename Dtype>
void ConvolutionLayerFFT<Dtype>::fft_convolve_cpu(std::complex<Dtype> *ffted_bottom_data, Dtype *top) {
  // alloc data for pointwise multiplication result (with channels added up) and set memory to 0

  std::complex<Dtype> *ptwise_result =
      reinterpret_cast<std::complex<Dtype> *>(fft_cpu_malloc<Dtype>(this->convolution_result_complex_size_));
  caffe_memset(this->convolution_result_complex_size_, 0., ptwise_result);

#if FFT_CONVOLUTION_KIND == FFT_CONVOLUTION_KIND_POINTWISE_IPP
  this->fft_pointwise_multiply_ipp_cpu(ffted_bottom_data, ptwise_result);
#elif FFT_CONVOLUTION_KIND == FFT_CONVOLUTION_KIND_POINTWISE_SIMPLE
  this->fft_pointwise_multiply_cpu(ffted_bottom_data, ptwise_result);
#elif FFT_CONVOLUTION_KIND == FFT_CONVOLUTION_KIND_CGEMM
  this->fft_pointwise_multiply_gemm_cpu(ffted_bottom_data, ptwise_result);
#endif

  fft_cpu_free<Dtype>(ffted_bottom_data);
  this->fft_normalize_cpu(ptwise_result, top);
}

template <typename Dtype>
void ConvolutionLayerFFT<Dtype>::fft_pointwise_multiply_cpu(const std::complex<Dtype> *ffted_bottom_data,
                                                            std::complex<Dtype> *ptwise_result) {
  const int N = this->num_output_;              //              256
  const int K = this->channels_ / this->group_; // 96 / 2     = 48
  const int H = this->fft_height_;              //              32 ?
  const int W = (this->fft_width_ / 2) + 1;     // 32 / 2 + 1 = 17

  const int weight_group_size = N / this->group_;

  int n = 0;
  int k = 0;
  int h = 0;
  int w = 0;

  const std::complex<Dtype> *bottom_complex = ffted_bottom_data;
  std::complex<Dtype> *weight_complex = this->ffted_weights_;

//#ifdef _OPENMP
//#pragma omp parallel for \
//          private(n, k, h, w) shared(ptwise_result, bottom_complex, weight_complex)
//#endif
  for (n = 0; n < N; ++n) {
    // check which group_ idx we are in
    const int group_idx = n / weight_group_size;
    for (k = 0; k < K; ++k) {
      // get the input_k. this is the k we use to index the input k-dimension. the max input_k is group_ times more
      // than the max k of the weight.
      const int input_k = k + group_idx * K; // 2 ops
      const int weight_offset = (n * K + k); // 2 ops

      /* in the following loops every filter response is being calculated. there are num_output_ * (channels_ / group_) filters...
       * each (1/group_) part is multiplied with each part of the input. e.g. for group_ = 2, n_o_ 256 and c_ = 96:
       * weights dim: 256x48x5x5, input dim: 1x96x27x27 --> splitted into [1x48x27x27, 1x48x27x27]
       * first 128 weights [128x48x5x5] will be convolved with first part of weights (dimension match!) --> 128 responses
       * same for 2nd part --> 2x128 responses to forward to the next channel
       */
      for (h = 0; h < H; ++h) {
        for (w = 0; w < W; ++w) {
          // Indexing: ((n * K + k) * H + h) * W + w
          const int input_idx = (input_k * H + h) * W + w; // 4 ops
          const std::complex<Dtype> input = bottom_complex[input_idx];

          const int weight_idx = (weight_offset * H + h) * W + w; // 4 ops
          const std::complex<Dtype> weight = weight_complex[weight_idx];

          // formula for complex mult from here: https://en.wikipedia.org/wiki/Complex_number#Multiplication_and_division
          // (a+bi) (c+di) = (ac-bd) + (bc+ad)i.
          Dtype a = std::real(weight);
          Dtype b = std::imag(weight);
          Dtype c = std::real(input);
          Dtype d = std::imag(input);

          const int res_idx = (n * H + h) * W + w; // 4 ops; before with channels: ((n * K + k) * H + h) * W + w;
          std::complex<Dtype> res(a * c - b * d, b * c + a * d); // 6 ops
          ptwise_result[res_idx] += res; // 2 ops
        }
      }
    }
  }


  //this->write_arr_to_disk("/home/harzigph/res_cpu.txt", this->num_output_, ptwise_result, true);
}

template <typename Dtype>
void ConvolutionLayerFFT<Dtype>::fft_pointwise_multiply_ipp_cpu(const std::complex<Dtype> *ffted_bottom_data,
                                                                std::complex<Dtype> *ptwise_result) {
  const int N = this->num_output_;
  const int K = this->channels_ / this->group_;
  const int weight_group_size = N / this->group_;

  int n = 0;
  int k = 0;

  const std::complex<Dtype> *bottom_complex = ffted_bottom_data;
  std::complex<Dtype> *weight_complex = this->ffted_weights_;

//#ifdef _OPENMP
//#pragma omp parallel for \
//          private(n, k) shared(bottom_complex, weight_complex, ptwise_result)
//#endif
  for (n = 0; n < N; ++n) {
    const int res_offset = n * this->fft_complex_size_;
    // check which group_ idx we are in
    const int group_idx = n / weight_group_size;
    for (k = 0; k < K; ++k) {
      // get the input_k. this is the k we use to index the input k-dimension. the max input_k is group_ times more
      // than the max k of the weight.
      const int input_offset = (k + group_idx * K) * this->fft_complex_size_;
      const int weight_offset = (n * K + k) * this->fft_complex_size_;

      // pSrcDst[n ] = pSrcDst[n ] + pSrc1[n ] * pSrc2[n ], 0 ��������� n < len.
      ipp_complex_add_product<Dtype>(bottom_complex + input_offset, weight_complex + weight_offset, ptwise_result + res_offset, this->fft_complex_size_);
    }
  }
}


template <typename Dtype>
void ConvolutionLayerFFT<Dtype>::fft_pointwise_multiply_gemm_cpu(const std::complex<Dtype> *ffted_bottom_data,
                                                                 std::complex<Dtype> *ptwise_result) {
  // alloc data for result
  std::complex<Dtype> * fft_transposed_result =
      reinterpret_cast<std::complex<Dtype> *>(fft_cpu_malloc<Dtype>(this->convolution_result_complex_size_));

  //                      num_output * channels / group = 96 * 3
  const int weight_size = this->num_output_ * (this->channels_ / this->group_);

  //                      num_images      * channels    =  1 * 3
  const int bottom_size = 1 * this->channels_;

  //                      num_output * num_images       = 96 * 1
  const int output_size = this->num_output_ * 1;

  const std::complex<Dtype> one_complex(1., 0.);
  const std::complex<Dtype> zero_complex(0., 0.);

  const int H = this->fft_height_;
  const int W = (this->fft_width_ / 2) + 1;
  const int G = this->group_;

  const int group_offset_weight = weight_size / G;
  const int group_offset_input = bottom_size / G;
  const int group_offset_output = output_size / G;

  const int M = this->num_output_ / G;
  const int N = 1;
  const int K = this->channels_ / this->group_;

  const std::complex<Dtype> **weight_arr = new const std::complex<Dtype> *[H*W*G];
  const std::complex<Dtype> **input_arr = new const std::complex<Dtype> *[H*W*G];
  std::complex<Dtype> **output_arr = new std::complex<Dtype> *[H*W*G];

  int idx = 0;

  //const int shape[] = {this->num_output_, this->channels_ / this->group_, this->fft_height_, (this->fft_width_ / 2) + 1};
  for (int h = 0; h < H; ++h) {
    for (int w = 0; w < W; ++w) {

      const std::complex<Dtype> *weight = this->ffted_weights_ + (h * W + w ) * weight_size;
      const std::complex<Dtype> *input = ffted_bottom_data + (h * W + w ) * bottom_size;
      std::complex<Dtype> *output = fft_transposed_result + (h * W + w ) * output_size;

      for (int g = 0; g < G; ++g) {
        weight_arr[idx] = weight + g * group_offset_weight;
        input_arr[idx] = input + g * group_offset_input;
        output_arr[idx] = output + g * group_offset_output;
        idx++;
//        std::complex<Dtype> *weight_g = weight + g * group_offset_weight;
//        std::complex<Dtype> *input_g = input + g * group_offset_input;
//        std::complex<Dtype> *output_g = output + g * group_offset_output;
//
//        caffe_cpu_gemm_complex<Dtype>(CblasNoTrans, CblasTrans, M, N, K,
//                                      &one_complex, weight_g, input_g, &zero_complex, output_g);
      }
    }
  }

  // Do batched matrix multiplication
  caffe_cpu_gemm_complex_batch<Dtype>(CblasNoTrans, CblasTrans, M, N, K,
                                      &one_complex, weight_arr, input_arr, &zero_complex, output_arr, H*W*G);

  delete weight_arr;
  delete input_arr;
  delete output_arr;

  // result_dim = 256 x 129 x 96 x 1 ==> 1 x 96 x 256 x 129
  const int shape_result[] = {H, W, this->num_output_, 1};
  const int permutation_result[] = {2, 3, 0, 1};
  this->fft_permute_4d_cpu(fft_transposed_result, ptwise_result, shape_result, permutation_result);
}


template <typename Dtype>
void ConvolutionLayerFFT<Dtype>::fft_normalize_cpu(std::complex<Dtype> *ptwise_result, Dtype *top_data) {
  // alloc data for real result
  Dtype *fft_convolution_result_real_ = reinterpret_cast<Dtype *>(fft_cpu_malloc<Dtype>(this->convolution_result_real_size_));

  caffe_memset(this->convolution_result_real_size_, 0., fft_convolution_result_real_);

  // Create the IFFT plan (the c2r transformation) and execute it
  const void *ifft_plan = fft_cpu_plan_many_dft_c2r_2d<Dtype>(this->fft_height_,
                                                              this->fft_width_,
                                                              this->num_output_,
                                                              ptwise_result,
                                                              fft_convolution_result_real_,
                                                              FFTW_ESTIMATE);

  fft_cpu_execute_plan<Dtype>(ifft_plan);

  // Destroy the IFFT plan and free the complex convolution result:
  fft_cpu_destroy_plan<Dtype>(ifft_plan);
  fft_cpu_free<Dtype>(ptwise_result);

  // here the stride handling and FFT normalization is happening:
  Dtype ifft_normalize_factor = 1. / (this->fft_width_ * this->fft_height_);
  int N = this->num_output_;

//#ifdef _OPENMP
//#pragma omp parallel for
//#endif
  for (int n = 0; n < N; ++n) {
    int offset_res_real = n * this->fft_real_size_;

    for (int h = 0; h < this->height_out_; ++h) // =55 in 1st layer
    {
      // caffe does a valid convolution. fft is a full convolution. so the first 'valid' result is at
      // idx (kernel_h_ - 1). The stride times the idx of the output pixel will be added onto this.
      int h_idx = (this->kernel_h_ - 1) + h * this->stride_h_; // 3 ops

      for (int w = 0; w < this->width_out_; ++w) // =55 in 1st layer
      {
        // caffe does a valid convolution. fft is a full convolution. so the first 'valid' result is at
        // idx (kernel_w_ - 1). The stride times the idx of the output pixel will be added onto this.
        int w_idx = (this->kernel_w_ - 1) + w * this->stride_w_; // 3 ops
        //((n * K + k) * H + h) * W + w;
        int top_data_idx = (n * this->height_out_ + h) * this->width_out_ + w; // 4 ops

        // the index in the data of the convolution result array (the real one)
        int res_data_idx = offset_res_real + h_idx * this->fft_width_ + w_idx; // 3 ops

        // normalize fft and sum up everything from the input channels...
        top_data[top_data_idx] = fft_convolution_result_real_[res_data_idx] * ifft_normalize_factor; // 1 op
      }
    }
  }

  // delete the temporary result.
  fft_cpu_free<Dtype>(fft_convolution_result_real_);
}

template <typename Dtype>
void ConvolutionLayerFFT<Dtype>::pad_real_blob(std::vector<int> shape, const Dtype *blob_data, Dtype *padded_data,
                                               int pad_h, int pad_w, bool flip) {
  const int N = shape[0];
  const int K = shape[1];
  const int H = shape[2];
  const int W = shape[3];

  int num_arr = N * K; // # of arrays (for weights it is num_weights [96 x 3]
  // for input data it is channels [ 1 x 3]

  // set everything to 0 before --> so not set weights are 0-padded :)
  caffe_memset(this->fft_real_size_ * num_arr * sizeof(Dtype), 0., padded_data);
//#ifdef _OPENMP
//#pragma omp parallel for
//#endif
  for (int n = 0; n < N; n++) {
    for (int k = 0; k < K; k++) {
      for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
          // ((n * ch_gr + c) * fft_height_ + h)* 2 * (fft_width_ / 2 + 1) + w
          const int offset_weight_real = (n * K + k) * this->fft_real_size_;
          // e.g. a 3x3 filter should fit into a 5x5 because the image size is 5x5
          // <--W-->
          // ^ f f f 0 0
          // H f f f 0 0
          // _ f f f 0 0
          //   0 0 0 0 0
          //   0 0 0 0 0

          const int idx_weight_real = offset_weight_real + (h + pad_h) * this->fft_width_ + (w + pad_w);
          // copy each weight into the fft_weights_in_real_
          // get ptr to blob data. indexing see: http://caffe.berkeleyvision.org/tutorial/net_layer_blob.html
          // Blob memory is row-major in layout, so the last / rightmost dimension changes fastest. For example,
          // in a 4D blob, the value at index (n, k, h, w) is physically located at index ((n * K + k) * H + h) * W + w.
          // 96 x 3 x 11 x 11 (num_output_, channels_ / group_, kernel_height, width_)

          // if flip = true ==> flip the indices of the weights. Caffe actually does not a convolution but a
          // a cross-correlation according to: https://github.com/BVLC/caffe/issues/2513
          const int h_idx = flip ? H - (h + 1) : h;
          const int w_idx = flip ? W - (w + 1) : w;
          const int idx_weight_in_blob = ((n * K + k) * H + h_idx) * W + w_idx;

          padded_data[idx_weight_real] = blob_data[idx_weight_in_blob];
        }
      }
    }
  }
}

template<typename Dtype>
void ConvolutionLayerFFT<Dtype>::write_arr_to_disk(const char *output_name, size_t size, void *arr, bool is_complex) {
  std::ofstream fout(output_name); //opening an output stream for file test.txt
  if (fout.is_open()) {
    //file opened successfully so we are here
    std::cout << "File Opened successfully!!!. Writing data from array to file" << std::endl;

    size_t size_multiplier = is_complex ? this->fft_complex_size_ : this->fft_real_size_;
    size_t length = size_multiplier * size;
    for (int i = 0; i < length; i++) {
      if (is_complex) {
        std::complex<Dtype> *arr_conv = reinterpret_cast<std::complex<Dtype> *>(arr);
        fout << arr_conv[i].real() << " + " << arr_conv[i].imag() << " * i\n";
      }
      else {
        Dtype *arr_conv = reinterpret_cast<Dtype *>(arr);
        fout << *(arr_conv + i) << "\n";
      }
    }
    std::cout << "Array data successfully saved into the file " << output_name << std::endl;
  }

  fout.close();
}

#ifdef CPU_ONLY
STUB_GPU(ConvolutionLayerFFT);
#endif

INSTANTIATE_CLASS(ConvolutionLayerFFT);

}  // namespace caffe
