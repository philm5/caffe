#include <vector>
#include <complex>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/fft_util.hpp"
#include "caffe/vision_layers.hpp"

#include <ctime>

#ifdef _OPENMP
#include <omp.h>
#include <sys/time.h>
#endif
// #define WRITE_ARRAYS_T0_DISK
#define WRITE_DEBUG
// #define WRITE_DEBUG_FW

#define FFT_CONVOLUTION_KIND_POINTWISE_IPP 0
#define FFT_CONVOLUTION_KIND_POINTWISE_MKL 1
#define FFT_CONVOLUTION_KIND_POINTWISE_SIMPLE 2
#define FFT_CONVOLUTION_KIND_CGEMM 3

#define FFT_CONVOLUTION_KIND FFT_CONVOLUTION_KIND_CGEMM


namespace caffe {
template<typename Dtype>
ConvolutionLayerFFT<Dtype>::~ConvolutionLayerFFT() {
  fft_cpu_free<Dtype>(fft_weights_out_complex_);
#ifdef WRITE_DEBUG
  LOG(ERROR) << "complex weights freed.";
#endif
}

template<typename Dtype>
void ConvolutionLayerFFT<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                                            const vector<Blob<Dtype> *> &top) {
  ConvolutionLayer<Dtype>::LayerSetUp(bottom, top);
}

template<typename Dtype>
void ConvolutionLayerFFT<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
  ConvolutionLayer<Dtype>::Reshape(bottom, top);
//  if (this->layer_param().name() == "conv1")
//  {
//    // do normal convolution...
//    this->fft_on_ = false;
//  }
//  else
  {
    if (!this->weights_converted_) {
      this->fft_set_up();
      this->fft_on_ = true;
    }
  }
}

template<typename Dtype>
void ConvolutionLayerFFT<Dtype>::fft_set_up() {
  // here only the memory should be reserved... The weight data will
  // either be trained or loaded from disk. So there is no data available
  // yet...

  // ---- openmp ------------------------------------------
  num_of_threads_ = 1;
#ifdef _OPENMP
  num_of_threads_ = omp_get_max_threads();
  if (num_of_threads_ < 1) {
    LOG(WARNING) << "FFT Convolution Layer: omp_get_max_threads() =" << num_of_threads_;
    num_of_threads_ = 1;
  }
  fft_cpu_init_threads<Dtype>();
  fft_cpu_plan_with_nthreads<Dtype>(this->num_of_threads_);
#endif

  // set fft width and height to be the image width and height (for now without padding...)
  // Check if width and height is a power of 2. If not, round up to the next power of 2.

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

  // for sizes see: http://www.fftw.org/doc/Multi_002dDimensional-DFTs-of-Real-Data.html
  this->fft_real_size_ = this->fft_height_ * this->fft_width_;
  this->fft_complex_size_ = this->fft_height_ * (this->fft_width_ / 2 + 1);

  // Allocations & plan for weights
  int num_weights = this->num_output_ * (this->channels_ / this->group_);

  this->weight_alloc_size_in = this->fft_real_size_ * num_weights * sizeof(Dtype);
  this->fft_weights_in_real_ = reinterpret_cast<Dtype *>(fft_cpu_malloc<Dtype>(this->weight_alloc_size_in));

  this->weight_alloc_size_out = this->fft_complex_size_ * num_weights * sizeof(std::complex<Dtype>);
  this->fft_weights_out_complex_ = reinterpret_cast<std::complex<Dtype> *>(fft_cpu_malloc<Dtype>(this->weight_alloc_size_out));

  // The plan. Is a plan for the actual conversion. Conversion will be done when weights are rdy...
  this->fft_weight_plan_ = fft_cpu_plan_many_dft_r2c_2d<Dtype>(this->fft_height_,
                                                     this->fft_width_,
                                                     num_weights,
                                                     this->fft_weights_in_real_,
                                                     this->fft_weights_out_complex_,
                                                     FFTW_ESTIMATE);
}

template<typename Dtype>
void ConvolutionLayerFFT<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {

  if (this->fft_on_ == false)
  {
    this->Forward_cpu_normal(bottom, top);
  }
  else
  {
    if (!this->weights_converted_) {
      // if weights were converted alrdy don't do that again :)
      this->convert_weights_fft();
    }

#ifdef WRITE_DEBUG_FW
    double begin_clock = cpu_time();
#endif
    this->Forward_cpu_fft(bottom, top);
#ifdef WRITE_DEBUG_FW
    double end_clock = cpu_time();
  LOG(ERROR) << this->layer_param().name() << ": " << 1000.0 * (end_clock - begin_clock) << " ms.";
#endif
  }

}


template<typename Dtype>
void ConvolutionLayerFFT<Dtype>::Forward_cpu_normal(const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
  const Dtype *weight = this->blobs_[0]->cpu_data();

  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype *bottom_data = bottom[i]->cpu_data();
    Dtype *top_data = top[i]->mutable_cpu_data();


    for (int n = 0; n < this->num_; ++n) {

#ifdef WRITE_DEBUG_FW
      double begin_clock = cpu_time();
#endif
      this->forward_cpu_gemm(bottom_data + bottom[i]->offset(n), weight,
                             top_data + top[i]->offset(n));
#ifdef WRITE_DEBUG_FW
      double end_clock = cpu_time();
            LOG(ERROR) << this->layer_param().name() << ": " << 1000.0 * (end_clock - begin_clock) << " ms.";
#endif

      if (this->bias_term_) {
        const Dtype *bias = this->blobs_[1]->cpu_data();
        this->forward_cpu_bias(top_data + top[i]->offset(n), bias);
      }
    }
  }
}


template<typename Dtype>
void ConvolutionLayerFFT<Dtype>::Forward_cpu_fft(const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {

  for (int i = 0; i < bottom.size(); ++i) {
    this->Forward_cpu_fft_single(bottom[i], top[i]);

    // bias
    Dtype *top_data = top[i]->mutable_cpu_data();
    for (int n = 0; n < this->num_; ++n) {
      if (this->bias_term_) {
        const Dtype *bias = this->blobs_[1]->cpu_data();
        this->forward_cpu_bias(top_data + top[i]->offset(n), bias);
      }
    }
  }
#ifdef _OPENMP
  fft_cpu_cleanup_threads<Dtype>();
#endif
}

template<typename Dtype>
void ConvolutionLayerFFT<Dtype>::
Forward_cpu_fft_single(const Blob<Dtype> *bottom, Blob<Dtype> *top) {
// Allocations & plan for input values (bottom values)
  this->alloc_size_input_real = this->fft_real_size_ * 1 * this->channels_ * sizeof(Dtype);
  this->alloc_size_input_complex = this->fft_complex_size_ * 1 * this->channels_ * sizeof(std::complex<Dtype>);
  this->fft_input_real_ = reinterpret_cast<Dtype *>(fft_cpu_malloc<Dtype>(this->alloc_size_input_real));
  this->fft_input_complex_ = reinterpret_cast<std::complex<Dtype> *>(fft_cpu_malloc<Dtype>(this->alloc_size_input_complex));

  // ---------------------------------------------------
  // The plan to compute the input values to complex
  this->fft_input_plan_ = fft_cpu_plan_many_dft_r2c_2d<Dtype>(this->fft_height_,
                                                          this->fft_width_,
                                                          this->channels_,
                                                          this->fft_input_real_,
                                                          this->fft_input_complex_,
                                                          FFTW_ESTIMATE);
  // Convert input and fft it...
  this->convert_bottom(bottom);

  // Destroy input plan...
  fft_cpu_destroy_plan<Dtype>(this->fft_input_plan_);

  size_t summed_result_size_complex = this->fft_complex_size_ * this->num_output_ * sizeof(std::complex<Dtype>);
//  this->fft_summed_up_result_complex_ =
//      reinterpret_cast<std::complex<Dtype> *>(fft_cpu_malloc<Dtype>(summed_result_size_complex));
//
//  // set complex mem to 0
//  caffe_memset(summed_result_size_complex, 0., this->fft_summed_up_result_complex_);

  // alloc data for result
  this->fft_conv_result_complex_ =
      reinterpret_cast<std::complex<Dtype> *>(fft_cpu_malloc<Dtype>(summed_result_size_complex));

  // set complex mem to 0
  caffe_memset(summed_result_size_complex, 0., this->fft_conv_result_complex_);

#ifdef WRITE_DEBUG
  double begin_clock = cpu_time();
#endif
  this->convolve_fft();
#ifdef WRITE_DEBUG
  double end_clock = cpu_time();
#endif

  this->write_arr_to_disk("conv_complex.txt", this->num_output_, this->fft_conv_result_complex_, true);

#ifdef WRITE_DEBUG
  LOG(ERROR) << "fft convolve took " << 1000.0 * (end_clock - begin_clock) << " ms.";
#endif

  // free the memory used only once per forward call (the input memory was already used in convolve_fft)
  fft_cpu_free<Dtype>(this->fft_input_complex_);

  size_t conv_result_real_size = this->fft_real_size_ * this->num_output_ * sizeof(Dtype);

  // alloc data for real result
  this->fft_conv_result_real_ = reinterpret_cast<Dtype *>(fft_cpu_malloc<Dtype>(conv_result_real_size));

  // set real mem to 0
  caffe_memset(conv_result_real_size, 0., this->fft_conv_result_real_);

  // The ifft plan; the backward conversion from c2r
  this->ifft_plan_ = fft_cpu_plan_many_dft_c2r_2d<Dtype>(this->fft_height_,
                                                         this->fft_width_,
                                                         this->num_output_,
                                                         this->fft_conv_result_complex_,
                                                         this->fft_conv_result_real_,
                                                         FFTW_ESTIMATE);


#ifdef WRITE_DEBUG
  begin_clock = cpu_time();
  // execute the actual ifft
#endif
  fft_cpu_execute_plan<Dtype>(this->ifft_plan_);

  // Destroy input plan...
  fft_cpu_destroy_plan<Dtype>(this->ifft_plan_);
#ifdef WRITE_DEBUG
  end_clock = cpu_time();
#endif
//  // free the complex result, because it was already ifft-ed to real.
//  fft_cpu_free<Dtype>(this->fft_summed_up_result_complex_);

  // free the complex result, because it was already ifft-ed to real.
  fft_cpu_free<Dtype>(this->fft_conv_result_complex_);

#ifdef WRITE_DEBUG
  LOG(ERROR) << "ifft took " << 1000.0 * (end_clock - begin_clock)<< " ms.";
#endif

  this->normalize_ifft_result(top);

#ifndef WRITE_ARRAYS_T0_DISK
  // free the real result memory
  fft_cpu_free<Dtype>(this->fft_conv_result_real_);
#endif

#ifdef WRITE_TOP_RES
  std::stringstream ss;
  ss << "res_top_fft_" << this->layer_param_.name() << ".txt";
  const char *s = ss.str().c_str();
  this->write_simple_arr_to_disk(s, top[0]->count() , top[0]->cpu_data());
#endif

#ifdef WRITE_ARRAYS_T0_DISK
  int N = this->num_output_;
  int K = (this->channels_ / this->group_);
  this->write_arr_to_disk("conv_res_real.txt", N , this->fft_conv_result_real_);
  this->write_simple_arr_to_disk("top_data.txt", this->num_output_ * this->height_out_ * this->width_out_, top->mutable_cpu_data());
#endif
}

template<typename Dtype>
void ConvolutionLayerFFT<Dtype>::convert_bottom(const Blob<Dtype> *bottom) {
  const Dtype *input_data_blob = bottom->cpu_data();
  vector<int> shape = bottom->shape();
#ifdef WRITE_DEBUG
  LOG(ERROR) << "start of conversion of bottom data...";
#endif
  // bottom data has to be shifted with the padding param to ensure the correct calculation; but not the weights
  this->transform_blob_to_real_array(shape[0], shape[1], shape[2], shape[3], input_data_blob, this->fft_input_real_, this->pad_h_, this->pad_w_);

  caffe_memset(this->fft_complex_size_ * shape[0] * shape[1] * sizeof(std::complex<Dtype>),
               0.,
               this->fft_input_complex_);
#ifdef WRITE_DEBUG
  double begin_clock = cpu_time();
#endif
  fft_cpu_execute_plan<Dtype>(this->fft_input_plan_);
#ifdef WRITE_DEBUG
  double end_clock = cpu_time();
#endif

#ifndef WRITE_ARRAYS_T0_DISK
  fft_cpu_free<Dtype>(this->fft_input_real_);
#endif

#ifdef WRITE_DEBUG
  LOG(ERROR) << "fft for bottom data took " << 1000.0 * (end_clock - begin_clock)<< " ms.";
#endif

#ifdef WRITE_ARRAYS_T0_DISK
  this->write_arr_to_disk("input_real_in.txt", shape[0] * shape[1], this->fft_input_real_);
  this->write_arr_to_disk("input_complex_out.txt", shape[0] * shape[1], this->fft_input_complex_, true);
#endif
}

template<typename Dtype>
void ConvolutionLayerFFT<Dtype>::transform_blob_to_real_array(int N,
                                                              int K,
                                                              int H,
                                                              int W,
                                                              const Dtype *blob_data,
                                                              Dtype *padded_real_data,
                                                              int pad_h,
                                                              int pad_w,
                                                              bool flip) {
  int num_arr = N * K; // # of arrays (for weights it is num_weights [96 x 3]
  //              for input data it is channels [ 1 x 3]

  // set everything to 0 before --> so not set weights are 0-padded :)
  caffe_memset(this->fft_real_size_ * num_arr * sizeof(Dtype), 0., padded_real_data);
#ifdef _OPENMP
  #pragma omp parallel for
#endif
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

          const int idx_weight_real = offset_weight_real + (h + pad_h) * this->fft_height_ + (w + pad_w);
          // copy each weight into the fft_weights_in_real_
          // get ptr to blob data. indexing see: http://caffe.berkeleyvision.org/tutorial/net_layer_blob.html
          // Blob memory is row-major in layout, so the last / rightmost dimension changes fastest. For example,
          // in a 4D blob, the value at index (n, k, h, w) is physically located at index ((n * K + k) * H + h) * W + w.
          // 96 x 3 x 11 x 11 (num_output_, channels_ / group_, kernel_height, width_)

          // if flip = true ==> flip the indices of the weights. Caffe actually does not a convolution but a
          // a cross-correlation according to: https://github.com/BVLC/caffe/issues/2513
          const int h_idx = flip ? H - (h + 1) : h;
          const int w_idx = flip ? W - (w + 1) : w;
          int idx_weight_in_blob = ((n * K + k) * H + h_idx) * W + w_idx;

          padded_real_data[idx_weight_real] = blob_data[idx_weight_in_blob];
        }
      }
    }
  }
}

template<typename Dtype>
void ConvolutionLayerFFT<Dtype>::convert_weights_fft() {

  // the data location of the weights before the conversion...
  const Dtype *cpu_data = this->blobs_[0]->cpu_data();

  // transform data to real
  int N = this->num_output_;
  int K = (this->channels_ / this->group_);
  int H = this->kernel_h_;
  int W = this->kernel_w_;

  // bottom data has to be shifted with the padding param to ensure the correct calculation; but not the weights ==> 0,0
  this->transform_blob_to_real_array(N, K, H, W, cpu_data, this->fft_weights_in_real_, 0, 0, true);

  // set complex mem to 0
  caffe_memset(this->fft_complex_size_ * N * K * sizeof(std::complex<Dtype>), 0.,
               this->fft_weights_out_complex_);
#ifdef WRITE_DEBUG
  double begin_clock = cpu_time();
#endif
  fft_cpu_execute_plan<Dtype>(this->fft_weight_plan_);

#ifdef WRITE_DEBUG
  double end_clock = cpu_time();
#endif

#ifndef WRITE_ARRAYS_T0_DISK
  fft_cpu_free<Dtype>(fft_weights_in_real_);
#endif

#ifdef WRITE_DEBUG
  LOG(ERROR) << "fft for weight layer took " << 1000.0 * (end_clock - begin_clock) << " ms.";
#endif

  this->weights_converted_ = true;

#ifdef WRITE_ARRAYS_T0_DISK
  this->write_arr_to_disk("weights_real_in.txt", N * K, this->fft_weights_in_real_);
  this->write_arr_to_disk("weights_complex_out.txt", N * K, this->fft_weights_out_complex_, true);
#endif
}

#if FFT_CONVOLUTION_KIND == FFT_CONVOLUTION_KIND_POINTWISE_IPP
template<typename Dtype>
void ConvolutionLayerFFT<Dtype>::convolve_fft() {
  const int N = this->num_output_;
  const int K = this->channels_ / this->group_;

  const int weight_group_size = N / this->group_;
  //const int G = this->group_;
  //const int multiply_size = this->fft_complex_size_ * K;

  int n = 0;
  int k = 0;

  std::complex<Dtype> *in_complex = this->fft_input_complex_;
  std::complex<Dtype> *weight_complex = this->fft_weights_out_complex_;
  std::complex<Dtype> *res_complex = this->fft_conv_result_complex_;

#ifdef _OPENMP
#pragma omp parallel for \
          private(n, k) shared(res_complex, in_complex, weight_complex)
#endif
  for (n = 0; n < N; ++n) {
    const int res_offset = n * this->fft_complex_size_;
    // check which group_ idx we are in
    const int group_idx = n / weight_group_size;
    for (k = 0; k < K; ++k) {
      // get the input_k. this is the k we use to index the input k-dimension. the max input_k is group_ times more
      // than the max k of the weight.
      const int input_offset = (k + group_idx * K) * this->fft_complex_size_;
      const int weight_offset = (n * K + k) * this->fft_complex_size_;

      // pSrcDst[n ] = pSrcDst[n ] + pSrc1[n ] * pSrc2[n ], 0 ≤ n < len.
      ipp_complex_add_product<Dtype>(in_complex + input_offset, weight_complex + weight_offset, res_complex + res_offset, this->fft_complex_size_);
    }
  }
}
#elif FFT_CONVOLUTION_KIND == FFT_CONVOLUTION_KIND_POINTWISE_MKL
template<typename Dtype>
void ConvolutionLayerFFT<Dtype>::convolve_fft() {
  const int N = this->num_output_ / this->group_;
  const int K = this->channels_ / this->group_;
  const int G = this->group_;
  const int multiply_size = this->fft_complex_size_ * K;

  int n = 0;
  int g = 0;
  int k = 0;

  std::complex<Dtype> *in_complex = this->fft_input_complex_;
  std::complex<Dtype> *weight_complex = this->fft_weights_out_complex_;
  std::complex<Dtype> *res_complex = this->fft_conv_result_complex_;

#ifdef _OPENMP
#pragma omp parallel for \
          private(n, g) shared(res_complex, in_complex, weight_complex)
#endif
  for (n = 0; n < N; ++n) {
    for (g = 0; g < G; ++g) {
      const int input_offset = g * K * this->fft_complex_size_;
      const int weight_offset = ((n + g * N) * K) * this->fft_complex_size_;
      caffe_complex_mul<Dtype>(multiply_size, in_complex + input_offset, weight_complex + weight_offset, res_complex + weight_offset);
    }
  }


  // sum up result from each channel into one channel. So fewer iffts have to be done!!! (faster!)
  std::complex<Dtype> *sum_res_complex = this->fft_summed_up_result_complex_;

#ifdef _OPENMP
#pragma omp parallel for \
          private(n, k) shared(res_complex, sum_res_complex)
#endif
  for (n = 0; n < N * G; ++n) {
    std::complex<Dtype> *sum_res = this->fft_summed_up_result_complex_ + n * this->fft_complex_size_;
    for (k = 0; k < K; ++k) {
      const int offset_conv_res = (n * K + k) * this->fft_complex_size_;
      caffe_complex_add(this->fft_complex_size_, sum_res, res_complex + offset_conv_res, sum_res);
    }
  }
}
#elif FFT_CONVOLUTION_KIND == FFT_CONVOLUTION_KIND_POINTWISE_SIMPLE
template<typename Dtype>
void ConvolutionLayerFFT<Dtype>::convolve_fft() {
  const int N = this->num_output_;              //              256
  const int K = this->channels_ / this->group_; // 96 / 2     = 48
  const int H = this->fft_height_;              //              32 ?
  const int W = (this->fft_width_ / 2) + 1;     // 32 / 2 + 1 = 17

  const int weight_group_size = N / this->group_;

  int n = 0;
  int k = 0;
  int h = 0;
  int w = 0;

  std::complex<Dtype> *in_complex = this->fft_input_complex_;
  std::complex<Dtype> *weight_complex = this->fft_weights_out_complex_;
  std::complex<Dtype> *res_complex = this->fft_conv_result_complex_;

#ifdef _OPENMP
  #pragma omp parallel for \
          private(n, k, h, w) shared(res_complex, in_complex, weight_complex)
#endif
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
          const std::complex<Dtype> input = in_complex[input_idx];

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
          res_complex[res_idx] += res; // 2 ops
        }
      }
    }
  }
}
#elif FFT_CONVOLUTION_KIND == FFT_CONVOLUTION_KIND_CGEMM
template<typename Dtype>
void ConvolutionLayerFFT<Dtype>::convolve_fft() {

  // matrix width  is : H*W*C = this->fft_complex_size_ * this->channels_
  // matrix height is : N_O = this->num_output_
  const int H = this->fft_height_;
  const int W = (this->fft_width_ / 2) + 1;
  const int G = this->group_;
  this->fft_transposed_weights_ = reinterpret_cast<std::complex<Dtype> *>(fft_cpu_malloc<Dtype>(weight_alloc_size_out));
  const int shape[] = {this->num_output_, this->channels_ / this->group_, H, W};
  const int permutation[] = {2, 3, 0, 1};
  this->permute_4d(this->fft_weights_out_complex_, this->fft_transposed_weights_, shape, permutation);

  this->fft_transposed_bottom_ = reinterpret_cast<std::complex<Dtype> *>(fft_cpu_malloc<Dtype>(this->alloc_size_input_complex));
  const int shape_bottom[] = {1, this->channels_, H, W};
  const int permutation_bottom[] = {2, 3, 0, 1};
  this->permute_4d(this->fft_input_complex_, this->fft_transposed_bottom_, shape_bottom, permutation_bottom);

  // alloc data for result
  size_t summed_result_size_complex = this->fft_complex_size_ * this->num_output_ * sizeof(std::complex<Dtype>);
  this->fft_transposed_result_ =
      reinterpret_cast<std::complex<Dtype> *>(fft_cpu_malloc<Dtype>(summed_result_size_complex));

  //                      num_output * channels / group = 96 * 3
  const int weight_size = shape[0] * shape[1];

  //                      num_images      * channels    =  1 * 3
  const int bottom_size = shape_bottom[0] * shape_bottom[1];

  //                      num_output * num_images       = 96 * 1
  const int output_size = shape[0] * shape_bottom[0];

  std::complex<Dtype> one_complex(1., 0.);
  std::complex<Dtype> zero_complex(0., 0.);

  for (int h = 0; h < H; ++h) {
    for (int w = 0; w < W; ++w) {

      std::complex<Dtype> *weight = this->fft_transposed_weights_ + (h * W + w ) * weight_size;
      std::complex<Dtype> *input = this->fft_transposed_bottom_ + (h * W + w ) * bottom_size;
      std::complex<Dtype> *output = this->fft_transposed_result_ + (h * W + w ) * output_size;

      caffe_cpu_gemm_complex<Dtype>(CblasNoTrans, CblasTrans, shape[0], shape_bottom[0], shape[1],
                                    &one_complex, weight, input, &zero_complex, output);
    }
  }



  // result_dim = 256 x 129 x 96 x 1 ==> 1 x 96 x 256 x 129
  const int shape_result[] = {H, W, this->num_output_, 1};
  const int permutation_result[] = {2, 3, 0, 1};
  this->permute_4d(this->fft_transposed_result_, this->fft_conv_result_complex_, shape_result, permutation_result);
}
#endif



template<typename Dtype>
void ConvolutionLayerFFT<Dtype>::permute_4d(const std::complex<Dtype> *in, std::complex<Dtype> *out, const int shape[4], const int permutation[4]) {

#ifdef WRITE_DEBUG
  double begin_clock = cpu_time();
#endif


  const int N = shape[0];              // 96
  const int K = shape[1];              // 3
  const int H = shape[2];              // 256
  const int W = shape[3];              // 129

  // const int N_T = shape[permutation[0]];
  const int K_T = shape[permutation[1]];
  const int H_T = shape[permutation[2]];
  const int W_T = shape[permutation[3]];

  // indexing is here split up, to speed up loops ((n * K + k) * H + h) * W + w

  int n = 0;
  int k = 0;
  int h = 0;
  int w = 0;

  int *vars[] = {&n, &k, &h, &w};

  int *n_t = vars[permutation[0]];
  int *k_t = vars[permutation[1]];
  int *h_t = vars[permutation[2]];
  int *w_t = vars[permutation[3]];

  for (n = 0; n < N; ++n) {
    for (k = 0; k < K; ++k) {
      for (h = 0; h < H; ++h) {
        for (w = 0; w < W; ++w) {

          const int idx_nt = ((n * K + k) * H + h) * W + w;
          // alter indexing for t_idx
          const int idx_t = ((*n_t * K_T + *k_t) * H_T + *h_t) * W_T + *w_t;
          out[idx_t] = in[idx_nt];
        }
      }
    }
  }

#ifdef WRITE_DEBUG
  double end_clock = cpu_time();
  LOG(ERROR) << "weights transposed.... took:" << 1000.0 * (end_clock - begin_clock) << " ms.";
  LOG(ERROR) << "weights transposed";
#endif
}


template<typename Dtype>
void ConvolutionLayerFFT<Dtype>::normalize_ifft_result(Blob<Dtype> *top) {
  Dtype *top_data = top->mutable_cpu_data();

  Dtype ifft_normalize_factor = 1. / (this->fft_width_ * this->fft_height_);

  int N = this->num_output_;
  int K = (this->channels_ / this->group_);

#ifdef _OPENMP
  #pragma omp parallel for
#endif
  for (int n = 0; n < N; ++n) {
    //for (int k = 0; k < K; ++k) {
      // 1 op
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
          top_data[top_data_idx] = this->fft_conv_result_real_[res_data_idx] * ifft_normalize_factor; // 1 op
      //  }
      }
   }
  }
}

template<typename Dtype>
void ConvolutionLayerFFT<Dtype>::write_arr_to_disk(const char *output_name, int size, void *arr, bool is_complex) {
  std::ofstream fout(output_name); //opening an output stream for file test.txt
  if (fout.is_open()) {
    //file opened successfully so we are here
    std::cout << "File Opened successfully!!!. Writing data from array to file" << std::endl;

    int size_multiplier = is_complex ? this->fft_complex_size_ : this->fft_real_size_;
    for (int i = 0; i < size_multiplier * size; i++) {
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

template<typename Dtype>
void ConvolutionLayerFFT<Dtype>::write_simple_arr_to_disk(const char *output_name, int size, const Dtype *arr) {
  std::ofstream fout(output_name); //opening an output stream for file test.txt
  if (fout.is_open()) {
    //file opened successfully so we are here
    std::cout << "File Opened successfully!!!. Writing data from array to file" << std::endl;

    for (int i = 0; i < size; i++) {
      fout << *(arr + i) << "\n";
    }
    std::cout << "Array data successfully saved into the file " << output_name << std::endl;
  }

  fout.close();
}

//#ifdef CPU_ONLY
//	STUB_GPU(ConvolutionLayerFFT);
//#endif

INSTANTIATE_CLASS(ConvolutionLayerFFT);

}  // namespace caffe
