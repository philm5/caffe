#include <vector>
#include <caffe/util/fft_util.hpp>
#include <omp.h>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
ConvolutionLayerFFT<Dtype>::~ConvolutionLayerFFT() {
  // TODO: free weights...
}

template <typename Dtype>
void ConvolutionLayerFFT<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  this->fft_set_up();

  std::complex<Dtype> *ffted_bottom_data;

  this->fft_bottom_cpu(bottom[0], ffted_bottom_data);

  Dtype *iffted_convolution_result;
  this->fft_convolve_cpu(ffted_bottom_data, top[0]);
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

    // TODO: clean fft???
    // Do specific handling for allocation on cpu/gpu:
    switch (Caffe::mode()) {
      case Caffe::CPU:
        this->fft_set_up_cpu();
        this->fft_cpu_initialized_ = true;
        break;
      case Caffe::GPU:
        this->fft_set_up_gpu();
        this->fft_gpu_initialized_ = true;
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
  int shape[] = {this->num_output_, (this->channels_ / this->group_), this->kernel_h_, this->kernel_w_};

  // weights do not have to be padded. But the weights have to be flipped, since the convolution is actually a
  // cross-correlation.
  this->pad_real_blob(shape, weight_data, padded_real_weights, 0, 0, true);

  // The plan for fft of the weights
  // TODO: Do inplace fft to save memory???
  void *fft_weight_plan = fft_cpu_plan_many_dft_r2c_2d<Dtype>(this->fft_height_,
                                                              this->fft_width_,
                                                              this->num_weights_,
                                                              padded_real_weights,
                                                              this->fft_weights_,
                                                              FFTW_ESTIMATE);

  this->fft_weights_ = reinterpret_cast<std::complex<Dtype> *>(fft_cpu_malloc<Dtype>(this->padded_weights_complex_size_));

  // set the complex data to 0.
  caffe_memset(this->padded_weights_complex_size_, 0., this->fft_weights_);

  // Do the FFT of the padded weights:
  fft_cpu_execute_plan(fft_weight_plan);

  // free the padded real weights. There is no need for them anymore. Also free weights in blobs_[0] ???
  fft_cpu_free(padded_real_weights);

  // TODO: transpose weights here for gemm version of complex pt-wise multiplication.
}

template <typename Dtype>
void ConvolutionLayerFFT<Dtype>::fft_bottom_cpu(const Blob<Dtype> *bottom, std::complex<Dtype> *ffted_bottom_data) {
  // TODO: fft the bottom data
  // TODO: transpose if gemm version
}

template <typename Dtype>
void ConvolutionLayerFFT<Dtype>::fft_convolve_cpu(std::complex<Dtype> *ffted_bottom_data, Blob<Dtype> *top) {
  std::complex<Dtype> *ptwise_result;
  this->fft_pointwise_multiply_cpu(ffted_bottom_data, ptwise_result);

  // TODO: free ffted_bottom_data / transposed_bottom_data

  this->fft_normalize_cpu(ptwise_result, top);

  // TODO:
  // TODO: free ptwise_result
}

template <typename Dtype>
void ConvolutionLayerFFT<Dtype>::fft_pointwise_multiply_cpu(const std::complex<Dtype> *ffted_bottom_data,
                                                            std::complex<Dtype> *ptwise_result) {

}

template <typename Dtype>
void ConvolutionLayerFFT<Dtype>::fft_pointwise_multiply_ipp_cpu(const std::complex<Dtype> *ffted_bottom_data,
                                                                std::complex<Dtype> *ptwise_result) {

}


template <typename Dtype>
void ConvolutionLayerFFT<Dtype>::fft_pointwise_multiply_gemm_cpu(const std::complex<Dtype> *ffted_bottom_data,
                                                                 std::complex<Dtype> *ptwise_result) {

}


template <typename Dtype>
void ConvolutionLayerFFT<Dtype>::fft_normalize_cpu(const std::complex<Dtype> *ptwise_result, Blob<Dtype> *top) {
  // TODO: ifft
  // TODO: actual normalization and padding...
}

template <typename Dtype>
virtual void ConvolutionLayerFFT<Dtype>::pad_real_blob(int shape[4], const Dtype *blob_data, Dtype *padded_data,
                                                       int pad_h = 0, int pad_w = 0, bool flip = false) {
  const int N = shape[0];
  const int K = shape[1];
  const int H = shape[2];
  const int W = shape[3];

  int num_arr = N * K; // # of arrays (for weights it is num_weights [96 x 3]
  // for input data it is channels [ 1 x 3]

  // set everything to 0 before --> so not set weights are 0-padded :)
  caffe_memset(this->fft_real_size_ * num_arr * sizeof(Dtype), 0., padded_data);
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

          padded_data[idx_weight_real] = blob_data[idx_weight_in_blob];
        }
      }
    }
  }
}

/**
 * FFT GPU Stuff:
 */

template <typename Dtype>
void ConvolutionLayerFFT<Dtype>::fft_set_up_gpu() {
}

#ifdef CPU_ONLY
STUB_GPU(ConvolutionLayerFFT);
#endif

INSTANTIATE_CLASS(ConvolutionLayerFFT);

}  // namespace caffe
