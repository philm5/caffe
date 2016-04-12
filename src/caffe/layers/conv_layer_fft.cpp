#ifdef USE_FFT
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/conv_layer_fft.hpp"

using namespace std;

#ifndef USE_MKL
// cgemm_batched is not available in atlas.... perhaps in openblas???
#undef FFT_CONVOLUTION_KIND
#define FFT_CONVOLUTION_KIND FFT_CONVOLUTION_KIND_POINTWISE_SIMPLE
#endif

namespace caffe {

template <typename Dtype>
void ConvolutionLayerFFT<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                         const vector<Blob<Dtype>*>& top) {
  ConvolutionLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(2, this->num_spatial_axes_)
  << "FFT Convolution input must have 2 spatial axes "
  << "(e.g., height and width). "
  << "Use 'engine: CAFFE' for general ND convolution.";
}

template <typename Dtype>
void ConvolutionLayerFFT<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                            const vector<Blob<Dtype>*>& top) {
  ConvolutionLayer<Dtype>::LayerSetUp(bottom, top);

  ConvolutionParameter conv_param = this->layer_param_.convolution_param();

  if (conv_param.has_fft_update_weights_each_batch()) {
    this->fft_update_weights_each_batch_ = conv_param.fft_update_weights_each_batch();
  } else {
    this->fft_update_weights_each_batch_ = false;
  }

  if (conv_param.has_fft_inplace()) {
    this->fft_inplace_ = conv_param.fft_inplace();
  } else {
    this->fft_inplace_ = false;
  }
}

template <typename Dtype>
void ConvolutionLayerFFT<Dtype>::LayerUpdateBeforeBatch() {
  ConvolutionLayer<Dtype>::LayerUpdateBeforeBatch();
  if (!this->fft_initialized_) {
    this->fft_set_up();
  }

  switch (Caffe::mode()) {
    case Caffe::CPU:
      this->fft_update_weights_cpu();
      break;
    case Caffe::GPU:
#ifndef CPU_ONLY
      this->fft_update_weights_gpu();
#else
      NO_GPU;
#endif
      break;
  }
}

template <typename Dtype>
ConvolutionLayerFFT<Dtype>::~ConvolutionLayerFFT() {
  if (this->fft_on_) {
    if (this->fft_cpu_initialized_ == true) {
      this->fft_free_weights_cpu();
    }
    if (this->fft_gpu_initialized_ == true) {
      this->fft_free_weights_gpu();
    }

  }
}

template <typename Dtype>
void ConvolutionLayerFFT<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                             const vector<Blob<Dtype>*>& top) {
  this->fft_on_ = true;

  if (this->fft_on_) {
    this->Forward_cpu_fft(bottom, top);
  } else {
    this->Forward_cpu_normal(bottom, top);
  }
}

template <typename Dtype>
void ConvolutionLayerFFT<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                              const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  this->fft_on_ = true;

  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->cpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_cpu_bias(bias_diff, top_diff + top[i]->offset(n));
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      const Dtype* bottom_data = bottom[i]->cpu_data();
      Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
      if (!this->fft_on_) {
        for (int n = 0; n < this->num_; ++n) {
          // gradient w.r.t. weight. Note that we will accumulate diffs.
          if (this->param_propagate_down_[0]) {
            this->weight_cpu_gemm(bottom_data + bottom[i]->offset(n),
                                  top_diff + top[i]->offset(n), weight_diff);
          }
          // TODO: caffe doesnt this part in benchmark???
          // gradient w.r.t. bottom data, if necessary.
          if (propagate_down[i]) {
            this->backward_cpu_gemm(top_diff + top[i]->offset(n), weight,
                                    bottom_diff + bottom[i]->offset(n));
          }
        }
      } else {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->Weight_cpu_fft(bottom_data + bottom[i]->offset(0),
                               top_diff + top[i]->offset(0), weight_diff);
        }

        this->fft_update_weights_cpu();

        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->Backward_cpu_fft(top_diff + top[i]->offset(0),
                                 bottom_diff + bottom[i]->offset(0));
        }
      }
    }
  }
}

template <typename Dtype>
void ConvolutionLayerFFT<Dtype>::Weight_cpu_fft(const Dtype* input, const Dtype* output, Dtype* weight) {
  // fft the bottom data...
  this->fft_bottom_cpu(input);

  // fft the top diff data...
  this->fft_top_cpu(output);

  this->fft_convolve_weight_cpu(weight);
}

template <typename Dtype>
void ConvolutionLayerFFT<Dtype>::Backward_cpu_fft(const Dtype* top_blob, Dtype* bottom_blob) {
  // fft the top diff data...
  this->fft_top_cpu(top_blob);

  this->fft_convolve_backward_cpu(bottom_blob);
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
  this->fft_set_up();

  // the unit tests modify weights between two Forward ops by step_size to check if backward calculates the gradient correctly.
  // But since the weights are only ffted once to save compute power, changes arent reflected in the complex values (ffted ones).
  // If fft_update_weights_each_batch_ mode is on, the weights are ffted every pass!!! Costs extra computing effort if done.
  if (this->fft_update_weights_each_batch_) {
    this->fft_update_weights_cpu();
  }

  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = top[i]->mutable_cpu_data();

    this->Forward_cpu_fft_single(bottom_data + bottom[i]->offset(0), top_data + top[i]->offset(0));

    for (int n = 0; n < this->num_; ++n) {
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->cpu_data();
        this->forward_cpu_bias(top_data + top[i]->offset(n), bias);
      }
    }
  }
}

template <typename Dtype>
void ConvolutionLayerFFT<Dtype>::Forward_cpu_fft_single(const Dtype *bottom,
                                                        Dtype *top) {
  this->fft_bottom_cpu(bottom);

  this->fft_convolve_cpu(top);
}

/**
 * Generic FFT Stuff:
 */
template <typename Dtype>
void ConvolutionLayerFFT<Dtype>::fft_set_up() {
  // Only do the set-up if FFT mode is enabled.
  if(this->fft_on_ && !this->fft_initialized_) {
    // set fft width and height to be the image width and height (Padding and kernel size are considered!). These
    // sizes are checked if they are a power of 2.
    int h_to_check = this->input_shape(1) + std::max(2 * this->pad_.cpu_data()[0], (this->kernel_shape_.cpu_data()[0] - 1));
    int w_to_check = this->input_shape(2) + std::max(2 * this->pad_.cpu_data()[1], (this->kernel_shape_.cpu_data()[1] - 1));

//    LOG(ERROR) << this->layer_param_.name() << " - H: " << this->input_shape(1) << " PAD_H: " <<
//        this->pad_.cpu_data()[0] << " KERNEL_H: " << this->kernel_shape_.cpu_data()[0] <<
//        " H_TO_CHECK: " << h_to_check;

    // Make the fft size a multiple of 16. FFTW and CUFFT perform best on FFT sizes which are factorable like this:
    // 2^a * 3^b * 5^c * 7^d. If divisible by 4 there is extra speed up.
    if ((h_to_check % 16) > 0) {
      this->fft_height_ = h_to_check + (16 - (h_to_check % 16));
    }

    if ((w_to_check % 16) > 0) {
      this->fft_width_ = w_to_check + (16 - (w_to_check % 16));
    }
//
//    LOG(ERROR) << "FFT_H: " << this->fft_height_;

    //  If not, round up to the next power of 2.
    //    if (!check_power_of_2(w_to_check)) {
    //      this->fft_width_ = next_power_of_2(w_to_check);
    //    }
    //    else {
    //      this->fft_width_ = w_to_check;
    //    }
    //
    //    if (!check_power_of_2(h_to_check)) {
    //      this->fft_height_ = next_power_of_2(h_to_check);
    //    }
    //    else {
    //      this->fft_height_ = h_to_check;
    //    }

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
    this->padded_bottom_real_size_ = this->fft_real_size_ * this->channels_ * this->num_ * sizeof(Dtype);
    this->padded_bottom_complex_size_ = this->fft_complex_size_ * this->channels_ * this->num_ * sizeof(std::complex<Dtype>);
    // and the sizes for the result (before normalizing and applying the stride):
    this->convolution_result_real_size_ = this->fft_real_size_ * this->num_output_ * this->num_ * sizeof(Dtype);
    this->convolution_result_complex_size_ = this->fft_complex_size_ * this->num_output_ * this->num_ * sizeof(std::complex<Dtype>);

//    LOG(ERROR) << "padded_weights_real_size_: " << padded_weights_real_size_;
//    LOG(ERROR) << "padded_weights_complex_size_: " << padded_weights_complex_size_;
//
//    LOG(ERROR) << "padded_bottom_real_size_: " << padded_bottom_real_size_;
//    LOG(ERROR) << "padded_bottom_complex_size_: " << padded_bottom_complex_size_;
//
//    LOG(ERROR) << "convolution_result_real_size_: " << convolution_result_real_size_;
//    LOG(ERROR) << "convolution_result_complex_size_: " << convolution_result_complex_size_;

    std::vector<int> top_shape;
    top_shape.push_back(this->num_);
    top_shape.push_back(this->num_output_);
    top_shape.push_back(this->output_shape_[0]);
    top_shape.push_back(this->output_shape_[1]);
    this->top_shape_ = top_shape;

    this->fft_bottom_plan_ = NULL;
    this->fft_top_plan_ = NULL;
    this->ifft_plan_ = NULL;


    switch (Caffe::mode()) {
      case Caffe::CPU:
        this->fft_set_up_cpu();
        this->fft_cpu_initialized_ = true;
        break;
      case Caffe::GPU:
#ifndef CPU_ONLY
        this->fft_set_up_gpu();
        this->fft_gpu_initialized_ = true;
#else
        NO_GPU;
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

  // The plan for fft of the weights
  this->ffted_weights_ = reinterpret_cast<std::complex<Dtype> *>(fft_cpu_malloc<Dtype>(this->padded_weights_complex_size_));

  // init the weights...
  this->fft_update_weights_cpu();

  // Allocate the real and complex memory for the bottom data

  this->ffted_bottom_data_ =
      reinterpret_cast<std::complex<Dtype> *>(fft_cpu_malloc<Dtype>(this->padded_bottom_complex_size_));

  this->ptwise_result_ =
      reinterpret_cast<std::complex<Dtype> *>(fft_cpu_malloc<Dtype>(this->convolution_result_complex_size_));

  // dont allocate real buffers in inplace mode!
  if (!this->fft_inplace_) {
    this->padded_real_bottom_ =
        reinterpret_cast<Dtype *>(fft_cpu_malloc<Dtype>(this->padded_bottom_real_size_));

    this->fft_convolution_result_real_ =
        reinterpret_cast<Dtype *>(fft_cpu_malloc<Dtype>(this->convolution_result_real_size_));
  } else {
    this->padded_real_bottom_ = NULL;
    this->fft_convolution_result_real_ = NULL;
  }
}

template<typename Dtype>
void ConvolutionLayerFFT<Dtype>::fft_update_weights_cpu() {
  // get the pointer to the weights
  const Dtype *weight_data = this->blobs_[0]->cpu_data();
  vector<int> shape;
  shape.push_back(this->num_output_);
  shape.push_back((this->channels_ / this->group_));
  shape.push_back(this->kernel_shape_.cpu_data()[0]);
  shape.push_back(this->kernel_shape_.cpu_data()[1]);

  if (!this->fft_inplace_) {
    Dtype *padded_real_weights = reinterpret_cast<Dtype *>(fft_cpu_malloc<Dtype>(this->padded_weights_real_size_));
    // weights do not have to be padded (only 0-padded). But the weights have to be flipped, since the convolution is actually a
    // cross-correlation.
    this->pad_real_blob(shape, weight_data, padded_real_weights, 0, 0);

    const void *fft_weight_plan = fft_cpu_plan_many_dft_r2c_2d<Dtype>(this->fft_height_,
                                                                      this->fft_width_,
                                                                      this->num_weights_,
                                                                      padded_real_weights,
                                                                      this->ffted_weights_,
                                                                      FFTW_ESTIMATE);

    // Do the FFT of the padded weights:
    fft_cpu_execute_plan<Dtype>(fft_weight_plan);

    // Destroy the weight plan:
    fft_cpu_destroy_plan<Dtype>(fft_weight_plan);
    // free the padded real weights. There is no need for them anymore. TODO: Also free weights in blobs_[0] ???
    fft_cpu_free<Dtype>(padded_real_weights);
  } else {
    Dtype *ffted_weights = reinterpret_cast<Dtype *>(this->ffted_weights_);
    this->pad_real_blob(shape, weight_data, ffted_weights, 0, 0, 1, 1, true);

    const void *fft_weight_plan = fft_cpu_plan_many_dft_r2c_2d<Dtype>(this->fft_height_,
                                                                      this->fft_width_,
                                                                      this->num_weights_,
                                                                      ffted_weights,
                                                                      this->ffted_weights_,
                                                                      FFTW_ESTIMATE);

    // Do the FFT of the padded weights:
    fft_cpu_execute_plan<Dtype>(fft_weight_plan);

    // Destroy the weight plan:
    fft_cpu_destroy_plan<Dtype>(fft_weight_plan);
  }


#if FFT_CONVOLUTION_KIND == FFT_CONVOLUTION_KIND_CGEMM
  // transpose weights if cgemm pt-wise product should be done
  std::complex<Dtype> *transposed_weights =
      reinterpret_cast<std::complex<Dtype> *>(fft_cpu_malloc<Dtype>(this->padded_weights_complex_size_));
  const int weight_shape[] = {this->num_output_, this->channels_ / this->group_,
      this->fft_height_, (this->fft_width_ / 2) + 1};
  // const int permutation[] = {2, 3, 0, 1};
  this->fft_geam_transpose_cpu(this->ffted_weights_, transposed_weights, weight_shape, 2);
  // this->fft_permute_4d_cpu(this->ffted_weights_, transposed_weights, weight_shape, permutation);
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

template<typename Dtype>
void ConvolutionLayerFFT<Dtype>::fft_geam_transpose_cpu(const std::complex<Dtype> *in, std::complex<Dtype> *out,
                                                        const int shape[4], const int sep) {
#ifdef USE_MKL
  // idea taken from fbfft paper.
  int rows = 1;
  int cols = 1;

  for (int i = 0; i < sep; i++) {
    rows *= shape[i];
  }


  for (int i = sep; i < 4; i++) {
    cols *= shape[i];
  }


  std::complex<Dtype> one_complex(1., 0.);
  std::complex<Dtype> zero_complex(0., 0.);

  caffe_cpu_geam_complex<Dtype>(CblasTrans, CblasNoTrans, rows, cols, &one_complex, in, cols,
                                NULL, &zero_complex, rows, out, rows);
#else
#warning considers sep = 2!
#warning much slower than with mkl. Build with MKL instead!
  // considers sep = 2, so the seperator within a 4d array with dim-indexes 0,1,2,3 is between 1 and 2!
  // Adapt permutation accordingly for other sep.
  const int permutation[] = {2, 3, 0, 1};
  this->fft_permute_4d_cpu(in, out, shape, permutation);
#endif
}


template<typename Dtype>
void ConvolutionLayerFFT<Dtype>::fft_free_weights_cpu() {
  if (this->fft_initialized_) {
    fft_cpu_free<Dtype>(this->ffted_weights_);
    fft_cpu_free<Dtype>(this->padded_real_bottom_);
    fft_cpu_free<Dtype>(this->ffted_bottom_data_);
    fft_cpu_free<Dtype>(this->ptwise_result_);
    fft_cpu_free<Dtype>(this->fft_convolution_result_real_);

    if (this->fft_bottom_plan_ != NULL) {
      fft_cpu_destroy_plan<Dtype>(this->fft_bottom_plan_);
      this->fft_bottom_plan_ = NULL;
    }

    if (this->fft_top_plan_ != NULL) {
      fft_cpu_destroy_plan<Dtype>(this->fft_top_plan_);
      this->fft_top_plan_ = NULL;
    }

    if (this->ifft_plan_ != NULL) {
      fft_cpu_destroy_plan<Dtype>(this->ifft_plan_);
      this->ifft_plan_ = NULL;
    }

    fft_cpu_cleanup_threads<Dtype>();

    this->fft_initialized_ = false;
  }
}

template <typename Dtype>
void ConvolutionLayerFFT<Dtype>::fft_bottom_cpu(const Dtype *bottom) {
  const Dtype *bottom_blob = bottom;

  if (!this->fft_inplace_) {
    // now pad the bottom data (it should have its origin in (h,w)), but don't flip it.1
    this->pad_real_blob(*this->bottom_shape_, bottom_blob, this->padded_real_bottom_, this->pad_.cpu_data()[0], this->pad_.cpu_data()[1]);

    this->fft_bottom_plan_ = fft_cpu_plan_many_dft_r2c_2d<Dtype>(this->fft_height_,
                                                                 this->fft_width_,
                                                                 this->channels_ * this->num_,
                                                                 this->padded_real_bottom_,
                                                                 this->ffted_bottom_data_,
                                                                 FFTW_ESTIMATE);
  } else {
    Dtype *ffted_bottom_data = reinterpret_cast<Dtype *>(this->ffted_bottom_data_);
    // now pad the bottom data (it should have its origin in (h,w)), but don't flip it.1
    this->pad_real_blob(*this->bottom_shape_, bottom_blob, ffted_bottom_data, this->pad_.cpu_data()[0],
                        this->pad_.cpu_data()[1], 1, 1, true);

    this->fft_bottom_plan_ = fft_cpu_plan_many_dft_r2c_2d<Dtype>(this->fft_height_,
                                                                 this->fft_width_,
                                                                 this->channels_ * this->num_,
                                                                 ffted_bottom_data,
                                                                 this->ffted_bottom_data_,
                                                                 FFTW_ESTIMATE);
  }
  fft_cpu_execute_plan<Dtype>(this->fft_bottom_plan_);
  fft_cpu_free<Dtype>(this->fft_bottom_plan_);
  this->fft_bottom_plan_ = NULL;

#if FFT_CONVOLUTION_KIND == FFT_CONVOLUTION_KIND_CGEMM
  // transpose input if cgemm pt-wise product should be done
  std::complex<Dtype> *fft_transposed_bottom =
      reinterpret_cast<std::complex<Dtype> *>(fft_cpu_malloc<Dtype>(this->padded_bottom_complex_size_));
  const int shape_bottom[] = {this->num_, this->channels_, this->fft_height_, (this->fft_width_ / 2) + 1};

  this->fft_geam_transpose_cpu(this->ffted_bottom_data_, fft_transposed_bottom, shape_bottom, 2);
  // this->fft_permute_4d_cpu(this->ffted_bottom_data_, fft_transposed_bottom, shape_bottom, permutation_bottom);

  fft_cpu_free<Dtype>(this->ffted_bottom_data_);
  this->ffted_bottom_data_ = fft_transposed_bottom;
#endif
}

template <typename Dtype>
void ConvolutionLayerFFT<Dtype>::fft_top_cpu(const Dtype *top) {
  const Dtype *top_blob = top;

  if (!this->fft_inplace_) {
    // now pad the top data (it should have its origin in (h,w)), but don't flip it.1
    this->pad_real_blob(this->top_shape_, top_blob, this->fft_convolution_result_real_, 0, 0,
                        this->stride_.cpu_data()[0], this->stride_.cpu_data()[1]);

    this->fft_top_plan_ = fft_cpu_plan_many_dft_r2c_2d<Dtype>(this->fft_height_,
                                                              this->fft_width_,
                                                              this->num_output_ * this->num_,
                                                              this->fft_convolution_result_real_,
                                                              this->ptwise_result_,
                                                              FFTW_ESTIMATE);
  } else {
    Dtype *ptwise_result = reinterpret_cast<Dtype *>(this->ptwise_result_);
    // now pad the top data (it should have its origin in (h,w)), but don't flip it.1
    this->pad_real_blob(this->top_shape_, top_blob, ptwise_result, 0, 0,
                        this->stride_.cpu_data()[0], this->stride_.cpu_data()[1], true);

    this->fft_top_plan_ = fft_cpu_plan_many_dft_r2c_2d<Dtype>(this->fft_height_,
                                                              this->fft_width_,
                                                              this->num_output_ * this->num_,
                                                              ptwise_result,
                                                              this->ptwise_result_,
                                                              FFTW_ESTIMATE);
  }

  fft_cpu_execute_plan<Dtype>(this->fft_top_plan_);
  fft_cpu_free<Dtype>(this->fft_top_plan_);
  this->fft_top_plan_ = NULL;

#if FFT_CONVOLUTION_KIND == FFT_CONVOLUTION_KIND_CGEMM
  // transpose input if cgemm pt-wise product should be done
  std::complex<Dtype> *fft_transposed_top =
      reinterpret_cast<std::complex<Dtype> *>(fft_cpu_malloc<Dtype>(this->convolution_result_complex_size_));
  const int shape_top[] = {this->num_, this->num_output_, this->fft_height_, (this->fft_width_ / 2) + 1};
  // const int permutation_bottom[] = {2, 3, 0, 1};
  this->fft_geam_transpose_cpu(this->ptwise_result_, fft_transposed_top, shape_top, 2);
  // this->fft_permute_4d_cpu(this->ffted_bottom_data_, fft_transposed_bottom, shape_bottom, permutation_bottom);

  fft_cpu_free<Dtype>(this->ptwise_result_);
  this->ptwise_result_ = fft_transposed_top;
#endif
}

template <typename Dtype>
void ConvolutionLayerFFT<Dtype>::fft_convolve_cpu(Dtype *top) {
  // alloc data for pointwise multiplication result (with channels added up) and set memory to 0
  caffe_memset(this->convolution_result_complex_size_, 0., this->ptwise_result_);

#if FFT_CONVOLUTION_KIND == FFT_CONVOLUTION_KIND_IPP
#ifdef USE_IPP
  this->fft_pointwise_multiply_ipp_cpu();
#else
  this->fft_pointwise_multiply_cpu();
#endif
#elif FFT_CONVOLUTION_KIND == FFT_CONVOLUTION_KIND_POINTWISE_SIMPLE
  this->fft_pointwise_multiply_cpu();
#elif FFT_CONVOLUTION_KIND == FFT_CONVOLUTION_KIND_CGEMM
  this->fft_pointwise_multiply_gemm_cpu();
#endif


  vector<int> shape;
  shape.push_back(this->num_);
  shape.push_back(this->num_output_);
  shape.push_back(this->output_shape_[0]);
  shape.push_back(this->output_shape_[1]);

  this->fft_normalize_cpu(shape, this->stride_.cpu_data()[0], this->stride_.cpu_data()[1], 0, 0,
                          this->ptwise_result_, this->fft_convolution_result_real_,
                          top, false);
}

template <typename Dtype>
void ConvolutionLayerFFT<Dtype>::fft_convolve_backward_cpu(Dtype *bottom) {
  // now pointwise multiply. (since we have to flip weights in the backward pass [in contrast to the fw pass], we have to do a convolution here
  // (vs. cross-correlation in fw pass).
  // weight. Confused?? See here:
  // FW Pass (cross-correlation, but fft-pointwise-mult is a convolution.) --> do multiplication with conjugate
  // BW Pass (conv) --> do normal pointwise multiplication.
  caffe_memset(padded_bottom_complex_size_, (Dtype) 0., this->ffted_bottom_data_);

#if FFT_CONVOLUTION_KIND == FFT_CONVOLUTION_KIND_IPP
#warning no ipp implementation here!
  this->fft_pointwise_multiply_backward_cpu();
#elif FFT_CONVOLUTION_KIND == FFT_CONVOLUTION_KIND_POINTWISE_SIMPLE
  this->fft_pointwise_multiply_backward_cpu();
#elif FFT_CONVOLUTION_KIND == FFT_CONVOLUTION_KIND_CGEMM
  this->fft_pointwise_multiply_gemm_backward_cpu();
#endif

  vector<int> shape;
  shape.push_back(this->num_);
  shape.push_back(this->channels_);
  shape.push_back(this->input_shape(1));
  shape.push_back(this->input_shape(2));

  this->fft_normalize_cpu(shape, 1, 1, this->pad_.cpu_data()[0], this->pad_.cpu_data()[1],
                          this->ffted_bottom_data_, this->padded_real_bottom_,
                          bottom, false);
}

template <typename Dtype>
void ConvolutionLayerFFT<Dtype>::fft_convolve_weight_cpu(Dtype *weight) {
  // alloc data for pointwise multiplication result (with channels added up) and set memory to 0
  caffe_memset(this->padded_weights_complex_size_, 0., this->ffted_weights_);

#if FFT_CONVOLUTION_KIND == FFT_CONVOLUTION_KIND_IPP
#warning no ipp implementation here!
  this->fft_pointwise_multiply_weight_cpu();
#elif FFT_CONVOLUTION_KIND == FFT_CONVOLUTION_KIND_POINTWISE_SIMPLE
  this->fft_pointwise_multiply_weight_cpu();
#elif FFT_CONVOLUTION_KIND == FFT_CONVOLUTION_KIND_CGEMM
  this->fft_pointwise_multiply_gemm_weight_cpu();
#endif

  vector<int> shape;
  shape.push_back(this->num_output_);
  shape.push_back(this->channels_ / this->group_);
  shape.push_back(this->kernel_shape_.cpu_data()[0]);
  shape.push_back(this->kernel_shape_.cpu_data()[1]);

  Dtype *padded_real_weights = NULL;
  if (!this->fft_inplace_) {
    padded_real_weights = reinterpret_cast<Dtype *>(fft_cpu_malloc<Dtype>(this->padded_weights_real_size_));
  }

  this->fft_normalize_cpu(shape, 1, 1, 0, 0, this->ffted_weights_ ,
                          padded_real_weights, weight, true);

  if (!this->fft_inplace_) {
    fft_cpu_free<Dtype>(padded_real_weights);
  }
}

template <typename Dtype>
void ConvolutionLayerFFT<Dtype>::fft_pointwise_multiply_cpu() {
  const int batch_size = this->num_;            //              10
  const int N = this->num_output_;              //              256
  const int K = this->channels_ / this->group_; // 96 / 2     = 48
  const int H = this->fft_height_;              //              32 ?
  const int W = (this->fft_width_ / 2) + 1;     // 32 / 2 + 1 = 17

  const int weight_group_size = N / this->group_;

  int batch_idx = 0;
  int n = 0;
  int k = 0;
  int hw = 0;

  std::complex<Dtype> *weight_complex = this->ffted_weights_;
  std::complex<Dtype> *this_bottom_data = this->ffted_bottom_data_;
  std::complex<Dtype> *this_ptwise_result = this->ptwise_result_;

#ifdef _OPENMP
#pragma omp parallel for private(batch_idx, n, hw, k) collapse(3)
#endif
  for (batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
    for (n = 0; n < N; ++n) {
      /* in the following loops every filter response is being calculated. there are num_output_ * (channels_ / group_) filters...
       * each (1/group_) part is multiplied with each part of the input. e.g. for group_ = 2, n_o_ 256 and c_ = 96:
       * weights dim: 256x48x5x5, input dim: 1x96x27x27 --> splitted into [1x48x27x27, 1x48x27x27]
       * first 128 weights [128x48x5x5] will be convolved with first part of weights (dimension match!) --> 128 responses
       * same for 2nd part --> 2x128 responses to forward to the next channel
       */
      for (hw = 0; hw < H * W; ++hw) {
        // check which group_ idx we are in
        const int group_idx = n / weight_group_size;

        // offset bottom to batch_idx image
        const int bottom_offset = K * this->group_ * H * W * batch_idx;
        std::complex<Dtype> *bottom_data = this_bottom_data + bottom_offset;

        // offset res to batch_idx image
        const int res_offset = N * H * W * batch_idx;
        std::complex<Dtype> *res_data = this_ptwise_result + res_offset;


        // this loop cannot be parallelized (because it is adding up stuff in the same memory address!
        for (k = 0; k < K; ++k) {
          // get the input_k. this is the k we use to index the input k-dimension. the max input_k is group_ times more
          // than the max k of the weight.
          const int input_k = k + group_idx * K;
          const int weight_offset = (n * K + k);

          // Indexing: ((n * K + k) * H + h) * W + w
          const int input_idx = input_k * H * W + hw;
          const int weight_idx = weight_offset * H * W + hw;
          const int res_idx = n * H * W + hw;

          const std::complex<Dtype> single_input = bottom_data[input_idx];
          const std::complex<Dtype> single_weight = weight_complex[weight_idx];
          std::complex<Dtype> *single_res = res_data + res_idx;

          *single_res += fft_cpu_multiply_complex(single_input, single_weight, true);
        }
      }
    }
  }
}

template <typename Dtype>
void ConvolutionLayerFFT<Dtype>::fft_pointwise_multiply_backward_cpu() {
  const int batch_size = this->num_;            //              10
  const int N = this->num_output_;              //              256
  const int K = this->channels_ / this->group_; // 96 / 2     = 48
  const int H = this->fft_height_;              //              32 ?
  const int W = (this->fft_width_ / 2) + 1;     // 32 / 2 + 1 = 17

  const int weight_group_size = N / this->group_;

  int batch_idx = 0;
  int n = 0;
  int k = 0;
  int hw = 0;

  std::complex<Dtype> *weight_complex = this->ffted_weights_;
  std::complex<Dtype> *this_bottom_data = this->ffted_bottom_data_;
  std::complex<Dtype> *this_ptwise_result = this->ptwise_result_;

#ifdef _OPENMP
#pragma omp parallel for private(batch_idx, hw, k, n) collapse(3)
#endif
  for (batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
    for (hw = 0; hw < H * W; ++hw) {
      for (k = 0; k < K; ++k) {
        const int bottom_offset = K * this->group_ * H * W * batch_idx;
        std::complex<Dtype> *bottom_data = this_bottom_data + bottom_offset;

        const int res_offset = N * H * W * batch_idx;
        std::complex<Dtype> *res_data = this_ptwise_result + res_offset;

        // this loop cannot be parallelized (because it is adding up stuff in the same memory address!
        for (n = 0; n < N; ++n) {
          // check which group_ idx we are in
          const int group_idx = n / weight_group_size;

          const int input_k = k + group_idx * K;
          const int weight_offset = (n * K + k);

          const int input_idx = input_k * H * W + hw;
          const int weight_idx = weight_offset * H * W + hw;
          const int res_idx = n * H * W + hw;

          const std::complex<Dtype> single_input = res_data[res_idx];
          const std::complex<Dtype> single_weight = weight_complex[weight_idx];
          std::complex<Dtype> *single_res = bottom_data + input_idx;

          *single_res += fft_cpu_multiply_complex(single_input, single_weight, false);
        }
      }
    }
  }
}

template <typename Dtype>
void ConvolutionLayerFFT<Dtype>::fft_pointwise_multiply_weight_cpu() {
  // here it is summed up over the batch size not k!!!
  const int batch_size = this->num_;            //              10
  const int N = this->num_output_;              //              256
  const int K = this->channels_ / this->group_; // 96 / 2     = 48
  const int H = this->fft_height_;              //              32 ?
  const int W = (this->fft_width_ / 2) + 1;     // 32 / 2 + 1 = 17

  const int weight_group_size = N / this->group_;

  int batch_idx = 0;
  int n = 0;
  int k = 0;
  int hw = 0;

  std::complex<Dtype> *res_ptr = this->ffted_weights_;

#ifdef _OPENMP
#pragma omp parallel for private(n, hw, k, batch_idx) collapse(3)
#endif
  for (n = 0; n < N; ++n) {
    for (hw = 0; hw < H * W; ++hw) {
      for (k = 0; k < K; ++k) {
        const int group_idx = n / weight_group_size;
        // this loop cannot be parallelized (because it is adding up stuff in the same memory address!
        for (batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
          const int bottom_offset = K * this->group_ * H * W * batch_idx;
          std::complex<Dtype> *bottom_data = this->ffted_bottom_data_ + bottom_offset;

          const int top_offset = N * H * W * batch_idx;
          std::complex<Dtype> *top_data = this->ptwise_result_ + top_offset;

          const int input_k = k + group_idx * K;
          const int input_idx = input_k * H * W + hw;
          // top is filter here!
          const int top_idx = n * H * W + hw;
          // weight is result here!!
          const int weight_offset = (n * K + k);
          const int weight_idx = weight_offset * H * W + hw;

          const std::complex<Dtype> single_input = bottom_data[input_idx];
          const std::complex<Dtype> single_top = top_data[top_idx];
          std::complex<Dtype> *single_res = res_ptr + weight_idx;

          *single_res += fft_cpu_multiply_complex(single_input, single_top, true);
        }
      }
    }
  }
}

#ifdef USE_IPP
template <typename Dtype>
void ConvolutionLayerFFT<Dtype>::fft_pointwise_multiply_ipp_cpu() {
  const int N = this->num_output_;
  const int K = this->channels_ / this->group_;
  const int weight_group_size = N / this->group_;

  int n = 0;
  int k = 0;

  const std::complex<Dtype> *bottom_complex = this->ffted_bottom_data_;
  std::complex<Dtype> *weight_complex = this->ffted_weights_;

  //#ifdef _OPENMP
  //#pragma omp parallel for private(n, k) shared(bottom_complex, weight_complex, ptwise_result)
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

      // pSrcDst[n ] = pSrcDst[n ] + pSrc1[n ] * pSrc2[n ], 0 ≤ n < len.
      ipp_complex_add_product<Dtype>(bottom_complex + input_offset, weight_complex + weight_offset, this->ptwise_result_ + res_offset, this->fft_complex_size_);
    }
  }
}
#endif

template <typename Dtype>
cgemm_sizes ConvolutionLayerFFT<Dtype>::fft_pointwise_multiply_gemm_init_cpu(PASS_TYPE pass_type) {
  cgemm_sizes sizes;
  sizes.num = this->num_;
  sizes.num_output = this->num_output_;
  sizes.channels = this->channels_;

  sizes.H = this->fft_height_;
  sizes.W = (this->fft_width_ / 2) + 1;
  sizes.G = this->group_;

  switch (pass_type) {
    case FORWARD:
      //                      num_output * channels / group = 96 * 3
      sizes.weight_size = this->num_output_ * (this->channels_ / sizes.G);
      //                      num_images      * channels / G   =  1 * 3
      // if G = 2, the size is half but there are two of bottom data blob behind each other in memory. e.g.:
      // batch size = 3: three times group 1 and then three times group 2: 123|123 for each input.
      sizes.bottom_size = (this->num_ * this->channels_);

      //                      num_output * num_images       = 96 * 1
      sizes.output_size = (this->num_output_ * this->num_);

      sizes.group_offset_input = sizes.bottom_size / this->num_ / sizes.G;
      sizes.group_offset_weight = sizes.weight_size / sizes.G;
      sizes.group_offset_output = sizes.output_size / this->num_ / sizes.G;
      break;
    case BACKWARD:
      sizes.weight_size = this->num_output_ * (this->channels_ / sizes.G);
      // bottom and output sizes are reversed in backward pass...
      sizes.bottom_size = (this->num_output_ * this->num_);
      sizes.output_size = (this->num_ * this->channels_);

      sizes.group_offset_input = sizes.bottom_size / this->num_ / sizes.G;
      sizes.group_offset_weight = sizes.weight_size / sizes.G;
      sizes.group_offset_output = sizes.output_size / this->num_ / sizes.G;
      break;
    case WEIGHT:
      sizes.weight_size = (this->num_ * this->channels_);
      // bottom and output sizes are reversed in backward pass...
      sizes.bottom_size = (this->num_output_ * this->num_);
      sizes.output_size = this->num_output_ * (this->channels_ / sizes.G);

      sizes.group_offset_input = sizes.bottom_size / this->num_ / sizes.G;
      sizes.group_offset_weight = sizes.weight_size / this->num_ / sizes.G;
      sizes.group_offset_output = sizes.output_size  / sizes.G;
      break;
    default:
      break;
  }

  return sizes;
}

#ifdef USE_MKL
// the batched gemm calls are only available in mkl. TODO: implement loop without batching??? poor performance!

template <typename Dtype>
void ConvolutionLayerFFT<Dtype>::fft_pointwise_multiply_gemm_cpu() {
  // alloc data for result
  size_t alloc_size = this->convolution_result_complex_size_ ;

  std::complex<Dtype> *fft_transposed_result =
      reinterpret_cast<std::complex<Dtype> *>(fft_cpu_malloc<Dtype>(alloc_size));

  cgemm_sizes sizes = this->fft_pointwise_multiply_gemm_init_cpu(FORWARD);

  const std::complex<Dtype> one_complex(1., 0.);
  const std::complex<Dtype> zero_complex(0., 0.);

  const int M = this->num_;
  const int N = this->num_output_ / sizes.G;
  const int K = this->channels_ / sizes.G;

  int array_size = sizes.H * sizes.W * sizes.G;
  const std::complex<Dtype> **weight_arr = new const std::complex<Dtype> *[array_size];
  const std::complex<Dtype> **input_arr = new const std::complex<Dtype> *[array_size];
  std::complex<Dtype> **output_arr = new std::complex<Dtype> *[array_size];


  this->fft_pointwise_multiply_gemm_construct_array_cpu(this->ffted_weights_, this->ffted_bottom_data_,
                                                        fft_transposed_result, sizes, weight_arr, input_arr, output_arr);

  int lda = K * sizes.G;
  int ldb = K; // because TransB = Trans!
  int ldc = N * sizes.G;

  // Do batched matrix multiplication (conjugate in forward pass)
  caffe_cpu_gemm_complex_batch<Dtype>(CblasNoTrans, CblasConjTrans, M, N, K,
                                      &one_complex, input_arr, weight_arr, &zero_complex, output_arr, array_size, &lda, &ldb, &ldc);

  delete[] weight_arr;
  delete[] input_arr;
  delete[] output_arr;

  std::complex<Dtype> *dst = this->ptwise_result_;
  const int shape_result[] = {sizes.H, sizes.W, this->num_, this->num_output_};

  this->fft_geam_transpose_cpu(fft_transposed_result, dst, shape_result, 2);
  fft_cpu_free<Dtype>(fft_transposed_result);
}

template <typename Dtype>
void ConvolutionLayerFFT<Dtype>::fft_pointwise_multiply_gemm_backward_cpu() {
  // alloc data for result
  size_t alloc_size = this->padded_bottom_complex_size_;

  std::complex<Dtype> * fft_transposed_result =
      reinterpret_cast<std::complex<Dtype> *>(fft_cpu_malloc<Dtype>(alloc_size));

  cgemm_sizes sizes = this->fft_pointwise_multiply_gemm_init_cpu(BACKWARD);

  const std::complex<Dtype> one_complex(1., 0.);
  const std::complex<Dtype> zero_complex(0., 0.);

  // N and K are reversed in backward pass...
  const int M = this->num_;
  const int N = this->channels_ / sizes.G;
  const int K = this->num_output_ / sizes.G;

  int array_size = sizes.H * sizes.W * sizes.G;
  const std::complex<Dtype> **weight_arr = new const std::complex<Dtype> *[array_size];
  const std::complex<Dtype> **input_arr = new const std::complex<Dtype> *[array_size];
  std::complex<Dtype> **output_arr = new std::complex<Dtype> *[array_size];

  // in the backward pass the input is ptwise_result! also output_size and bottom_size are reversed...
  this->fft_pointwise_multiply_gemm_construct_array_cpu(this->ffted_weights_, this->ptwise_result_,
                                                        fft_transposed_result, sizes, weight_arr, input_arr, output_arr);

  int lda = K * sizes.G; // because TransA = NoTrans!
  int ldb = N; // because TransB = NoTrans!
  int ldc = N * sizes.G;

  // Do batched matrix multiplication (BW: NoTrans and Convolution [vs. conjugate/cross-correlation in forward pass])
  caffe_cpu_gemm_complex_batch<Dtype>(CblasNoTrans, CblasNoTrans, M, N, K,
                                      &one_complex, input_arr, weight_arr, &zero_complex, output_arr, array_size, &lda, &ldb, &ldc);

  delete[] weight_arr;
  delete[] input_arr;
  delete[] output_arr;

  // The destination in backward pass is bottom_data (input in fw pass)
  std::complex<Dtype> *dst = this->ffted_bottom_data_;
  const int shape_result[] = {sizes.H, sizes.W, this->num_, this->channels_};
  this->fft_geam_transpose_cpu(fft_transposed_result, dst, shape_result, 2);
  fft_cpu_free<Dtype>(fft_transposed_result);
}

template <typename Dtype>
void ConvolutionLayerFFT<Dtype>::fft_pointwise_multiply_gemm_weight_cpu() {
  // alloc data for result
  size_t alloc_size = this->padded_weights_complex_size_;

  std::complex<Dtype> * fft_transposed_result =
      reinterpret_cast<std::complex<Dtype> *>(fft_cpu_malloc<Dtype>(alloc_size));

  cgemm_sizes sizes = this->fft_pointwise_multiply_gemm_init_cpu(WEIGHT);

  const std::complex<Dtype> one_complex(1., 0.);
  const std::complex<Dtype> zero_complex(0., 0.);

  // In weight pass: M = N_old, N = K_old, K = M_old
  const int M = this->num_output_ / sizes.G;
  const int N = this->channels_ / sizes.G;
  const int K = this->num_;

  int array_size = sizes.H * sizes.W * sizes.G;
  const std::complex<Dtype> **weight_arr = new const std::complex<Dtype> *[array_size];
  const std::complex<Dtype> **input_arr = new const std::complex<Dtype> *[array_size];
  std::complex<Dtype> **output_arr = new std::complex<Dtype> *[array_size];

  // in the backward pass the input is ptwise_result! also output_size and bottom_size are reversed...
  this->fft_pointwise_multiply_gemm_construct_array_cpu(this->ffted_bottom_data_, this->ptwise_result_,
                                                        fft_transposed_result, sizes, weight_arr, input_arr, output_arr);

  int lda = M * sizes.G;  // because TransA = Trans!
  int ldb = N * sizes.G;
  int ldc = N;

  // Do batched matrix multiplication (W: ....??)
  caffe_cpu_gemm_complex_batch<Dtype>(CblasConjTrans, CblasNoTrans, M, N, K,
                                      &one_complex, input_arr, weight_arr, &zero_complex, output_arr, array_size, &lda, &ldb, &ldc);

  delete[] weight_arr;
  delete[] input_arr;
  delete[] output_arr;

  // The destination in backward pass is bottom_data (input in fw pass)
  std::complex<Dtype> *dst = this->ffted_weights_;
  const int shape_result[] = {sizes.H, sizes.W, this->num_output_, this->channels_ / this->group_};
  this->fft_geam_transpose_cpu(fft_transposed_result, dst, shape_result, 2);
  fft_cpu_free<Dtype>(fft_transposed_result);
}

template <typename Dtype>
void ConvolutionLayerFFT<Dtype>::fft_pointwise_multiply_gemm_construct_array_cpu(const std::complex<Dtype> *weight_complex,
                                                                                 const std::complex<Dtype> *bottom_complex,
                                                                                 std::complex<Dtype> *fft_transposed_result,
                                                                                 cgemm_sizes sizes,
                                                                                 const std::complex<Dtype> **weight_arr,
                                                                                 const std::complex<Dtype> **input_arr,
                                                                                 std::complex<Dtype> **output_arr) {
  int idx = 0;

  for (int h = 0; h < sizes.H; ++h) {
    for (int w = 0; w < sizes.W; ++w) {
      const std::complex<Dtype> *weight = weight_complex + (h * sizes.W + w ) * sizes.weight_size;
      const std::complex<Dtype> *input = bottom_complex + (h * sizes.W + w ) * sizes.bottom_size;
      std::complex<Dtype> *output = fft_transposed_result + (h * sizes.W + w ) * sizes.output_size;

      for (int g = 0; g < sizes.G; ++g) {
        weight_arr[idx] = weight + g * sizes.group_offset_weight;
        input_arr[idx] = input + g * sizes.group_offset_input;
        output_arr[idx] = output + g * sizes.group_offset_output;
        idx++;
      }
    }
  }
}

#endif

template <typename Dtype>
void ConvolutionLayerFFT<Dtype>::fft_normalize_cpu(std::vector<int> shape, const int stride_h, const int stride_w,
                                                   const int pad_h, const int pad_w,
                                                   std::complex<Dtype> *ffted_result,
                                                   Dtype *iffted_result, Dtype *result,
                                                   bool add_to_result) {
  const int N = shape[0];
  const int K = shape[1];
  const int H = shape[2];
  const int W = shape[3];

  int fft_batch_size = N * K;
  if (this->fft_inplace_) {
    iffted_result = reinterpret_cast<Dtype *>(ffted_result);
  }

  this->ifft_plan_ = fft_cpu_plan_many_dft_c2r_2d<Dtype>(this->fft_height_,
                                                         this->fft_width_,
                                                         fft_batch_size,
                                                         ffted_result,
                                                         iffted_result,
                                                         FFTW_ESTIMATE);

  fft_cpu_execute_plan<Dtype>(this->ifft_plan_);
  fft_cpu_free<Dtype>(this->ifft_plan_);
  this->ifft_plan_ = NULL;

  // if inplace take fft_complex_size * 2 because complex has double the size [sizeof(std::complex)]
  int fft_width = this->fft_inplace_ ? (this->fft_width_ / 2 + 1) * 2 : this->fft_width_;
  int fft_size = this->fft_height_ * fft_width;

  // here the stride handling and FFT normalization is happening:
  Dtype ifft_normalize_factor = 1. / (this->fft_width_ * this->fft_height_);

#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
  for (int n = 0; n < N; ++n) {
    for (int k = 0; k < K; ++k) {
      const int fft_res_real_offset = (n * K + k) * fft_size;

      for (int h = 0; h < H; ++h) // =55 in 1st layer
      {
        for (int w = 0; w < W; ++w) // =55 in 1st layer
        {
          // The first valid result in the real conv array is @ pad_h. Every stride steps
          // is the next valid result!
          int h_idx = pad_h + h * stride_h;
          int w_idx = pad_w + w * stride_w;

          //((n * K + k) * H + h) * W + w;
          const int result_idx = ((n * K + k) * H + h) * W + w;

          // the index in the data of the convolution result array (the real one)
          const int fft_result_real_idx = fft_res_real_offset + h_idx * fft_width + w_idx;

          // normalize fft and sum up everything from the input channels...
          Dtype tmp_result = iffted_result[fft_result_real_idx] * ifft_normalize_factor;
          if (add_to_result) {
            result[result_idx] += tmp_result;
          } else {
            result[result_idx] = tmp_result;
          }
        }
      }
    }
  }
}

template <typename Dtype>
void ConvolutionLayerFFT<Dtype>::pad_real_blob(std::vector<int> shape, const Dtype *blob_data, Dtype *padded_data,
                                               int pad_h, int pad_w, int stride_h, int stride_w, bool inplace) {
  const int N = shape[0];
  const int K = shape[1];
  const int H = shape[2];
  const int W = shape[3];

  int size = N * K; // # of arrays (for weights it is num_weights [96 x 3]
  // for input data it is channels [ 1 x 3]

  // if inplace take fft_complex_size * 2 because complex has double the size [sizeof(std::complex)]
  int fft_width = inplace ? (this->fft_width_ / 2 + 1) * 2 : this->fft_width_;
  int fft_size = this->fft_height_ * fft_width;
  size = size * fft_size;

  // set everything to 0 before --> so not set weights are 0-padded :)
  caffe_memset(size * sizeof(Dtype), 0., padded_data);
  //#ifdef _OPENMP
  //#pragma omp parallel for
  //#endif
  for (int n = 0; n < N; n++) {
    for (int k = 0; k < K; k++) {
      for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
          // ((n * ch_gr + c) * fft_height_ + h)* 2 * (fft_width_ / 2 + 1) + w
          const int offset_weight_real = (n * K + k) * fft_size;
          // e.g. a 3x3 filter should fit into a 5x5 because the image size is 5x5 (here the stride is 1)
          // <--W-->
          // ^ f f f 0 0
          // H f f f 0 0
          // _ f f f 0 0
          //   0 0 0 0 0
          //   0 0 0 0 0

          const int idx_weight_real = offset_weight_real + (h * stride_h + pad_h) * fft_width + (w  * stride_w + pad_w);
          // copy each weight into the fft_weights_in_real_
          // get ptr to blob data. indexing see: http://caffe.berkeleyvision.org/tutorial/net_layer_blob.html
          // Blob memory is row-major in layout, so the last / rightmost dimension changes fastest. For example,
          // in a 4D blob, the value at index (n, k, h, w) is physically located at index ((n * K + k) * H + h) * W + w.
          // 96 x 3 x 11 x 11 (num_output_, channels_ / group_, kernel_height, width_)

          const int idx_weight_in_blob = ((n * K + k) * H + h) * W + w;

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

template <typename Dtype>
void ConvolutionLayerFFT<Dtype>::Forward_gpu_fft(const vector<Blob<Dtype>*>& bottom,
                                                 const vector<Blob<Dtype>*>& top) { NO_GPU; }

template <typename Dtype>
void ConvolutionLayerFFT<Dtype>::Forward_gpu_normal(const vector<Blob<Dtype>*>& bottom,
                                                    const vector<Blob<Dtype>*>& top) { NO_GPU; }

template <typename Dtype>
void ConvolutionLayerFFT<Dtype>::Forward_gpu_fft_single(const Dtype *bottom, Dtype *top) { NO_GPU; }

template <typename Dtype>
void ConvolutionLayerFFT<Dtype>::Backward_gpu_fft(const Dtype* input, Dtype* output) { NO_GPU; }

template <typename Dtype>
void ConvolutionLayerFFT<Dtype>::Weight_gpu_fft(const Dtype* input, const Dtype* output, Dtype* weight) { NO_GPU; }

template <typename Dtype>
void ConvolutionLayerFFT<Dtype>::fft_set_up_gpu() { NO_GPU; }

template <typename Dtype>
void ConvolutionLayerFFT<Dtype>::fft_update_weights_gpu() { NO_GPU; }

template <typename Dtype>
void ConvolutionLayerFFT<Dtype>::fft_free_weights_gpu() { NO_GPU; }

template <typename Dtype>
void ConvolutionLayerFFT<Dtype>::fft_bottom_gpu(const Dtype *bottom) { NO_GPU; }

template <typename Dtype>
void ConvolutionLayerFFT<Dtype>::fft_top_gpu(const Dtype *top) { NO_GPU; }

template <typename Dtype>
void ConvolutionLayerFFT<Dtype>::fft_convolve_gpu(Dtype *top) { NO_GPU; }

template <typename Dtype>
void ConvolutionLayerFFT<Dtype>::fft_convolve_backward_gpu(Dtype *bottom) { NO_GPU; }

template <typename Dtype>
void ConvolutionLayerFFT<Dtype>::fft_convolve_weight_gpu(Dtype *weight) { NO_GPU; }

template <typename Dtype>
void ConvolutionLayerFFT<Dtype>::fft_pointwise_multiply_gpu() { NO_GPU; }

template <typename Dtype>
void ConvolutionLayerFFT<Dtype>::fft_pointwise_multiply_backward_gpu() { NO_GPU; }

template <typename Dtype>
void ConvolutionLayerFFT<Dtype>::fft_pointwise_multiply_weight_gpu() { NO_GPU; }

template <typename Dtype>
void ConvolutionLayerFFT<Dtype>::fft_pointwise_multiply_gemm_gpu() { NO_GPU; }

template <typename Dtype>
void ConvolutionLayerFFT<Dtype>::fft_pointwise_multiply_gemm_backward_gpu() {NO_GPU; }

template <typename Dtype>
void ConvolutionLayerFFT<Dtype>::fft_pointwise_multiply_gemm_weight_gpu() {NO_GPU; }

template <typename Dtype>
void ConvolutionLayerFFT<Dtype>::fft_normalize_gpu(Dtype *top_data) { NO_GPU; }

template <typename Dtype>
void ConvolutionLayerFFT<Dtype>::fft_normalize_backward_gpu(Dtype *bottom) { NO_GPU; }

template <typename Dtype>
void ConvolutionLayerFFT<Dtype>::fft_normalize_weight_gpu(Dtype *weight) { NO_GPU; }

template <typename Dtype>
void ConvolutionLayerFFT<Dtype>::mem_info_gpu() { NO_GPU; }
#endif

INSTANTIATE_CLASS(ConvolutionLayerFFT);

}  // namespace caffe
#endif /* USE_FFT */
