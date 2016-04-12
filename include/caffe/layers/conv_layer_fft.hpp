#ifndef CONV_LAYER_FFT_HPP_
#define CONV_LAYER_FFT_HPP_

#ifdef USE_FFT

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/base_conv_layer.hpp"
#include "caffe/layers/conv_layer.hpp"

#include "caffe/util/fft_util.hpp"
#include <complex>

namespace caffe {

/* FFT convolution kind defines */

#define FFT_CONVOLUTION_KIND_POINTWISE_IPP 0
#define FFT_CONVOLUTION_KIND_POINTWISE_SIMPLE 1
#define FFT_CONVOLUTION_KIND_CGEMM 2

#define FFT_CONVOLUTION_KIND FFT_CONVOLUTION_KIND_CGEMM

template <typename Dtype>
class ConvolutionLayerFFT : public ConvolutionLayer<Dtype> {
 public:
  explicit ConvolutionLayerFFT(const LayerParameter& param)
      : ConvolutionLayer<Dtype>(param),
        fft_initialized_(false),
        fft_cpu_initialized_(false),
        fft_gpu_initialized_(false),
        fft_on_(true) {}
  virtual ~ConvolutionLayerFFT();

  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);

  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                       const vector<Blob<Dtype>*>& top);

  /**
   * @brief Updates a layer before a new batch. Needed for updating weights in the FFT-Layer
   */
  virtual void LayerUpdateBeforeBatch();

  enum PASS_TYPE { FORWARD, BACKWARD, WEIGHT };

 protected:
  /**
   * Generic FFT Stuff:
   */
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  virtual void Backward_cpu_fft(const Dtype* input, Dtype* output);

  virtual void Weight_cpu_fft(const Dtype* input, const Dtype* output, Dtype* weight);

  virtual void Forward_cpu_fft(const vector<Blob<Dtype>*>& bottom,
                               const vector<Blob<Dtype>*>& top);

  virtual void Forward_cpu_normal(const vector<Blob<Dtype>*>& bottom,
                                  const vector<Blob<Dtype>*>& top);

  virtual void Forward_cpu_fft_single(const Dtype *bottom, Dtype *top);

  virtual void fft_set_up();

  virtual void fft_free_weights_cpu();

  virtual void pad_real_blob(std::vector<int> shape, const Dtype *blob_data, Dtype *padded_data,
                             int pad_h = 0, int pad_w = 0, int stride_h = 1, int stride_w = 1, bool inplace = false);

  virtual void fft_set_up_cpu();

  virtual void fft_update_weights_cpu();

  virtual void fft_permute_4d_cpu(const std::complex<Dtype> *in, std::complex<Dtype> *out,
                                  const int shape[4], const int permutation[4]);

  virtual void fft_geam_transpose_cpu(const std::complex<Dtype> *in, std::complex<Dtype> *out,
                                      const int shape[4], const int sep);

  virtual void fft_bottom_cpu(const Dtype *bottom);

  virtual void fft_top_cpu(const Dtype *top);

  virtual void fft_convolve_cpu(Dtype *top);

  virtual void fft_convolve_backward_cpu(Dtype *bottom);

  virtual void fft_convolve_weight_cpu(Dtype *weight);

  /**
   * Pointwise multiplication via loops
   */

  virtual void fft_pointwise_multiply_cpu();

  virtual void fft_pointwise_multiply_backward_cpu();

  virtual void fft_pointwise_multiply_weight_cpu();

#ifdef USE_IPP
  /**
   * Pointwise multiplication via IPP (SLOW; depreceated)
   */

  virtual void fft_pointwise_multiply_ipp_cpu();
#endif

#ifdef USE_MKL
  /**
   * Pointwise multiplication via gemm
   */

  virtual void fft_pointwise_multiply_gemm_cpu();

  virtual void fft_pointwise_multiply_gemm_backward_cpu();

  virtual void fft_pointwise_multiply_gemm_weight_cpu();

  virtual void fft_pointwise_multiply_gemm_construct_array_cpu(const std::complex<Dtype> *weight_complex,
                                                               const std::complex<Dtype> *bottom_complex,
                                                               std::complex<Dtype> *fft_transposed_result,
                                                               cgemm_sizes sizes,
                                                               const std::complex<Dtype> **weight_arr,
                                                               const std::complex<Dtype> **input_arr,
                                                               std::complex<Dtype> **output_arr);
#endif

  // this method is also used for the gpu so dont exlude it if there's no mkl!
  virtual cgemm_sizes fft_pointwise_multiply_gemm_init_cpu(PASS_TYPE pass_type);

  virtual void fft_normalize_cpu(std::vector<int> shape, const int stride_h, const int stride_w,
                                 const int pad_h, const int pad_w,
                                 std::complex<Dtype> *ffted_result,
                                 Dtype *iffted_result, Dtype *result,
                                 bool add_to_result);

  /**
   * Helper stuff
   */

  virtual void write_arr_to_disk(const char* output_name, size_t size, void *arr, bool is_complex = false);

  /**
   * FFT GPU Stuff:
   */

  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);

  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);


  virtual void Backward_gpu_fft(const Dtype* input, Dtype* output);

  virtual void Weight_gpu_fft(const Dtype* input, const Dtype* output, Dtype* weight);

  virtual void Forward_gpu_fft(const vector<Blob<Dtype>*>& bottom,
                               const vector<Blob<Dtype>*>& top);

  virtual void Forward_gpu_normal(const vector<Blob<Dtype>*>& bottom,
                                  const vector<Blob<Dtype>*>& top);

  virtual void Forward_gpu_fft_single(const Dtype *bottom, Dtype *top);

  virtual void fft_set_up_gpu();

  virtual void fft_update_weights_gpu();

  virtual void fft_free_weights_gpu();

  virtual void fft_bottom_gpu(const Dtype *bottom);

  virtual void fft_top_gpu(const Dtype *top);

  virtual void fft_convolve_gpu(Dtype *top);

  virtual void fft_convolve_backward_gpu(Dtype *bottom);

  virtual void fft_convolve_weight_gpu(Dtype *weight);

  virtual void fft_pointwise_multiply_gpu();

  virtual void fft_pointwise_multiply_backward_gpu();

  virtual void fft_pointwise_multiply_weight_gpu();

  virtual void fft_pointwise_multiply_gemm_gpu();

  virtual void fft_pointwise_multiply_gemm_backward_gpu();

  virtual void fft_pointwise_multiply_gemm_weight_gpu();

  virtual void fft_normalize_gpu(Dtype *top_data);

  virtual void fft_normalize_backward_gpu(Dtype *bottom);

  virtual void fft_normalize_weight_gpu(Dtype *weight);

  virtual void mem_info_gpu();

  /**
   * FFT specific fields:
   */
  bool fft_initialized_;
  bool fft_cpu_initialized_;
  bool fft_gpu_initialized_;
  bool fft_on_;

  int fft_height_;
  int fft_width_;
  int fft_complex_size_;
  int fft_real_size_;

  bool fft_update_weights_each_batch_;
  bool fft_inplace_;

  // Allocation sizes:
  size_t padded_weights_real_size_;
  size_t padded_weights_complex_size_;
  size_t padded_bottom_real_size_;
  size_t padded_bottom_complex_size_;
  size_t convolution_result_real_size_;
  size_t convolution_result_complex_size_;

  // Pointers to weight memory...
  std::complex<Dtype> *ffted_weights_;

  int num_threads_;
  int num_weights_;
  std::vector<int> top_shape_;


  // Plans and in memory values. It is defined globally so no free and malloc has to be called all the time...
  // Both different for CPU/GPU
  void *fft_bottom_plan_;
  void *fft_top_plan_;
  void *ifft_plan_;

  std::complex<Dtype> *ptwise_result_;
  Dtype *fft_convolution_result_real_;
  Dtype *padded_real_bottom_;
  std::complex<Dtype> *ffted_bottom_data_;
#ifndef CPU_ONLY
  /* GPU Stuff */
  cufftHandle fft_weight_plan_;
  cufftHandle fft_bottom_plan_gpu_;
  cufftHandle fft_top_plan_gpu_;
  cufftHandle ifft_plan_gpu_;
  cufftHandle ifft_backward_plan_gpu_;
  cufftHandle ifft_weight_plan_gpu_;

  bool fft_top_plan_gpu__initialized_;
  bool ifft_backward_plan_gpu_initialized_;
  bool ifft_weight_plan_gpu_initialized_;


  std::complex<Dtype> *ptwise_result_gpu_;
  Dtype *fft_convolution_result_real_gpu_;
  Dtype *padded_real_bottom_gpu_;
  std::complex<Dtype> *ffted_bottom_data_gpu_;
#endif
};
}



#endif /* USE_FFT */
#endif /* CONV_LAYER_FFT_HPP_ */
