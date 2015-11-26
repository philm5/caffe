#include <algorithm>

#include <cmath>
#include <cstdlib>
#include <cstring>

#include "caffe/common.hpp"
#include "caffe/util/fft_util.hpp"
#include "caffe/util/device_alternate.hpp"
#include <npp.h>

namespace caffe {
template <typename Dtype>
__global__ void pad_real_blob_gpu_kernel(const int K, const int H, const int W, const int fft_height, const int fft_width, const Dtype *blob_data, Dtype *padded_data,
                                         const int pad_h, const int pad_w, const int stride_h, const int stride_w, bool inplace) {

  // blockDim (256, 1) ----- (num_output_, (ch_gr) / CUDA_NUM_THREADS)
  //                              x                y
  const int n = blockIdx.x;

  // calculate the channel index. The blockIdx.y is usually zero. Because CUDA_THREADS = 512 > 48 = ch/gr. (48 / 512) + 1 = 1.
  const int hw = blockIdx.y * blockDim.x + threadIdx.x;


  if (hw < H*W) {
    // if inplace take fft_complex_size * 2 because complex has double the size [sizeof(std::complex)]
    int fft_size = inplace ? fft_height * (fft_width / 2 + 1) * 2 : fft_height * fft_width;

    for (int k = 0; k < K; ++k) {
      // get offset with channels and the idx of the output.
      const int offset_weight_real = (n * K + k) * fft_size;
      const int offset_blob_real = (n * K + k) * H * W;
      const int h = hw / W;
      const int w = hw % W;

      const int idx_weight_real = offset_weight_real + (h * stride_h + pad_h) * fft_width + (w * stride_w + pad_w);
      // copy each weight into the fft_weights_in_real_
      // get ptr to blob data. indexing see: http://caffe.berkeleyvision.org/tutorial/net_layer_blob.html
      // Blob memory is row-major in layout, so the last / rightmost dimension changes fastest. For example,
      // in a 4D blob, the value at index (n, k, h, w) is physically located at index ((n * K + k) * H + h) * W + w.
      // 96 x 3 x 11 x 11 (num_output_, channels_ / group_, kernel_height, width_)
      const int idx_weight_in_blob = offset_blob_real + h * W + w;

      Dtype data_in_blob = blob_data[idx_weight_in_blob];
      padded_data[idx_weight_real] = data_in_blob;

    }
  }
}

template __global__ void pad_real_blob_gpu_kernel<float>(const int K, const int H, const int W, const int fft_height, const int fft_width,
                                                         const float *blob_data, float *padded_data,
                                                         const int pad_h, const int pad_w, const int stride_h, const int stride_w, bool inplace);

template __global__ void pad_real_blob_gpu_kernel<double>(const int K, const int H, const int W, const int fft_height, const int fft_width,
                                                          const double *blob_data, double *padded_data,
                                                          const int pad_h, const int pad_w, const int stride_h, const int stride_w, bool inplace);

__global__ void fft_pointwise_multiply_float_gpu_kernel(const int N, const int K, const int H, const int W,
                                                        const int weight_group_size, const cufftComplex *ffted_bottom_data,
                                                        const cufftComplex *weight_complex, cufftComplex *ptwise_result) {
  // blockDim (10, 256, 2) ----- (batch_size, num_output_, (H*W) / CUDA_NUM_THREADS)
  //                              x                y
  const int batch_idx = blockIdx.x;
  const int n = blockIdx.y;

  // calculate the channel index. blockIdx.y is of size (H*W/CUDA_NUM_THREADS). So 1 or 2...
  const int hw = blockIdx.z * blockDim.x + threadIdx.x;


  if (hw < H*W) {
    // check which group_idx we are in
    const int group_idx = n / weight_group_size;
    const int group = (N / weight_group_size);

    // offset bottom to batch_idx image
    const int bottom_offset = K * group * H * W * batch_idx;
    const cufftComplex *bottom_data = ffted_bottom_data + bottom_offset;

    // offset res to batch_idx image
    const int res_offset = N * H * W * batch_idx;
    cufftComplex *res_data = ptwise_result + res_offset;

    // loop over channels
    for (int k = 0; k < K; ++k) {
      // get the input_k. this is the k we use to index the input k-dimension. the max input_k is group_ times more
      // than the max k of the weight.
      const int input_k = k + group_idx * K;
      const int weight_offset = (n * K + k);

      const int input_idx = input_k * H * W + hw;
      const cufftComplex input = bottom_data[input_idx];

      const int weight_idx = weight_offset * H * W + hw;
      const cufftComplex weight = weight_complex[weight_idx];

      const int res_idx = n * H * W + hw;

      fft_gpu_cmultiply_add(input, weight, res_data + res_idx, true);
    }
  }
}

__global__ void fft_pointwise_multiply_double_gpu_kernel(const int N, const int K, const int H, const int W,
                                                         const int weight_group_size, const cufftDoubleComplex *ffted_bottom_data,
                                                         const cufftDoubleComplex *weight_complex, cufftDoubleComplex *ptwise_result) {
  // blockDim (10, 256, 2) ----- (batch_size, num_output_, (H*W) / CUDA_NUM_THREADS)
  //                              x                y
  const int batch_idx = blockIdx.x;
  const int n = blockIdx.y;

  // calculate the channel index. blockIdx.y is of size (H*W/CUDA_NUM_THREADS). So 1 or 2...
  const int hw = blockIdx.z * blockDim.x + threadIdx.x;


  if (hw < H*W) {
    // check which group_idx we are in
    const int group_idx = n / weight_group_size;
    const int group = (N / weight_group_size);

    // offset bottom to batch_idx image
    const int bottom_offset = K * group * H * W * batch_idx;
    const cufftDoubleComplex *bottom_data = ffted_bottom_data + bottom_offset;

    // offset res to batch_idx image
    const int res_offset = N * H * W * batch_idx;
    cufftDoubleComplex *res_data = ptwise_result + res_offset;

    // loop over channels
    for (int k = 0; k < K; ++k) {
      // get the input_k. this is the k we use to index the input k-dimension. the max input_k is group_ times more
      // than the max k of the weight.
      const int input_k = k + group_idx * K;
      const int weight_offset = (n * K + k);

      const int input_idx = input_k * H * W + hw;
      const cufftDoubleComplex input = bottom_data[input_idx];

      const int weight_idx = weight_offset * H * W + hw;
      const cufftDoubleComplex weight = weight_complex[weight_idx];

      const int res_idx = n * H * W + hw;

      fft_gpu_zmultiply_add(input, weight, res_data + res_idx, true);
    }
  }
}

__global__ void fft_pointwise_multiply_backward_float_gpu_kernel(const int N, const int K, const int H, const int W,
                                                                 const int weight_group_size, cufftComplex *ffted_bottom_data,
                                                                 const cufftComplex *weight_complex, const cufftComplex *ptwise_result) {

  // blockDim (10, 256, 2) ----- (batch_size, num_output_, (H*W) / CUDA_NUM_THREADS)
  //                              x                y
  const int batch_idx = blockIdx.x;
  const int k = blockIdx.y;

  // calculate the channel index. blockIdx.y is of size (H*W/CUDA_NUM_THREADS). So 1 or 2...
  const int hw = blockIdx.z * blockDim.x + threadIdx.x;


  if (hw < H*W) {
    // check which group_idx we are in
    const int group = (N / weight_group_size);

    // offset bottom to batch_idx image
    const int bottom_offset = K * group * H * W * batch_idx;
    cufftComplex *bottom_data = ffted_bottom_data + bottom_offset;

    // offset res to batch_idx image
    const int res_offset = N * H * W * batch_idx;
    const cufftComplex *res_data = ptwise_result + res_offset;

    // loop over num_output [this loop cannot be parallelized]
    for (int n = 0; n < N; ++n) {
      // check wich group_idx we are in
      const int group_idx = n / weight_group_size;

      const int input_k = k + group_idx * K;
      const int weight_offset = (n * K + k);

      const int input_idx = input_k * H * W + hw;
      const int weight_idx = weight_offset * H * W + hw;
      const int res_idx = n * H * W + hw;


      const cufftComplex single_input = res_data[res_idx];
      const cufftComplex single_weight = weight_complex[weight_idx];

      fft_gpu_cmultiply_add(single_input, single_weight, bottom_data + input_idx, false);
    }
  }
}


__global__ void fft_pointwise_multiply_backward_double_gpu_kernel(const int N, const int K, const int H, const int W,
                                                                  const int weight_group_size, cufftDoubleComplex *ffted_bottom_data,
                                                                  const cufftDoubleComplex *weight_complex, const cufftDoubleComplex *ptwise_result) {


  // blockDim (10, 256, 2) ----- (batch_size, num_output_, (H*W) / CUDA_NUM_THREADS)
  //                              x                y
  const int batch_idx = blockIdx.x;
  const int k = blockIdx.y;

  // calculate the channel index. blockIdx.y is of size (H*W/CUDA_NUM_THREADS). So 1 or 2...
  const int hw = blockIdx.z * blockDim.x + threadIdx.x;


  if (hw < H*W) {
    // check which group_idx we are in
    const int group = (N / weight_group_size);

    // offset bottom to batch_idx image
    const int bottom_offset = K * group * H * W * batch_idx;
    cufftDoubleComplex *bottom_data = ffted_bottom_data + bottom_offset;

    // offset res to batch_idx image
    const int res_offset = N * H * W * batch_idx;
    const cufftDoubleComplex *res_data = ptwise_result + res_offset;

    // loop over num_output [this loop cannot be parallelized]
    for (int n = 0; n < N; ++n) {
      // check wich group_idx we are in
      const int group_idx = n / weight_group_size;

      const int input_k = k + group_idx * K;
      const int weight_offset = (n * K + k);

      const int input_idx = input_k * H * W + hw;
      const int weight_idx = weight_offset * H * W + hw;
      const int res_idx = n * H * W + hw;


      const cufftDoubleComplex single_input = res_data[res_idx];
      const cufftDoubleComplex single_weight = weight_complex[weight_idx];

      fft_gpu_zmultiply_add(single_input, single_weight, bottom_data + input_idx, false);
    }
  }
}

__global__ void fft_pointwise_multiply_weight_float_gpu_kernel(const int batch_size, const int N, const int K, const int H, const int W,
                                                               const int weight_group_size, const cufftComplex *ffted_bottom_data,
                                                               cufftComplex *weight_complex, const cufftComplex *ptwise_result) {
  // blockDim (10, 256, 2) ----- (batch_size, num_output_, (H*W) / CUDA_NUM_THREADS)
  //                              x                y
  const int n = blockIdx.x;
  const int k = blockIdx.y;

  // calculate the channel index. blockIdx.y is of size (H*W/CUDA_NUM_THREADS). So 1 or 2...
  const int hw = blockIdx.z * blockDim.x + threadIdx.x;

  if (hw < H*W) {
    const int group = (N / weight_group_size);
    // check wich group_idx we are in
    const int group_idx = n / weight_group_size;

    const int input_k = k + group_idx * K;
    const int input_idx = input_k * H * W + hw;

    const int top_idx = n * H * W + hw;
    // weight is result here!!
    const int weight_offset = (n * K + k);
    const int weight_idx = weight_offset * H * W + hw;

    // loop over num_output [this loop cannot be parallelized]
    for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
      // offset bottom to batch_idx image
      const int bottom_offset = K * group * H * W * batch_idx;
      const cufftComplex *bottom_data = ffted_bottom_data + bottom_offset;

      // offset res to batch_idx image
      const int top_offset = N * H * W * batch_idx;
      const cufftComplex *top_data = ptwise_result + top_offset;

      const cufftComplex single_input = bottom_data[input_idx];
      const cufftComplex single_top = top_data[top_idx];

      fft_gpu_cmultiply_add(single_input, single_top, weight_complex + weight_idx, true);
    }
  }
}

__global__ void fft_pointwise_multiply_weight_double_gpu_kernel(const int batch_size, const int N, const int K, const int H, const int W,
                                                                const int weight_group_size, const cufftDoubleComplex *ffted_bottom_data,
                                                                cufftDoubleComplex *weight_complex, const cufftDoubleComplex *ptwise_result) {
  // blockDim (10, 256, 2) ----- (batch_size, num_output_, (H*W) / CUDA_NUM_THREADS)
  //                              x                y
  const int n = blockIdx.x;
  const int k = blockIdx.y;

  // calculate the channel index. blockIdx.y is of size (H*W/CUDA_NUM_THREADS). So 1 or 2...
  const int hw = blockIdx.z * blockDim.x + threadIdx.x;

  if (hw < H*W) {
    const int group = (N / weight_group_size);
    // check wich group_idx we are in
    const int group_idx = n / weight_group_size;

    const int input_k = k + group_idx * K;
    const int input_idx = input_k * H * W + hw;

    const int top_idx = n * H * W + hw;
    // weight is result here!!
    const int weight_offset = (n * K + k);
    const int weight_idx = weight_offset * H * W + hw;

    // loop over num_output [this loop cannot be parallelized]
    for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
      // offset bottom to batch_idx image
      const int bottom_offset = K * group * H * W * batch_idx;
      const cufftDoubleComplex *bottom_data = ffted_bottom_data + bottom_offset;

      // offset res to batch_idx image
      const int top_offset = N * H * W * batch_idx;
      const cufftDoubleComplex *top_data = ptwise_result + top_offset;

      const cufftDoubleComplex single_input = bottom_data[input_idx];
      const cufftDoubleComplex single_top = top_data[top_idx];

      fft_gpu_zmultiply_add(single_input, single_top, weight_complex + weight_idx, true);
    }
  }
}

template <typename Dtype>
__global__ void fft_util_normalize_gpu_kernel(const int K, const int H, const int W, int fft_height, int fft_width,
                                              const int stride_h, const int stride_w, const int pad_h, const int pad_w,
                                              Dtype normalize_factor, const Dtype *fft_result_real,
                                              Dtype *result, bool add_to_result) {

  // blockDim (256, 1) ----- (num_output_, (height_out) / CUDA_NUM_THREADS)
  //                              x                y
  const int n = blockIdx.x;
  const int k = blockIdx.y;

  // calculate the height index. The blockIdx.y is usually zero. Because CUDA_THREADS = 512 > 27 = ch/gr. (27 / 512) + 1 = 1.
  const int hw = blockIdx.z * blockDim.x + threadIdx.x;

  if (hw < H*W) {
    const int fft_real_size = fft_height * fft_width;
    const int fft_res_real_offset = (n * K + k) * fft_real_size;
    const int h = hw / W;
    const int w = hw % W;

    // The first valid result in the real conv array is @ pad_h. Every stride steps
    // is the next valid result!
    const int h_idx = pad_h + h * stride_h;
    const int w_idx = pad_w + w * stride_w;

    //((n * K + k) * H + h) * W + w;
    const int result_idx = ((n * K + k) * H + h) * W + w;

    // the index in the data of the convolution result array (the real one)
    const int fft_result_real_idx = fft_res_real_offset + h_idx * fft_width + w_idx;

    // normalize fft and sum up everything from the input channels...
    Dtype tmp_result = fft_result_real[fft_result_real_idx] * normalize_factor;
    if (add_to_result) {
      result[result_idx] += tmp_result;
    } else {
      result[result_idx] = tmp_result;
    }
  }
}

template __global__ void fft_util_normalize_gpu_kernel<float>(const int K, const int H, const int W, int fft_height, int fft_width,
                                                              const int stride_h, const int stride_w, const int pad_h, const int pad_w,
                                                              float normalize_factor, const float *fft_result_real,
                                                              float *result, bool add_to_result);

template __global__ void fft_util_normalize_gpu_kernel<double>(const int K, const int H, const int W, int fft_height, int fft_width,
                                                               const int stride_h, const int stride_w, const int pad_h, const int pad_w,
                                                               double normalize_factor, const double *fft_result_real,
                                                               double *result, bool add_to_result);

template <typename Dtype>
__global__ void fft_util_permute_4d_gpu_kernel(const std::complex<Dtype> *in, std::complex<Dtype> *out,
                                               const int shape[4], const int permutation[4]) {
  const int K = shape[1];              // 3
  const int H = shape[2];              // 256
  const int W = shape[3];              // 129


  // blockDim (256, 1) ----- (num_output_, (height_out) / CUDA_NUM_THREADS)
  //                              x                y
  int n = blockIdx.x;

  // calculate the height index. The blockIdx.y is usually zero. Because CUDA_THREADS = 512 > 27 = ch/gr. (27 / 512) + 1 = 1.
  const int hw = blockIdx.y * blockDim.x + threadIdx.x;

  if (hw < H*W) {
    // define the indexes for the transposed version, so the data can be transposed very easily:
    const int K_T = shape[permutation[1]];
    const int H_T = shape[permutation[2]];
    const int W_T = shape[permutation[3]];

    int h = hw / W;
    int w = hw % W;

    int k = 0;

    int *vars[] = {&n, &k, &h, &w};

    const int *n_t = vars[permutation[0]];
    int *k_t = vars[permutation[1]];
    const int *h_t = vars[permutation[2]];
    const int *w_t = vars[permutation[3]];

    for (k = 0; k < K; ++k) {
      const int nk_idx_nt = (n * K + k) * H;
      const int nkh_idx_nt = (nk_idx_nt + h ) * W;
      const int idx_nt = nkh_idx_nt + w;
      // alter indexing for t_idx
      const int idx_t = ((*n_t * K_T + *k_t) * H_T + *h_t) * W_T + *w_t;
      out[idx_t] = in[idx_nt];
    }
  }
}

template __global__ void fft_util_permute_4d_gpu_kernel<float>(const std::complex<float> *in, std::complex<float> *out,
                                                               const int shape[4], const int permutation[4]);

template __global__ void fft_util_permute_4d_gpu_kernel<double>(const std::complex<double> *in, std::complex<double> *out,
                                                                const int shape[4], const int permutation[4]);

template <typename Dtype>
__global__ void fft_init_alpha_beta_gpu_kernel(std::complex<Dtype> *one_complex, std::complex<Dtype> *zero_complex);

template <>
__global__ void fft_init_alpha_beta_gpu_kernel<float>(std::complex<float> *one_complex, std::complex<float> *zero_complex) {
  reinterpret_cast<cuComplex *>(one_complex)->x = 1.0f;
  reinterpret_cast<cuComplex *>(one_complex)->y = 0.0f;

  reinterpret_cast<cuComplex *>(zero_complex)->x = 0.0f;
  reinterpret_cast<cuComplex *>(zero_complex)->y = 0.0f;
}

template <>
__global__ void fft_init_alpha_beta_gpu_kernel<double>(std::complex<double> *one_complex, std::complex<double> *zero_complex) {
  reinterpret_cast<cuDoubleComplex *>(one_complex)->x = 1.0;
  reinterpret_cast<cuDoubleComplex *>(one_complex)->y = 0.0;

  reinterpret_cast<cuDoubleComplex *>(zero_complex)->x = 0.0;
  reinterpret_cast<cuDoubleComplex *>(zero_complex)->y = 0.0;
}

//// --- end of kernel methods ---

template <>
void npp_complex_add_product<float>(const std::complex<float> *src1, const std::complex<float> *src2, std::complex<float> *dst, int len)
{
  NPP_CHECK(nppsAddProduct_32fc(reinterpret_cast<const Npp32fc *> (src1),
                                reinterpret_cast<const Npp32fc *> (src2),
                                reinterpret_cast<Npp32fc *> (dst), len));
}

template <>
void npp_complex_add_product<double>(const std::complex<double> *src1, const std::complex<double> *src2, std::complex<double> *dst, int len)
{

  NPP_CHECK(nppsAddProduct_64fc(reinterpret_cast<const Npp64fc *> (src1),
                                reinterpret_cast<const Npp64fc *> (src2),
                                reinterpret_cast<Npp64fc *> (dst), len));
}

template <typename Dtype>
__global__ void fft_pointwise_multiply_gemm_construct_array_gpu_kernel(const std::complex<Dtype> *weight_complex,
                                                                       const std::complex<Dtype> *bottom_complex,
                                                                       std::complex<Dtype> *fft_transposed_result,
                                                                       cgemm_sizes sizes,
                                                                       const std::complex<Dtype> **weight_arr,
                                                                       const std::complex<Dtype> **input_arr,
                                                                       std::complex<Dtype> **output_arr) {

  // blockDim (256, 1) ----- (num_output_, (height_out) / CUDA_NUM_THREADS)
  //                              x                y
  int g = blockIdx.x;

  // calculate the height index. The blockIdx.y is usually zero. Because CUDA_THREADS = 512 > 27 = ch/gr. (27 / 512) + 1 = 1.
  const int hw = blockIdx.y * blockDim.x + threadIdx.x;

  if (hw < sizes.H * sizes.W) {
    int h = hw / sizes.W;
    int w = hw % sizes.W;

    const std::complex<Dtype> *weight = weight_complex + (h * sizes.W + w ) * sizes.weight_size;
    const std::complex<Dtype> *input = bottom_complex + (h * sizes.W + w ) * sizes.bottom_size;
    std::complex<Dtype> *output = fft_transposed_result + (h * sizes.W + w ) * sizes.output_size;

    const int idx = hw + g * sizes.H * sizes.W;

    weight_arr[idx] = weight + g * sizes.group_offset_weight;
    input_arr[idx] = input + g * sizes.group_offset_input;
    output_arr[idx] = output + g * sizes.group_offset_output;
  }

}

template __global__ void fft_pointwise_multiply_gemm_construct_array_gpu_kernel<float>(const std::complex<float> *weight_complex,
                                                                                       const std::complex<float> *bottom_complex,
                                                                                       std::complex<float> *fft_transposed_result,
                                                                                       cgemm_sizes sizes,
                                                                                       const std::complex<float> **weight_arr,
                                                                                       const std::complex<float> **input_arr,
                                                                                       std::complex<float> **output_arr);

template __global__ void fft_pointwise_multiply_gemm_construct_array_gpu_kernel<double>(const std::complex<double> *weight_complex,
                                                                                        const std::complex<double> *bottom_complex,
                                                                                        std::complex<double> *fft_transposed_result,
                                                                                        cgemm_sizes sizes,
                                                                                        const std::complex<double> **weight_arr,
                                                                                        const std::complex<double> **input_arr,
                                                                                        std::complex<double> **output_arr);


template <typename Dtype>
void pad_real_blob_gpu(std::vector<int> shape, const int fft_height, const int fft_width,
                       const Dtype *blob_data, Dtype *padded_data, const int pad_h,
                       const int pad_w, const int stride_h, const int stride_w, bool inplace) {

  const int N = shape[0]; // 10
  const int K = shape[1];
  const int H = shape[2];
  const int W = shape[3];

  const int fft_real_size = fft_height * fft_width;

  int size = N * K; // # of arrays (for weights it is num_weights [96 x 3]
  // for input data it is channels [ 1 x 3]

  // if inplace take fft_complex_size * 2 because complex has double the size [sizeof(std::complex)]
  size = inplace ? size * fft_height * (fft_width / 2 + 1) * 2 : size * fft_height * fft_width;


  // set everything to 0 before --> so not set weights are 0-padded :)
  caffe_gpu_memset(size * sizeof(Dtype), 0., padded_data);

  // N = 256 (num_output_)
  // K = 96 / 2 (channels / group) ==> (48 / 512 ) + 1 = 1
  dim3 block_num(N, (H*W / CAFFE_CUDA_NUM_THREADS) + 1);
  int thread_num = CAFFE_CUDA_NUM_THREADS;

  pad_real_blob_gpu_kernel<Dtype><<<block_num, thread_num>>>(
      K, H, W, fft_height, fft_width, blob_data, padded_data,
      pad_h, pad_w, stride_h, stride_w, inplace);
  CUDA_POST_KERNEL_CHECK;
}

template void pad_real_blob_gpu<float>(std::vector<int> shape, const int fft_height, const int fft_width,
                                       const float *blob_data, float *padded_data, const int pad_h,
                                       const int pad_w, const int stride_h, const int stride_w, bool inplace);

template void pad_real_blob_gpu<double>(std::vector<int> shape, const int fft_height, const int fft_width,
                                        const double *blob_data, double *padded_data, const int pad_h,
                                        const int pad_w, const int stride_h, const int stride_w, bool inplace);

template <>
void fft_util_pointwise_multiply_gpu<float>(std::vector<int> shape, int group, const std::complex<float> *bottom_complex,
                                            const std::complex<float> *weight_complex, std::complex<float> *ptwise_result) {
  const int batch_size = shape[0];
  const int N = shape[1];
  const int K = shape[2];
  const int H = shape[3];
  const int W = shape[4];

  const int weight_group_size = N / group;

  // N = 256 (num_output_)
  // K = 96 / 2 (channels / group) ==> (48 / 512 ) + 1 = 1
  // dim3 block_num(N, (K / CAFFE_CUDA_NUM_THREADS) + 1);

  // N = num_output, H * W as second dim so no races happen; because over K (channels) will be summed up
  // On thoise channels sum -ups the threads interfere with another...
  dim3 block_num(batch_size, N, (H * W / CAFFE_CUDA_NUM_THREADS) + 1);
  int thread_num = CAFFE_CUDA_NUM_THREADS;

  const cufftComplex *ffted_bottom_data_cuda  = reinterpret_cast<const cufftComplex *> (bottom_complex);
  const cufftComplex *weight_complex_cuda = reinterpret_cast<const cufftComplex *> (weight_complex);
  cufftComplex *ptwise_result_cuda = reinterpret_cast<cufftComplex *> (ptwise_result);

  fft_pointwise_multiply_float_gpu_kernel<<<block_num, thread_num>>>
      (N, K, H, W, weight_group_size, ffted_bottom_data_cuda, weight_complex_cuda, ptwise_result_cuda);
  CUDA_POST_KERNEL_CHECK;
}

template <>
void fft_util_pointwise_multiply_gpu<double>(std::vector<int> shape, int group, const std::complex<double> *bottom_complex,
                                             const std::complex<double> *weight_complex, std::complex<double> *ptwise_result) {
    const int batch_size = shape[0];
    const int N = shape[1];
    const int K = shape[2];
    const int H = shape[3];
    const int W = shape[4];

    const int weight_group_size = N / group;

    // N = 256 (num_output_)
    // K = 96 / 2 (channels / group) ==> (48 / 512 ) + 1 = 1
    // dim3 block_num(N, (K / CAFFE_CUDA_NUM_THREADS) + 1);

    // N = num_output, H * W as second dim so no races happen; because over K (channels) will be summed up
    // On thoise channels sum -ups the threads interfere with another...
    dim3 block_num(batch_size, N, (H * W / CAFFE_CUDA_NUM_THREADS) + 1);
    int thread_num = CAFFE_CUDA_NUM_THREADS;

    const cufftDoubleComplex *ffted_bottom_data_cuda  = reinterpret_cast<const cufftDoubleComplex *> (bottom_complex);
    const cufftDoubleComplex *weight_complex_cuda = reinterpret_cast<const cufftDoubleComplex *> (weight_complex);
    cufftDoubleComplex *ptwise_result_cuda = reinterpret_cast<cufftDoubleComplex *> (ptwise_result);

    fft_pointwise_multiply_double_gpu_kernel<<<block_num, thread_num>>>
        (N, K, H, W, weight_group_size, ffted_bottom_data_cuda, weight_complex_cuda, ptwise_result_cuda);
    CUDA_POST_KERNEL_CHECK;
}

template <>
void fft_util_pointwise_multiply_backward_gpu<float>(std::vector<int> shape, int group, std::complex<float> *bottom_complex,
                                                     const std::complex<float> *weight_complex, const std::complex<float> *ptwise_result) {
  const int batch_size = shape[0];
  const int N = shape[1];
  const int K = shape[2];
  const int H = shape[3];
  const int W = shape[4];

  const int weight_group_size = N / group;

  // K = channels, H * W as second dim so no races happen; because over N (num_output) will be summed up
  dim3 block_num(batch_size, K, (H * W / CAFFE_CUDA_NUM_THREADS) + 1);
  int thread_num = CAFFE_CUDA_NUM_THREADS;

  cufftComplex *ffted_bottom_data_cuda  = reinterpret_cast<cufftComplex *> (bottom_complex);
  const cufftComplex *weight_complex_cuda = reinterpret_cast<const cufftComplex *> (weight_complex);
  const cufftComplex *ptwise_result_cuda = reinterpret_cast<const cufftComplex *> (ptwise_result);

  fft_pointwise_multiply_backward_float_gpu_kernel<<<block_num, thread_num>>>
      (N, K, H, W, weight_group_size, ffted_bottom_data_cuda, weight_complex_cuda, ptwise_result_cuda);
  CUDA_POST_KERNEL_CHECK;
}

template <>
void fft_util_pointwise_multiply_backward_gpu<double>(std::vector<int> shape, int group, std::complex<double> *bottom_complex,
                                                      const std::complex<double> *weight_complex, const std::complex<double> *ptwise_result) {
  const int batch_size = shape[0];
  const int N = shape[1];
  const int K = shape[2];
  const int H = shape[3];
  const int W = shape[4];

  const int weight_group_size = N / group;

  // K = channels, H * W as second dim so no races happen; because over N (num_output) will be summed up
  dim3 block_num(batch_size, K, (H * W / CAFFE_CUDA_NUM_THREADS) + 1);
  int thread_num = CAFFE_CUDA_NUM_THREADS;

  cufftDoubleComplex *ffted_bottom_data_cuda  = reinterpret_cast<cufftDoubleComplex *> (bottom_complex);
  const cufftDoubleComplex *weight_complex_cuda = reinterpret_cast<const cufftDoubleComplex *> (weight_complex);
  const cufftDoubleComplex *ptwise_result_cuda = reinterpret_cast<const cufftDoubleComplex *> (ptwise_result);

  fft_pointwise_multiply_backward_double_gpu_kernel<<<block_num, thread_num>>>
      (N, K, H, W, weight_group_size, ffted_bottom_data_cuda, weight_complex_cuda, ptwise_result_cuda);
  CUDA_POST_KERNEL_CHECK;
}


template<>
void fft_util_pointwise_multiply_weight_gpu<float>(std::vector<int> shape, int group, const std::complex<float> *bottom_complex,
                                                   std::complex<float> *weight_complex, const std::complex<float> *ptwise_result) {
  const int batch_size = shape[0];
  const int N = shape[1];
  const int K = shape[2];
  const int H = shape[3];
  const int W = shape[4];

  const int weight_group_size = N / group;

  // K = channels, H * W as second dim so no races happen; because over N (num_output) will be summed up
  dim3 block_num(N, K, (H * W / CAFFE_CUDA_NUM_THREADS) + 1);
  int thread_num = CAFFE_CUDA_NUM_THREADS;

  const cufftComplex *ffted_bottom_data_cuda  = reinterpret_cast<const cufftComplex *> (bottom_complex);
  cufftComplex *weight_complex_cuda = reinterpret_cast<cufftComplex *> (weight_complex);
  const cufftComplex *ptwise_result_cuda = reinterpret_cast<const cufftComplex *> (ptwise_result);

  fft_pointwise_multiply_weight_float_gpu_kernel<<<block_num, thread_num>>>
      (batch_size, N, K, H, W, weight_group_size, ffted_bottom_data_cuda, weight_complex_cuda, ptwise_result_cuda);
  CUDA_POST_KERNEL_CHECK;
}


template<>
void fft_util_pointwise_multiply_weight_gpu<double>(std::vector<int> shape, int group, const std::complex<double> *bottom_complex,
                                                   std::complex<double> *weight_complex, const std::complex<double> *ptwise_result) {
  const int batch_size = shape[0];
  const int N = shape[1];
  const int K = shape[2];
  const int H = shape[3];
  const int W = shape[4];

  const int weight_group_size = N / group;

  // K = channels, H * W as second dim so no races happen; because over N (num_output) will be summed up
  dim3 block_num(N, K, (H * W / CAFFE_CUDA_NUM_THREADS) + 1);
  int thread_num = CAFFE_CUDA_NUM_THREADS;

  const cufftDoubleComplex *ffted_bottom_data_cuda  = reinterpret_cast<const cufftDoubleComplex *> (bottom_complex);
  cufftDoubleComplex *weight_complex_cuda = reinterpret_cast<cufftDoubleComplex *> (weight_complex);
  const cufftDoubleComplex *ptwise_result_cuda = reinterpret_cast<const cufftDoubleComplex *> (ptwise_result);

  fft_pointwise_multiply_weight_double_gpu_kernel<<<block_num, thread_num>>>
      (batch_size, N, K, H, W, weight_group_size, ffted_bottom_data_cuda, weight_complex_cuda, ptwise_result_cuda);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
void fft_util_pointwise_multiply_gemm_gpu(cgemm_sizes sizes, const std::complex<Dtype> *bottom_complex,
                                          const std::complex<Dtype> *weight_complex, std::complex<Dtype> *ptwise_result) {
  // alloc data for result
  const int convolution_result_complex_size = sizes.num * sizes.H * sizes.W * sizes.num_output * sizeof(std::complex<Dtype>);
  std::complex<Dtype> *fft_transposed_result;
  CUDA_CHECK(cudaMalloc(&fft_transposed_result, convolution_result_complex_size));

  std::complex<Dtype> one_complex(1., 0.);
  std::complex<Dtype> zero_complex(0., 0.);

  const int M_gemm = sizes.num;
  const int N_gemm = sizes.num_output / sizes.G;
  const int K_gemm = sizes.channels / sizes.G;

  int array_size = sizes.H * sizes.W * sizes.G;

  const std::complex<Dtype> **weight_arr_gpu;
  CUDA_CHECK(cudaMalloc(&weight_arr_gpu, array_size*sizeof(std::complex<Dtype>)));
  const std::complex<Dtype> **input_arr_gpu;
  CUDA_CHECK(cudaMalloc(&input_arr_gpu, array_size*sizeof(std::complex<Dtype>)));
  std::complex<Dtype> **output_arr_gpu;
  CUDA_CHECK(cudaMalloc(&output_arr_gpu, array_size*sizeof(std::complex<Dtype>)));

  dim3 block_num(sizes.G, (sizes.H * sizes.W / CAFFE_CUDA_NUM_THREADS) + 1);
  int thread_num = CAFFE_CUDA_NUM_THREADS;

  fft_pointwise_multiply_gemm_construct_array_gpu_kernel<<<block_num, thread_num>>>
      (weight_complex, bottom_complex, fft_transposed_result, sizes, weight_arr_gpu, input_arr_gpu, output_arr_gpu);
  CUDA_POST_KERNEL_CHECK;

  int lda = K_gemm * sizes.G;
  int ldb = K_gemm; // because TransB = Trans!
  int ldc = N_gemm * sizes.G;

  // Do batched matrix multiplication
  caffe_gpu_gemm_complex_batch<Dtype>(CblasNoTrans, CblasConjTrans, M_gemm, N_gemm, K_gemm,
                                      &one_complex, input_arr_gpu, weight_arr_gpu, &zero_complex, output_arr_gpu, sizes.H * sizes.W * sizes.G,
                                      &lda, &ldb, &ldc);

  CUDA_CHECK(cudaFree(weight_arr_gpu));
  CUDA_CHECK(cudaFree(input_arr_gpu));
  CUDA_CHECK(cudaFree(output_arr_gpu));

  // result_dim = 256 x 129 x 96 x 1 ==> 1 x 96 x 256 x 129
  const int shape_result[] = {sizes.H, sizes.W, sizes.num, sizes.num_output};
  fft_util_geam_transpose_gpu(fft_transposed_result, ptwise_result, shape_result, 2);
  CUDA_CHECK(cudaFree(fft_transposed_result));
}

template void fft_util_pointwise_multiply_gemm_gpu<float>(cgemm_sizes sizes, const std::complex<float> *bottom_complex,
                                                          const std::complex<float> *weight_complex, std::complex<float> *ptwise_result);

template void fft_util_pointwise_multiply_gemm_gpu<double>(cgemm_sizes sizes, const std::complex<double> *bottom_complex,
                                                           const std::complex<double> *weight_complex, std::complex<double> *ptwise_result);

template <typename Dtype>
void fft_util_pointwise_multiply_gemm_backward_gpu(cgemm_sizes sizes, std::complex<Dtype> *bottom_complex,
                                                   const std::complex<Dtype> *weight_complex, const std::complex<Dtype> *ptwise_result) {
  // alloc data for result
  const int convolution_result_complex_size = sizes.num * sizes.H * sizes.W * sizes.channels * sizeof(std::complex<Dtype>);
  std::complex<Dtype> *fft_transposed_result;
  CUDA_CHECK(cudaMalloc(&fft_transposed_result, convolution_result_complex_size));

  std::complex<Dtype> one_complex(1., 0.);
  std::complex<Dtype> zero_complex(0., 0.);

  const int M_gemm = sizes.num;
  const int N_gemm = sizes.channels / sizes.G;
  const int K_gemm = sizes.num_output / sizes.G;

  int array_size = sizes.H * sizes.W * sizes.G;

  const std::complex<Dtype> **weight_arr_gpu;
  CUDA_CHECK(cudaMalloc(&weight_arr_gpu, array_size*sizeof(std::complex<Dtype>)));
  const std::complex<Dtype> **input_arr_gpu;
  CUDA_CHECK(cudaMalloc(&input_arr_gpu, array_size*sizeof(std::complex<Dtype>)));
  std::complex<Dtype> **output_arr_gpu;
  CUDA_CHECK(cudaMalloc(&output_arr_gpu, array_size*sizeof(std::complex<Dtype>)));

  dim3 block_num(sizes.G, (sizes.H * sizes.W / CAFFE_CUDA_NUM_THREADS) + 1);
  int thread_num = CAFFE_CUDA_NUM_THREADS;

  fft_pointwise_multiply_gemm_construct_array_gpu_kernel<<<block_num, thread_num>>>
      (weight_complex, ptwise_result, fft_transposed_result, sizes, weight_arr_gpu, input_arr_gpu, output_arr_gpu);
  CUDA_POST_KERNEL_CHECK;

  int lda = K_gemm * sizes.G; // because TransA = NoTrans!
  int ldb = N_gemm; // because TransB = NoTrans!
  int ldc = N_gemm * sizes.G;

  // Do batched matrix multiplication (BW: NoTrans and Convolution [vs. conjugate/cross-correlation in forward pass])
  caffe_gpu_gemm_complex_batch<Dtype>(CblasNoTrans, CblasNoTrans, M_gemm, N_gemm, K_gemm,
                                      &one_complex, input_arr_gpu, weight_arr_gpu, &zero_complex, output_arr_gpu, sizes.H * sizes.W * sizes.G,
                                      &lda, &ldb, &ldc);

  CUDA_CHECK(cudaFree(weight_arr_gpu));
  CUDA_CHECK(cudaFree(input_arr_gpu));
  CUDA_CHECK(cudaFree(output_arr_gpu));

  // result_dim = 256 x 129 x 96 x 1 ==> 1 x 96 x 256 x 129
  const int shape_result[] = {sizes.H, sizes.W, sizes.num, sizes.channels};
  fft_util_geam_transpose_gpu(fft_transposed_result, bottom_complex, shape_result, 2);
  CUDA_CHECK(cudaFree(fft_transposed_result));
}

template void fft_util_pointwise_multiply_gemm_backward_gpu<float>(cgemm_sizes sizes, std::complex<float> *bottom_complex,
                                                                   const std::complex<float> *weight_complex, const std::complex<float> *ptwise_result);

template void fft_util_pointwise_multiply_gemm_backward_gpu<double>(cgemm_sizes sizes, std::complex<double> *bottom_complex,
                                                                    const std::complex<double> *weight_complex, const std::complex<double> *ptwise_result);

template <typename Dtype>
void fft_util_pointwise_multiply_gemm_weight_gpu(cgemm_sizes sizes, const std::complex<Dtype> *bottom_complex,
                                                 std::complex<Dtype> *weight_complex, const std::complex<Dtype> *ptwise_result)
{
  // alloc data for result
  const int convolution_result_complex_size = sizes.H * sizes.W * sizes.num_output * (sizes.channels / sizes.G) * sizeof(std::complex<Dtype>);
  std::complex<Dtype> *fft_transposed_result;
  CUDA_CHECK(cudaMalloc(&fft_transposed_result, convolution_result_complex_size));

  std::complex<Dtype> one_complex(1., 0.);
  std::complex<Dtype> zero_complex(0., 0.);

  const int M_gemm = sizes.num_output / sizes.G;
  const int N_gemm = sizes.channels / sizes.G;
  const int K_gemm = sizes.num;

  int array_size = sizes.H * sizes.W * sizes.G;

  const std::complex<Dtype> **weight_arr_gpu;
  CUDA_CHECK(cudaMalloc(&weight_arr_gpu, array_size*sizeof(std::complex<Dtype>)));
  const std::complex<Dtype> **input_arr_gpu;
  CUDA_CHECK(cudaMalloc(&input_arr_gpu, array_size*sizeof(std::complex<Dtype>)));
  std::complex<Dtype> **output_arr_gpu;
  CUDA_CHECK(cudaMalloc(&output_arr_gpu, array_size*sizeof(std::complex<Dtype>)));

  dim3 block_num(sizes.G, (sizes.H * sizes.W / CAFFE_CUDA_NUM_THREADS) + 1);
  int thread_num = CAFFE_CUDA_NUM_THREADS;

  fft_pointwise_multiply_gemm_construct_array_gpu_kernel<<<block_num, thread_num>>>
      (bottom_complex, ptwise_result, fft_transposed_result, sizes, weight_arr_gpu, input_arr_gpu, output_arr_gpu);
  CUDA_POST_KERNEL_CHECK;

  int lda = M_gemm * sizes.G;  // because TransA = (Conj)Trans!
  int ldb = N_gemm * sizes.G; // because TransB = NoTrans!
  int ldc = N_gemm;

  // Do batched matrix multiplication (W: ...?)
  caffe_gpu_gemm_complex_batch<Dtype>(CblasConjTrans, CblasNoTrans, M_gemm, N_gemm, K_gemm,
                                      &one_complex, input_arr_gpu, weight_arr_gpu, &zero_complex, output_arr_gpu, sizes.H * sizes.W * sizes.G,
                                      &lda, &ldb, &ldc);

  CUDA_CHECK(cudaFree(weight_arr_gpu));
  CUDA_CHECK(cudaFree(input_arr_gpu));
  CUDA_CHECK(cudaFree(output_arr_gpu));

  // result_dim = 256 x 129 x 96 x 1 ==> 1 x 96 x 256 x 129
  const int shape_result[] = {sizes.H, sizes.W, sizes.num_output, sizes.channels / sizes.G};
  fft_util_geam_transpose_gpu(fft_transposed_result, weight_complex, shape_result, 2);
  CUDA_CHECK(cudaFree(fft_transposed_result));
}

template void fft_util_pointwise_multiply_gemm_weight_gpu<float>(cgemm_sizes sizes, const std::complex<float> *bottom_complex,
                                                                 std::complex<float> *weight_complex, const std::complex<float> *ptwise_result);

template void fft_util_pointwise_multiply_gemm_weight_gpu<double>(cgemm_sizes sizes, const std::complex<double> *bottom_complex,
                                                                  std::complex<double> *weight_complex, const std::complex<double> *ptwise_result);

template <typename Dtype>
void fft_util_normalize_gpu(std::vector<int> shape, int fft_height, int fft_width,
                            const int stride_h, const int stride_w, const int pad_h, const int pad_w,
                            Dtype normalize_factor, const Dtype *fft_result_real,
                            Dtype *result, bool add_to_result) {

  const int N = shape[0];
  const int K = shape[1];
  const int H = shape[2];
  const int W = shape[3];

  dim3 block_num(N, K, (H * W / CAFFE_CUDA_NUM_THREADS) + 1);
  int thread_num = CAFFE_CUDA_NUM_THREADS;

  fft_util_normalize_gpu_kernel<<<block_num, thread_num>>>
      (K, H, W, fft_height, fft_width, stride_h, stride_w,
       pad_h, pad_w, normalize_factor, fft_result_real,
       result, add_to_result);
  CUDA_POST_KERNEL_CHECK;
}

template void fft_util_normalize_gpu<float>(std::vector<int> shape, int fft_height, int fft_width,
                                            const int stride_h, const int stride_w, const int pad_h, const int pad_w,
                                            float normalize_factor, const float *fft_result_real,
                                            float *result, bool add_to_result);

template void fft_util_normalize_gpu<double>(std::vector<int> shape, int fft_height, int fft_width,
                                             const int stride_h, const int stride_w, const int pad_h, const int pad_w,
                                             double normalize_factor, const double *fft_result_real,
                                             double *result, bool add_to_result);

template <typename Dtype>
void fft_util_geam_transpose_gpu(const std::complex<Dtype> *in, std::complex<Dtype> *out,
                                 const int shape[4], const int sep) {
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

  caffe_gpu_geam_complex<Dtype>(CblasTrans, CblasNoTrans, rows, cols, &one_complex, in, cols,
                                NULL, &zero_complex, rows, out, rows);
}

template void fft_util_geam_transpose_gpu<float>(const std::complex<float> *in, std::complex<float> *out,
                                                 const int shape[4], const int sep);

template void fft_util_geam_transpose_gpu<double>(const std::complex<double> *in, std::complex<double> *out,
                                                  const int shape[4], const int sep);

template <typename Dtype>
void fft_util_permute_4d_gpu(const std::complex<Dtype> *in, std::complex<Dtype> *out,
                             const int shape[4], const int permutation[4]) {

  const int N = shape[0];              // 96
  // const int K = shape[1];              // 3
  const int H = shape[2];              // 256
  const int W = shape[3];              // 129

  dim3 block_num(N, (H * W / CAFFE_CUDA_NUM_THREADS) + 1);
  int thread_num = CAFFE_CUDA_NUM_THREADS;


  int *shape_gpu;
  int *permutation_gpu;
  CUDA_CHECK(cudaMalloc(&shape_gpu, sizeof(int) * 4));
  CUDA_CHECK(cudaMalloc(&permutation_gpu, sizeof(int) * 4));
  CUDA_CHECK(cudaMemcpy(shape_gpu, shape, sizeof(int) * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(permutation_gpu, permutation, sizeof(int) * 4, cudaMemcpyHostToDevice));


  fft_util_permute_4d_gpu_kernel<<<block_num, thread_num>>>
      (in, out, shape_gpu, permutation_gpu);
  CUDA_POST_KERNEL_CHECK;

  CUDA_CHECK(cudaFree(shape_gpu));
  CUDA_CHECK(cudaFree(permutation_gpu));
}

template void fft_util_permute_4d_gpu<float>(const std::complex<float> *in, std::complex<float> *out,
                                             const int shape[4], const int permutation[4]);

template void fft_util_permute_4d_gpu<double>(const std::complex<double> *in, std::complex<double> *out,
                                              const int shape[4], const int permutation[4]);

// --- cufft calls ----


template<>
void fft_gpu_plan_many_dft_r2c_2d<float>(cufftHandle *plan, int n0,
                                         int n1,
                                         int how_many) {
  int rank = 2;
  int n[] = {n0, n1};
  int idist = n0 * n1; /* = 256*256, the distance in memory
                                          between the first element
                                          of the first array and the
                                          first element of the second array */
  int istride = 1; /* array is contiguous in memory */
  int *inembed = NULL;

  // out
  int odist = n0 * (n1 / 2 + 1);
  int ostride = 1;
  int *onembed = NULL;

  CUFFT_CHECK(cufftCreate(plan));
  CUFFT_CHECK(cufftPlanMany(plan, rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_R2C, how_many));
}

template<>
void fft_gpu_plan_many_dft_r2c_2d<double>(cufftHandle *plan, int n0,
                                          int n1,
                                          int how_many) {
  int rank = 2;
  int n[] = {n0, n1};
  int idist = n0 * n1; /* = 256*256, the distance in memory
                                          between the first element
                                          of the first array and the
                                          first element of the second array */
  int istride = 1; /* array is contiguous in memory */
  int *inembed = NULL;

  // out
  int odist = n0 * (n1 / 2 + 1);
  int ostride = 1;
  int *onembed = NULL;

  CUFFT_CHECK(cufftCreate(plan));
  CUFFT_CHECK(cufftPlanMany(plan, rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_D2Z, how_many));
}

template<>
void fft_gpu_plan_many_dft_c2r_2d<float>(cufftHandle *plan, int n0,
                                         int n1,
                                         int how_many) {

  int rank = 2;
  int n[] = {n0, n1};
  int idist = n0 * (n1 / 2 + 1); /* = 256*129, the distance in memory
                                          between the first element
                                          of the first array and the
                                          first element of the second array */
  int istride = 1; /* array is contiguous in memory */
  int *inembed = NULL;

  // out
  int odist = n0 * n1;
  int ostride = 1;
  int *onembed = NULL;

  CUFFT_CHECK(cufftCreate(plan));
  CUFFT_CHECK(cufftPlanMany(plan, rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_C2R, how_many));
}

template<>
void fft_gpu_plan_many_dft_c2r_2d<double>(cufftHandle *plan, int n0,
                                          int n1,
                                          int how_many) {

  int rank = 2;
  int n[] = {n0, n1};
  int idist = n0 * (n1 / 2 + 1); /* = 256*129, the distance in memory
                                          between the first element
                                          of the first array and the
                                          first element of the second array */
  int istride = 1; /* array is contiguous in memory */
  int *inembed = NULL;

  // out
  int odist = n0 * n1;
  int ostride = 1;
  int *onembed = NULL;

  CUFFT_CHECK(cufftCreate(plan));
  CUFFT_CHECK(cufftPlanMany(plan, rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_Z2D, how_many));
}

template<>
void fft_gpu_execute_plan_r2c<float>(cufftHandle plan, float *in, std::complex<float> *out) {
  CUFFT_CHECK(cufftExecR2C(plan, in, reinterpret_cast<cufftComplex *>(out)));
}

template<>
void fft_gpu_execute_plan_r2c<double>(cufftHandle plan, double *in, std::complex<double> *out) {
  CUFFT_CHECK(cufftExecD2Z(plan, in, reinterpret_cast<cufftDoubleComplex *>(out)));
}

template<>
void fft_gpu_execute_plan_c2r<float>(cufftHandle plan, std::complex<float> *in, float *out) {
  CUFFT_CHECK(cufftExecC2R(plan, reinterpret_cast<cufftComplex *>(in), out));
}

template<>
void fft_gpu_execute_plan_c2r<double>(cufftHandle plan, std::complex<double> *in, double *out) {
  CUFFT_CHECK(cufftExecZ2D(plan, reinterpret_cast<cufftDoubleComplex *>(in), out));
}

void fft_gpu_destroy_plan(cufftHandle plan_handle) {
  cufftDestroy(plan_handle);
}

__device__ void fft_gpu_cmultiply_add(const cufftComplex first, const cufftComplex second,
                                      cufftComplex *out, bool multiply_with_conjugate) {
  // formula for complex mult from here: https://en.wikipedia.org/wiki/Complex_number#Multiplication_and_division
  // (a+bi) (c+di) = (ac-bd) + (bc+ad)i.

  float a = first.x;
  float b = first.y;
  float c = second.x;
  float d = second.y;

  if (multiply_with_conjugate) {
    out->x += a * c + b * d;
    out->y += b * c - a * d;
  } else {
    out->x += a * c - b * d;
    out->y += b * c + a * d;
  }
}

__device__ void fft_gpu_zmultiply_add(const cufftDoubleComplex first, const cufftDoubleComplex second,
                                      cufftDoubleComplex *out, bool multiply_with_conjugate) {
  // formula for complex mult from here: https://en.wikipedia.org/wiki/Complex_number#Multiplication_and_division
  // (a+bi) (c+di) = (ac-bd) + (bc+ad)i.

  double a = first.x;
  double b = first.y;
  double c = second.x;
  double d = second.y;

  if (multiply_with_conjugate) {
    out->x += a * c + b * d;
    out->y += b * c - a * d;
  } else {
    out->x += a * c - b * d;
    out->y += b * c + a * d;
  }
}
}
