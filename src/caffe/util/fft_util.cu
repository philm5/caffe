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
__global__ void pad_real_blob_gpu_kernel(const int K, const int H, const int W, const int fft_height, const int fft_width,
                                         const int fft_real_size, const Dtype *blob_data, Dtype *padded_data,
                                         const int pad_h, const int pad_w, const bool flip) {

  // blockDim (256, 1) ----- (num_output_, (ch_gr) / CUDA_NUM_THREADS)
  //                              x                y
  const int n = blockIdx.x;

  // calculate the channel index. The blockIdx.y is usually zero. Because CUDA_THREADS = 512 > 48 = ch/gr. (48 / 512) + 1 = 1.
  const int hw = blockIdx.y * blockDim.x + threadIdx.x;


  if (hw < H*W) {

    for (int k = 0; k < K; ++k) {
      // get offset with channels and the idx of the output.
      const int offset_weight_real = (n * K + k) * fft_real_size;
      const int offset_blob_real = (n * K + k) * H * W;
      const int h = hw / W;
      const int w = hw % W;

      const int idx_weight_real = offset_weight_real + (h + pad_h) * fft_width + (w + pad_w);
      // copy each weight into the fft_weights_in_real_
      // get ptr to blob data. indexing see: http://caffe.berkeleyvision.org/tutorial/net_layer_blob.html
      // Blob memory is row-major in layout, so the last / rightmost dimension changes fastest. For example,
      // in a 4D blob, the value at index (n, k, h, w) is physically located at index ((n * K + k) * H + h) * W + w.
      // 96 x 3 x 11 x 11 (num_output_, channels_ / group_, kernel_height, width_)

      // if flip = true ==> flip the indices of the weights. Caffe actually does not a convolution but a
      // a cross-correlation according to: https://github.com/BVLC/caffe/issues/2513
      const int h_idx = flip ? H - (h + 1) : h;
      const int w_idx = flip ? W - (w + 1) : w;
      const int idx_weight_in_blob = offset_blob_real + h_idx * W + w_idx;

      Dtype data_in_blob = blob_data[idx_weight_in_blob];
      padded_data[idx_weight_real] = data_in_blob;

    }
  }
}

template __global__ void pad_real_blob_gpu_kernel<float>(const int K, const int H, const int W, const int fft_height, const int fft_width,
                                                         const int fft_real_size, const float *blob_data, float *padded_data,
                                                         const int pad_h, const int pad_w, const bool flip);

template __global__ void pad_real_blob_gpu_kernel<double>(const int K, const int H, const int W, const int fft_height, const int fft_width,
                                                          const int fft_real_size, const double *blob_data, double *padded_data,
                                                          const int pad_h, const int pad_w, const bool flip);

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

    // printf("<<<%i, %i>>>| n: %i k: %i K: %i\n", blockIdx.x, threadIdx.x, n, k, K);
    // check which group_ idx we are in
    const int group_idx = n / weight_group_size;
    const int group = (weight_group_size / N);

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

      // formula for complex mult from here: https://en.wikipedia.org/wiki/Complex_number#Multiplication_and_division
      // (a+bi) (c+di) = (ac-bd) + (bc+ad)i.
      float a = weight.x;
      float b = weight.y;
      float c = input.x;
      float d = input.y;

      const int res_idx = n * H * W + hw;
      res_data[res_idx].x += a * c - b * d;
      res_data[res_idx].y += b * c + a * d;
    }
  }
}

__global__ void fft_pointwise_multiply_double_gpu_kernel(const int N, const int K, const int H, const int W,
                                                         const int weight_group_size, const cufftDoubleComplex *ffted_bottom_data,
                                                         const cufftDoubleComplex *weight_complex, cufftDoubleComplex *ptwise_result) {
#warning IMPLEMENT !!!
}

template <typename Dtype>
__global__ void fft_util_normalize_gpu_kernel(const int N, const int H, const int W, const int kernel_h,
                                              const int kernel_w, const int stride_h, const int stride_w,
                                              float normalize_factor, int fft_height, int fft_width,
                                              const Dtype *fft_convolution_result_real, Dtype *top_data) {

  // blockDim (256, 1) ----- (num_output_, (height_out) / CUDA_NUM_THREADS)
  //                              x                y
  const int batch_idx = blockIdx.x;
  const int n = blockIdx.y;

  // calculate the height index. The blockIdx.y is usually zero. Because CUDA_THREADS = 512 > 27 = ch/gr. (27 / 512) + 1 = 1.
  const int hw = blockIdx.z * blockDim.x + threadIdx.x;

  if (hw < H*W) {
    const int fft_real_size = fft_height * fft_width;
    const int offset_res_real = (batch_idx * N + n) * fft_real_size;
    const int h = hw / W;
    const int w = hw % W;

    // caffe does a valid convolution. fft is a full convolution. so the first 'valid' result is at
    // idx (kernel_h_ - 1). The stride times the idx of the output pixel will be added onto this.
    const int h_idx = (kernel_h - 1) + h * stride_h;

    // caffe does a valid convolution. fft is a full convolution. so the first 'valid' result is at
    // idx (kernel_w_ - 1). The stride times the idx of the output pixel will be added onto this.
    const int w_idx = (kernel_w - 1) + w * stride_w;
    //((n * K + k) * H + h) * W + w;
    const int top_data_idx = ((batch_idx * N + n) * H + h) * W + w;

    // the index in the data of the convolution result array (the real one)
    const int res_data_idx = offset_res_real + h_idx * fft_width + w_idx;

    // normalize fft and sum up everything from the input channels...
    top_data[top_data_idx] = fft_convolution_result_real[res_data_idx] * normalize_factor;
  }
}

template __global__ void fft_util_normalize_gpu_kernel<float>(const int N, const int H, const int W, const int kernel_h,
                                                              const int kernel_w, const int stride_h, const int stride_w,
                                                              float normalize_factor, int fft_height, int fft_width,
                                                              const float *fft_convolution_result_real, float *top_data);

template __global__ void fft_util_normalize_gpu_kernel<double>(const int N, const int H, const int W, const int kernel_h,
                                                               const int kernel_w, const int stride_h, const int stride_w,
                                                               float normalize_factor, int fft_height, int fft_width,
                                                               const double *fft_convolution_result_real, double *top_data);

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
__global__ void fft_pointwise_multiply_gemm_construct_array_gpu_kernel(const int H, const int W, const int G,
                                                                       const std::complex<Dtype> *weight_complex,
                                                                       const std::complex<Dtype> *bottom_complex,
                                                                       std::complex<Dtype> *fft_transposed_result,
                                                                       const int weight_size, const int bottom_size,
                                                                       const int output_size, const int group_offset_weight,
                                                                       const int group_offset_input, const int group_offset_output,
                                                                       const std::complex<Dtype> **weight_arr,
                                                                       const std::complex<Dtype> **input_arr,
                                                                       std::complex<Dtype> **output_arr) {

  // blockDim (256, 1) ----- (num_output_, (height_out) / CUDA_NUM_THREADS)
  //                              x                y
  int g = blockIdx.x;

  // calculate the height index. The blockIdx.y is usually zero. Because CUDA_THREADS = 512 > 27 = ch/gr. (27 / 512) + 1 = 1.
  const int hw = blockIdx.y * blockDim.x + threadIdx.x;

  if (hw < H*W) {
    int h = hw / W;
    int w = hw % W;

    const std::complex<Dtype> *weight = weight_complex + (h * W + w ) * weight_size;
    const std::complex<Dtype> *input = bottom_complex + (h * W + w ) * bottom_size;
    std::complex<Dtype> *output = fft_transposed_result + (h * W + w ) * output_size;

    const int idx = hw + g * H * W;

    weight_arr[idx] = weight + g * group_offset_weight;
    input_arr[idx] = input + g * group_offset_input;
    output_arr[idx] = output + g * group_offset_output;
  }

}

template __global__ void fft_pointwise_multiply_gemm_construct_array_gpu_kernel<float>(const int H, const int W, const int G,
                                                                                       const std::complex<float> *weight_complex,
                                                                                       const std::complex<float> *bottom_complex,
                                                                                       std::complex<float> *fft_transposed_result,
                                                                                       const int weight_size, const int bottom_size,
                                                                                       const int output_size, const int group_offset_weight,
                                                                                       const int group_offset_input, const int group_offset_output,
                                                                                       const std::complex<float> **weight_arr,
                                                                                       const std::complex<float> **input_arr,
                                                                                       std::complex<float> **output_arr);

template __global__ void fft_pointwise_multiply_gemm_construct_array_gpu_kernel<double>(const int H, const int W, const int G,
                                                                                        const std::complex<double> *weight_complex,
                                                                                        const std::complex<double> *bottom_complex,
                                                                                        std::complex<double> *fft_transposed_result,
                                                                                        const int weight_size, const int bottom_size,
                                                                                        const int output_size, const int group_offset_weight,
                                                                                        const int group_offset_input, const int group_offset_output,
                                                                                        const std::complex<double> **weight_arr,
                                                                                        const std::complex<double> **input_arr,
                                                                                        std::complex<double> **output_arr);


template <typename Dtype>
void pad_real_blob_gpu(std::vector<int> shape, const int fft_height, const int fft_width,
                       const Dtype *blob_data, Dtype *padded_data, const int pad_h,
                       const int pad_w, const bool flip) {

  const int N = shape[0]; // 10
  const int K = shape[1];
  const int H = shape[2];
  const int W = shape[3];

  const int fft_real_size = fft_height * fft_width;

  int num_arr = N * K; // # of arrays (for weights it is num_weights [96 x 3]
  // for input data it is channels [ 1 x 3]

  // set everything to 0 before --> so not set weights are 0-padded :)
  caffe_gpu_memset(fft_real_size * num_arr * sizeof(Dtype), 0., padded_data);

  // N = 256 (num_output_)
  // K = 96 / 2 (channels / group) ==> (48 / 512 ) + 1 = 1
  dim3 block_num(N, (H*W / CAFFE_CUDA_NUM_THREADS) + 1);
  int thread_num = CAFFE_CUDA_NUM_THREADS;

  pad_real_blob_gpu_kernel<Dtype><<<block_num, thread_num>>>(
      K, H, W, fft_height, fft_width, fft_real_size,
      blob_data, padded_data, pad_h, pad_w, flip);
  CUDA_POST_KERNEL_CHECK;
}

template void pad_real_blob_gpu<float>(std::vector<int> shape, const int fft_height, const int fft_width,
                                       const float *blob_data, float *padded_data, const int pad_h,
                                       const int pad_w, const bool flip);

template void pad_real_blob_gpu<double>(std::vector<int> shape, const int fft_height, const int fft_width,
                                        const double *blob_data, double *padded_data, const int pad_h,
                                        const int pad_w, const bool flip);

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
#warning IMPLEMENT THIS METHOD!
}


template <typename Dtype>
void fft_util_pointwise_multiply_npp_gpu(std::vector<int> shape, int group, const std::complex<Dtype> *bottom_complex,
                                         const std::complex<Dtype> *weight_complex, std::complex<Dtype> *ptwise_result) {

#warning Try Streams????
#warning implement batch size!
  const int N = shape[0];
  const int K = shape[1];
  const int H = shape[2];
  const int W = shape[3];

  const int weight_group_size = N / group;
  const int fft_complex_size = H * W;

  int n = 0;
  int k = 0;

  for (n = 0; n < N; ++n) {
    const int res_offset = n * fft_complex_size;
    // check which group_ idx we are in
    const int group_idx = n / weight_group_size;
    for (k = 0; k < K; ++k) {
      // get the input_k. this is the k we use to index the input k-dimension. the max input_k is group_ times more
      // than the max k of the weight.
      const int input_offset = (k + group_idx * K) * fft_complex_size;
      const int weight_offset = (n * K + k) * fft_complex_size;
      npp_complex_add_product<Dtype>(bottom_complex + input_offset, weight_complex + weight_offset,
                                     ptwise_result + res_offset, fft_complex_size);
    }
  }
}

template void fft_util_pointwise_multiply_npp_gpu<float>(std::vector<int> shape, int group,
                                                         const std::complex<float> *bottom_complex,
                                                         const std::complex<float> *weight_complex,
                                                         std::complex<float> *ptwise_result);

template void fft_util_pointwise_multiply_npp_gpu<double>(std::vector<int> shape, int group,
                                                          const std::complex<double> *bottom_complex,
                                                          const std::complex<double> *weight_complex,
                                                          std::complex<double> *ptwise_result);

//template <typename Dtype>
//void fft_util_pointwise_multiply_gemm_gpu(std::vector<int> shape, int group, const std::complex<Dtype> *bottom_complex,
//                                         const std::complex<Dtype> *weight_complex, std::complex<Dtype> *ptwise_result) {
//  const int batch_size = shape[0];
//  const int N = shape[1];
//  const int K = shape[2];
//  const int H = shape[3];
//  const int W = shape[4];
//  const int G = group;
//
//  // alloc data for result
//  const int convolution_result_complex_size = H * W * N * sizeof(std::complex<Dtype>);
//  std::complex<Dtype> *fft_transposed_result;
//  CUDA_CHECK(cudaMalloc(&fft_transposed_result, convolution_result_complex_size));
//
//  //                      num_output * channels / group = 96 * 3
//  const int weight_size = N * K;
//
//  //                      num_images      * channels    =  1 * 3
//  const int bottom_size = batch_size * K * G;
//
//  //                      num_output * num_images       = 96 * 1
//  const int output_size = N * batch_size;
//
//  std::complex<Dtype> one_complex(1., 0.);
//  std::complex<Dtype> zero_complex(0., 0.);
//
//  const int group_offset_weight = weight_size / G;
//  const int group_offset_input = bottom_size / batch_size / G;
//  const int group_offset_output = output_size / batch_size / G;
//
//  const int M_gemm = batch_size;
//  const int N_gemm = N / G;
//  const int K_gemm = K;
//
//  const std::complex<Dtype> **weight_arr_gpu;
//  CUDA_CHECK(cudaMalloc(&weight_arr_gpu, H*W*G*sizeof(std::complex<Dtype>)));
//  const std::complex<Dtype> **input_arr_gpu;
//  CUDA_CHECK(cudaMalloc(&input_arr_gpu, H*W*G*sizeof(std::complex<Dtype>)));
//  std::complex<Dtype> **output_arr_gpu;
//  CUDA_CHECK(cudaMalloc(&output_arr_gpu, H*W*G*sizeof(std::complex<Dtype>)));
//
//  dim3 block_num(G, (H*W / CAFFE_CUDA_NUM_THREADS) + 1);
//  int thread_num = CAFFE_CUDA_NUM_THREADS;
//
//  fft_pointwise_multiply_gemm_construct_array_gpu_kernel<<<block_num, thread_num>>>
//      (H, W, G, weight_complex, bottom_complex, fft_transposed_result, weight_size, bottom_size, output_size,
//          group_offset_weight, group_offset_input, group_offset_output, weight_arr_gpu, input_arr_gpu, output_arr_gpu);
//  CUDA_POST_KERNEL_CHECK;
//
//  int lda = K_gemm * G; // 96
//  int ldb = K_gemm;
//  int ldc = N_gemm * G; // 256
//
//  // Do batched matrix multiplication
//  caffe_gpu_gemm_complex_batch<Dtype>(CblasNoTrans, CblasTrans, M_gemm, N_gemm, K_gemm,
//                                      &one_complex, input_arr_gpu, weight_arr_gpu, &zero_complex, output_arr_gpu, H*W*G,
//                                      &lda, &ldb, &ldc);
//
//  CUDA_CHECK(cudaFree(weight_arr_gpu));
//  CUDA_CHECK(cudaFree(input_arr_gpu));
//  CUDA_CHECK(cudaFree(output_arr_gpu));
//
//  // result_dim = 256 x 129 x 96 x 1 ==> 1 x 96 x 256 x 129
//  const int shape_result[] = {H, W, N, batch_size};
//  const int permutation_result[] = {2, 3, 0, 1};
//  fft_util_permute_4d_gpu(fft_transposed_result, ptwise_result, shape_result, permutation_result);
//  CUDA_CHECK(cudaFree(fft_transposed_result));
//}

// cublas stream
template <typename Dtype>
void fft_util_pointwise_multiply_gemm_gpu(std::vector<int> shape, int group, const std::complex<Dtype> *bottom_complex,
                                         const std::complex<Dtype> *weight_complex, std::complex<Dtype> *ptwise_result) {
  const int batch_size = shape[0];
  const int N = shape[1];
  const int K = shape[2];
  const int H = shape[3];
  const int W = shape[4];
  const int G = group;

  // alloc data for result
  const int convolution_result_complex_size = batch_size * H * W * N * sizeof(std::complex<Dtype>);
  std::complex<Dtype> *fft_transposed_result;
  CUDA_CHECK(cudaMalloc(&fft_transposed_result, convolution_result_complex_size));

  //                      num_output * channels / group = 96 * 3
  const int weight_size = N * K;

  //                      num_images      * channels    =  1 * 3
  const int bottom_size = batch_size * K * G;

  //                      num_output * num_images       = 96 * 1
  const int output_size = N * batch_size;

  std::complex<Dtype> one_complex(1., 0.);
  std::complex<Dtype> zero_complex(0., 0.);

  const int group_offset_weight = weight_size / G;
  const int group_offset_input = bottom_size / batch_size / G;
  const int group_offset_output = output_size / batch_size / G;

  const int M_gemm = batch_size;
  const int N_gemm = N / G;
  const int K_gemm = K;

  int streams_number = H * W * G;
  cudaStream_t stream[streams_number];

  for (int i = 0; i < streams_number; i++) {
    CUDA_CHECK(cudaStreamCreate(&stream[i]));
  }

  int idx = 0;
  int lda = K_gemm * G;
  int ldb = K_gemm;
  int ldc = N_gemm * G;

  //const int shape[] = {this->num_output_, this->channels_ / this->group_, this->fft_height_, (this->fft_width_ / 2) + 1};
  for (int h = 0; h < H; ++h) {
    for (int w = 0; w < W; ++w) {

      const std::complex<Dtype> *weight = weight_complex + (h * W + w ) * weight_size;
      const std::complex<Dtype> *input = bottom_complex + (h * W + w ) * bottom_size;
      std::complex<Dtype> *output = fft_transposed_result + (h * W + w ) * output_size;

      for (int g = 0; g < G; ++g) {

        const std::complex<Dtype> *weight_g = weight + g * group_offset_weight;
        const std::complex<Dtype> *input_g = input + g * group_offset_input;
        std::complex<Dtype> *output_g = output + g * group_offset_output;

        cublasSetStream(Caffe::cublas_handle(), stream[idx]);
        caffe_gpu_gemm_complex<Dtype>(CblasNoTrans, CblasTrans, M_gemm, N_gemm, K_gemm, &one_complex, input_g,
                                      weight_g, &zero_complex, output_g, &lda, &ldb, &ldc);
        ++idx;
      }
    }
  }

  LOG(ERROR) << "called cgemm";

  // set back to std stream.
  cublasSetStream(Caffe::cublas_handle(), NULL);

  for (int i = 0; i < streams_number; i++) {
    CUDA_CHECK(cudaStreamDestroy(stream[i]));
  }


  // result_dim = 256 x 129 x 96 x 10 ==> 96 x 10 x 256 x 129
  const int shape_result[] = {H, W, N, batch_size};
  const int permutation_result[] = {2, 3, 0, 1};
  fft_util_permute_4d_gpu(fft_transposed_result, ptwise_result, shape_result, permutation_result);
  CUDA_CHECK(cudaFree(fft_transposed_result));
}

template void fft_util_pointwise_multiply_gemm_gpu<float>(std::vector<int> shape, int group,
                                                          const std::complex<float> *bottom_complex,
                                                          const std::complex<float> *weight_complex,
                                                          std::complex<float> *ptwise_result);

template void fft_util_pointwise_multiply_gemm_gpu<double>(std::vector<int> shape, int group,
                                                           const std::complex<double> *bottom_complex,
                                                           const std::complex<double> *weight_complex,
                                                           std::complex<double> *ptwise_result);

template <typename Dtype>
void fft_util_normalize_gpu(std::vector<int> shape, const int kernel_h,
                            const int kernel_w, const int stride_h, const int stride_w,
                            float normalize_factor, int fft_height, int fft_width,
                            const Dtype *conv_result_real, Dtype *top_data) {

  const int batch_size = shape[0];
  const int N = shape[1];
  const int H = shape[2];
  const int W = shape[3];

  dim3 block_num(batch_size, N, (H * W / CAFFE_CUDA_NUM_THREADS) + 1);
  int thread_num = CAFFE_CUDA_NUM_THREADS;

  fft_util_normalize_gpu_kernel<<<block_num, thread_num>>>
      (N, H, W, kernel_h, kernel_w, stride_h, stride_w,
       normalize_factor, fft_height, fft_width,
       conv_result_real, top_data);
  CUDA_POST_KERNEL_CHECK;
}

template void fft_util_normalize_gpu<float>(std::vector<int> shape, const int kernel_h,
                                            const int kernel_w, const int stride_h, const int stride_w,
                                            float normalize_factor, int fft_height, int fft_width,
                                            const float *conv_result_real, float *top_data);

template void fft_util_normalize_gpu<double>(std::vector<int> shape, const int kernel_h,
                                            const int kernel_w, const int stride_h, const int stride_w,
                                            float normalize_factor, int fft_height, int fft_width,
                                            const double *conv_result_real, double *top_data);

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

}
