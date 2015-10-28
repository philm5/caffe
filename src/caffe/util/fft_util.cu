#include <algorithm>

#include <cmath>
#include <cstdlib>
#include <cstring>

#include "caffe/common.hpp"
#include "caffe/util/fft_util.hpp"
#include <cufft.h>
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
  const int k = blockIdx.y * blockDim.x + threadIdx.x;



  if (k < K) {

    // get offset with channels and the idx of the output.
    const int offset_weight_real = (n * K + k) * fft_real_size;
    const int offset_blob_real = (n * K + k) * H * W;

    for (int h = 0; h < H; h++) {
      for (int w = 0; w < W; w++) {
        // e.g. a 3x3 filter should fit into a 5x5 because the image size is 5x5
        // <--W-->
        // ^ f f f 0 0
        // H f f f 0 0
        // _ f f f 0 0
        //   0 0 0 0 0
        //   0 0 0 0 0

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

  // blockDim (256, 1) ----- (num_output_, (ch_gr) / CUDA_NUM_THREADS)
  //                              x                y
  const int n = blockIdx.x;

  // calculate the channel index. blockIdx.y is of size (H*W/CUDA_NUM_THREADS). So 1 or 2...
  const int hw = blockIdx.y * blockDim.x + threadIdx.x;


  if (hw < H*W) {

    // printf("<<<%i, %i>>>| n: %i k: %i K: %i\n", blockIdx.x, threadIdx.x, n, k, K);
    // check which group_ idx we are in
    const int group_idx = n / weight_group_size;

    // loop over channels
    for (int k = 0; k < K; ++k) {
      // get the input_k. this is the k we use to index the input k-dimension. the max input_k is group_ times more
      // than the max k of the weight.
      const int input_k = k + group_idx * K;
      const int weight_offset = (n * K + k);

      const int input_idx = input_k * H * W + hw;
      const cufftComplex input = ffted_bottom_data[input_idx];

      const int weight_idx = weight_offset * H * W + hw;
      const cufftComplex weight = weight_complex[weight_idx];

      // formula for complex mult from here: https://en.wikipedia.org/wiki/Complex_number#Multiplication_and_division
      // (a+bi) (c+di) = (ac-bd) + (bc+ad)i.
      float a = weight.x;
      float b = weight.y;
      float c = input.x;
      float d = input.y;

      const int res_idx = n * H * W + hw;
      ptwise_result[res_idx].x += a * c - b * d;
      ptwise_result[res_idx].y += b * c + a * d;
    }
  }
}

__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
                             (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

__global__ void fft_pointwise_multiply_double_gpu_kernel(const int N, const int K, const int H, const int W,
                                                         const int weight_group_size, const cufftDoubleComplex *ffted_bottom_data,
                                                         const cufftDoubleComplex *weight_complex, cufftDoubleComplex *ptwise_result) {

  // blockDim (256, 1) ----- (num_output_, (ch_gr) / CUDA_NUM_THREADS)
  //                              x                y
  const int n = blockIdx.x;

  // calculate the channel index. The blockIdx.y is usually zero. Because CUDA_THREADS = 512 > 48 = ch/gr. (48 / 512) + 1 = 1.
  const int k = blockIdx.y * blockDim.x + threadIdx.x;


  if (k < K) {
    // check which group_ idx we are in
    const int group_idx = n / weight_group_size;

    // get the input_k. this is the k we use to index the input k-dimension. the max input_k is group_ times more
    // than the max k of the weight.
    const int input_k = k + group_idx * K;
    const int weight_offset = (n * K + k);

    /* in the following loops every filter response is being calculated. there are num_output_ * (channels_ / group_) filters...
     * each (1/group_) part is multiplied with each part of the input. e.g. for group_ = 2, n_o_ 256 and c_ = 96:
     * weights dim: 256x48x5x5, input dim: 1x96x27x27 --> splitted into [1x48x27x27, 1x48x27x27]
     * first 128 weights [128x48x5x5] will be convolved with first part of weights (dimension match!) --> 128 responses
     * same for 2nd part --> 2x128 responses to forward to the next channel
     */
    for (int h = 0; h < H; ++h) {
      for (int w = 0; w < W; ++w) {
        // Indexing: ((n * K + k) * H + h) * W + w
        const int input_idx = (input_k * H + h) * W + w; // 4 ops
        const cufftDoubleComplex input = ffted_bottom_data[input_idx];

        const int weight_idx = (weight_offset * H + h) * W + w; // 4 ops
        const cufftDoubleComplex weight = weight_complex[weight_idx];

        // formula for complex mult from here: https://en.wikipedia.org/wiki/Complex_number#Multiplication_and_division
        // (a+bi) (c+di) = (ac-bd) + (bc+ad)i.
        double a = weight.x;
        double b = weight.y;
        double c = input.x;
        double d = input.y;

        const int res_idx = (n * H + h) * W + w; // 4 ops; before with channels: ((n * K + k) * H + h) * W + w;
        atomicAdd(&(ptwise_result[res_idx].x), a * c - b * d);
        atomicAdd(&(ptwise_result[res_idx].y), b * c + a * d);
      }
    }
  }
}

template <typename Dtype>
__global__ void fft_util_normalize_gpu_kernel(const int N, const int H, const int W, const int kernel_h,
                                              const int kernel_w, const int stride_h, const int stride_w,
                                              float normalize_factor, int fft_height, int fft_width,
                                              const Dtype *fft_convolution_result_real, Dtype *top_data) {

  // blockDim (256, 1) ----- (num_output_, (height_out) / CUDA_NUM_THREADS)
  //                              x                y
  const int n = blockIdx.x;

  // calculate the height index. The blockIdx.y is usually zero. Because CUDA_THREADS = 512 > 27 = ch/gr. (27 / 512) + 1 = 1.
  const int h = blockIdx.y * blockDim.x + threadIdx.x;

  if (h < H) {
    const int fft_real_size = fft_height * fft_width;
    const int offset_res_real = n * fft_real_size;
    // caffe does a valid convolution. fft is a full convolution. so the first 'valid' result is at
    // idx (kernel_h_ - 1). The stride times the idx of the output pixel will be added onto this.
    const int h_idx = (kernel_h - 1) + h * stride_h;

    for (int w = 0; w < W; ++w) // =55 in 1st layer
    {
      // caffe does a valid convolution. fft is a full convolution. so the first 'valid' result is at
      // idx (kernel_w_ - 1). The stride times the idx of the output pixel will be added onto this.
      const int w_idx = (kernel_w - 1) + w * stride_w;
      //((n * K + k) * H + h) * W + w;
      const int top_data_idx = (n * H + h) * W + w;

      // the index in the data of the convolution result array (the real one)
      const int res_data_idx = offset_res_real + h_idx * fft_width + w_idx;

      // normalize fft and sum up everything from the input channels...
      top_data[top_data_idx] = fft_convolution_result_real[res_data_idx] * normalize_factor;
    }
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
void pad_real_blob_gpu(std::vector<int> shape, const int fft_height, const int fft_width,
                       const Dtype *blob_data, Dtype *padded_data, const int pad_h,
                       const int pad_w, const bool flip) {

  const int N = shape[0];
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
  dim3 block_num(N, (K / CAFFE_CUDA_NUM_THREADS) + 1);
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
void fft_util_pointwise_multiply_gpu<float>(std::vector<int> shape, int group, const std::complex<float> *ffted_bottom_data,
                                            const std::complex<float> *weight_complex, std::complex<float> *ptwise_result) {
  const int N = shape[0];
  const int K = shape[1];
  const int H = shape[2];
  const int W = shape[3];

  const int weight_group_size = N / group;

  // N = 256 (num_output_)
  // K = 96 / 2 (channels / group) ==> (48 / 512 ) + 1 = 1
  // dim3 block_num(N, (K / CAFFE_CUDA_NUM_THREADS) + 1);

  // N = num_output, H * W as second dim so no races happen; because over K (channels) will be summed up
  // On thoise channels sum -ups the threads interfere with another...
  dim3 block_num(N, (H * W / CAFFE_CUDA_NUM_THREADS) + 1);
  int thread_num = CAFFE_CUDA_NUM_THREADS;

  const cufftComplex *ffted_bottom_data_cuda  = reinterpret_cast<const cufftComplex *> (ffted_bottom_data);
  const cufftComplex *weight_complex_cuda = reinterpret_cast<const cufftComplex *> (weight_complex);
  cufftComplex *ptwise_result_cuda = reinterpret_cast<cufftComplex *> (ptwise_result);

  fft_pointwise_multiply_float_gpu_kernel<<<block_num, thread_num>>>
      (N, K, H, W, weight_group_size, ffted_bottom_data_cuda, weight_complex_cuda, ptwise_result_cuda);
  CUDA_POST_KERNEL_CHECK;
}

template <>
void fft_util_pointwise_multiply_gpu<double>(std::vector<int> shape, int group, const std::complex<double> *ffted_bottom_data,
                                             const std::complex<double> *weight_complex, std::complex<double> *ptwise_result) {
  const int N = shape[0];
  const int K = shape[1];
  const int H = shape[2];
  const int W = shape[3];

  const int weight_group_size = N / group;

  // N = 256 (num_output_)
  // K = 96 / 2 (channels / group) ==> (48 / 512 ) + 1 = 1
  dim3 block_num(N, (K / CAFFE_CUDA_NUM_THREADS) + 1);
  int thread_num = CAFFE_CUDA_NUM_THREADS;

  const cufftDoubleComplex *ffted_bottom_data_cuda  = reinterpret_cast<const cufftDoubleComplex *> (ffted_bottom_data);
  const cufftDoubleComplex *weight_complex_cuda = reinterpret_cast<const cufftDoubleComplex *> (weight_complex);
  cufftDoubleComplex *ptwise_result_cuda = reinterpret_cast<cufftDoubleComplex *> (ptwise_result);

  fft_pointwise_multiply_double_gpu_kernel<<<block_num, thread_num>>>
      (N, K, H, W, weight_group_size, ffted_bottom_data_cuda, weight_complex_cuda, ptwise_result_cuda);
  CUDA_POST_KERNEL_CHECK;
}


template <typename Dtype>
void fft_util_pointwise_multiply_npp_gpu(std::vector<int> shape, int group, const std::complex<Dtype> *bottom_complex,
                                         const std::complex<Dtype> *weight_complex, std::complex<Dtype> *ptwise_result) {
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
                                                         const std::complex<float> *ffted_bottom_data,
                                                         const std::complex<float> *weight_complex,
                                                         std::complex<float> *ptwise_result);

template void fft_util_pointwise_multiply_npp_gpu<double>(std::vector<int> shape, int group,
                                                          const std::complex<double> *ffted_bottom_data,
                                                          const std::complex<double> *weight_complex,
                                                          std::complex<double> *ptwise_result);

template <typename Dtype>
void fft_util_normalize_gpu(std::vector<int> shape, const int kernel_h,
                            const int kernel_w, const int stride_h, const int stride_w,
                            float normalize_factor, int fft_height, int fft_width,
                            const Dtype *conv_result_real, Dtype *top_data) {

  // shape[0] is 0 here, because there is only one output image.
  const int N = shape[1];
  const int H = shape[2];
  const int W = shape[3];

  dim3 block_num(N, (H / CAFFE_CUDA_NUM_THREADS) + 1);
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
