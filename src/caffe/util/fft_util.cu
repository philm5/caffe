#include <algorithm>

#include <cmath>
#include <cstdlib>
#include <cstring>

#include "caffe/common.hpp"
#include "caffe/util/fft_util.hpp"

namespace caffe {
template <typename Dtype>
__global__ void pad_real_blob_gpu_kernel(const int K, const int H, const int W, const int fft_height, const int fft_width,
                                         const int fft_real_size, const Dtype *blob_data, Dtype *padded_data,
                                         const int pad_h, const int pad_w, const bool flip) {

  // blockDim (256, 1) ----- (num_output_, (ch_gr) / CUDA_NUM_THREADS)
  //                              x                y

  int out = blockIdx.x;

  // calculate the channel index. The blockIdx is usually zero. Because CUDA_THREADS = 512 > 48 = ch/gr. (48 / 512) + 1 = 1.
  const int channelIdx = blockIdx.y * blockDim.x + threadIdx.x;

  // get offset with channels and the idx of the output.
  const int offset_weight_real = (out * K + channelIdx) * fft_real_size;
  const int offset_blob_real = (out * K + channelIdx) * H * W;

  if (channelIdx < K) {
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

}

template void pad_real_blob_gpu<float>(std::vector<int> shape, const int fft_height, const int fft_width,
                                       const float *blob_data, float *padded_data, const int pad_h,
                                       const int pad_w, const bool flip);

template void pad_real_blob_gpu<double>(std::vector<int> shape, const int fft_height, const int fft_width,
                                        const double *blob_data, double *padded_data, const int pad_h,
                                        const int pad_w, const bool flip);
}
