#include "caffe/util/max_conv_util.hpp"
#include "caffe/util/device_alternate.hpp"
#include <float.h>

namespace caffe {

// kernel methods

template<typename Dtype>
__global__ void max_convolution_gpu_kernel(const Dtype *bottom,
                                           const Dtype *weight, Dtype *top,
                                           int kernel_h, int kernel_w,
                                           int kernel_radius, int channels,
                                           int height, int width);

template<>
__global__ void max_convolution_gpu_kernel<float>(const float *bottom,
                                                  const float *weight,
                                                  float *top, int kernel_h,
                                                  int kernel_w, int kernel_radius,
                                                  int channels, int height,
                                                  int width) {

  const int batch_idx = blockIdx.x;
  const int k = blockIdx.y;
  const int hw = blockIdx.z * blockDim.x + threadIdx.x;

  if (hw < height * width) {
    // do sth...
    int y = hw / width;
    int x = hw % width;

    const float *bottom_data = bottom
        + ((batch_idx * channels + k) * height) * width;
    const float *kernel_weight_data = weight + k * kernel_h * kernel_w;
    float *top_data = top + ((batch_idx * channels + k) * height) * width;

    float max_val = FLT_MIN;
    for (int q = 0; q < kernel_h; ++q) {
      for (int r = 0; r < kernel_w; ++r) {
        int y_bottom = y + (q - kernel_radius);
        int x_bottom = x + (r - kernel_radius);

        // ignore borders...
        if (!(y_bottom < 0 || x_bottom < 0 || y_bottom >= height
            || x_bottom >= width)) {
          // - for max conv and + for min conv ???
          float tmp = bottom_data[y_bottom * width + x_bottom]
              - kernel_weight_data[q * kernel_w + r];
          max_val = fmaxf(tmp, max_val);
        }
      }
    }
    top_data[y * width + x] = max_val;
  }
}

template<>
__global__ void max_convolution_gpu_kernel<double>(const double *bottom,
                                                   const double *weight,
                                                   double *top, int kernel_h,
                                                   int kernel_w,
                                                   int kernel_radius,
                                                   int channels, int height,
                                                   int width) {

}
// end of kernel methods

template<typename Dtype>
void max_convolution_gpu(const Dtype *bottom, const Dtype *weight, Dtype *top,
                         int kernel_h, int kernel_w, int num, int channels,
                         int height, int width) {
  dim3 block_num(num, channels, (height * width) / CAFFE_CUDA_NUM_THREADS);
  int thread_num = CAFFE_CUDA_NUM_THREADS; //CAFFE_CUDA_NUM_THREADS;

  int k_radius = (kernel_h - 1) / 2;  // we only support quadratic kernels and assume uneven filter sizes...

  max_convolution_gpu_kernel<<<block_num, thread_num>>>(bottom, weight, top,
                                                        kernel_h, kernel_w,
                                                        k_radius, channels,
                                                        height, width);

}

template void max_convolution_gpu<float>(const float *bottom,
                                         const float *weight, float *top,
                                         int kernel_h, int kernel_w, int num,
                                         int channels, int height, int width);

template void max_convolution_gpu<double>(const double *bottom,
                                          const double *weight, double *top,
                                          int kernel_h, int kernel_w, int num,
                                          int channels, int height, int width);

}  // namespace caffe
