#include "caffe/util/add_layer_util.hpp"
#include "caffe/util/device_alternate.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

// kernel methods

template<typename Dtype>
__global__ void add_layers_gpu_kernel(const Dtype *bottom, const Dtype *weight,
                                      Dtype *top, int channels, int num_output,
                                      int height, int width);

template<>
__global__ void add_layers_gpu_kernel<float>(const float *bottom,
                                             const float *weight, float *top,
                                             int channels, int num_output,
                                             int height, int width) {
  const int batch_idx = blockIdx.x;
  const int n = blockIdx.y;
  const int k = blockIdx.z * blockDim.x + threadIdx.x;

  if (k < channels) {
    const float *bottom_data = bottom
        + ((batch_idx * channels + k) * height) * width;
    float *top_data = top + ((batch_idx * num_output + n) * height) * width;

    // Add result on top of existing top (multiply with weight is 1 or 0)
    const float *alpha = weight + (n * channels + k);
    // caffe_gpu_axpy(height * width, alpha, bottom_data, top_data);
  }
}

template<>
__global__ void add_layers_gpu_kernel<double>(const double *bottom,
                                              const double *weight, double *top,
                                              int channels, int num_output,
                                              int height, int width) {

}
// end of kernel methods

template<typename Dtype>
void add_layers_gpu(const Dtype *bottom, const Dtype *weight, Dtype *top,
                    int num, int num_output, int channels, int height,
                    int width) {

  dim3 block_num(num, num_output, channels / CAFFE_CUDA_NUM_THREADS + 1);
  int thread_num = CAFFE_CUDA_NUM_THREADS;

  add_layers_gpu_kernel<<<block_num, thread_num>>>(bottom, weight, top,
                                                   channels, num_output, height,
                                                   width);

}

template void add_layers_gpu<float>(const float *bottom, const float *weight,
                                    float *top, int num, int num_output,
                                    int channels, int height, int width);

template void add_layers_gpu<double>(const double *bottom, const double *weight,
                                     double *top, int num, int num_output,
                                     int channels, int height, int width);

}  // namespace caffe
