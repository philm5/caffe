#include "caffe/util/inference_layer_util.hpp"
#include "caffe/util/device_alternate.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

// kernel methods

template<typename Dtype>
__global__ void add_maps_with_offset_gpu_kernel(const Dtype *src_1, const Dtype *src_2, Dtype *dst,
                                                Dtype offset_x, Dtype offset_y,
                                                int map_height, int map_width) {


  const int hw = blockIdx.x * blockDim.x + threadIdx.x;

  if (hw < map_height * map_width) {

    const int y = hw / map_width;
    const int x = hw % map_width;

    Dtype src_x = x - offset_x;
    Dtype src_y = y - offset_y;

    if (src_x < 0 || src_y < 0 || src_x >= map_width || src_y >= map_height) {
      dst[y * map_width + x] = src_2[y * map_width + x];
    } else {
      Dtype out = 0.;
      // interpolate billinear pixel from src_1

      const int x1 = __float2int_rd(src_x);
      const int y1 = __float2int_rd(src_y);
      const int x2 = x1 + 1;
      const int y2 = y1 + 1;
      const int x2_read = ::min(x2, map_width - 1);
      const int y2_read = ::min(y2, map_height - 1);
      Dtype src_reg = src_1[y1 * map_width + x1];  //   src(y1, x1);
      out = out + src_reg * ((x2 - src_x) * (y2 - src_y));

      src_reg = src_1[y1 * map_width + x2_read];  //   src(y1, x2_read);
      out = out + src_reg * ((src_x - x1) * (y2 - src_y));

      src_reg = src_1[y2_read * map_width + x1];  //   src(y2_read, x1);
      out = out + src_reg * ((x2 - src_x) * (src_y - y1));

      src_reg = src_1[y2_read * map_width + x2_read];  //   src(y2_read, x2_read);
      out = out + src_reg * ((src_x - x1) * (src_y - y1));

      Dtype tmp = out + src_2[y * map_width + x];
      // printf("%f\n", tmp);
      dst[y * map_width + x] = tmp;
    }
  }
}

template
__global__ void add_maps_with_offset_gpu_kernel<float>(const float *src_1, const float *src_2, float *dst,
                                                       float offset_x, float offset_y,
                                                       int map_height, int map_width);

template
__global__ void add_maps_with_offset_gpu_kernel<double>(const double *src_1, const double *src_2, double *dst,
                                                        double offset_x, double offset_y,
                                                        int map_height, int map_width);

// end of kernel methods

template<typename Dtype>
void add_maps_with_offset_gpu(const Dtype *src_1, const Dtype *src_2, Dtype *dst,
                              Dtype offset_x, Dtype offset_y,
                              int map_height, int map_width) {

  dim3 block_num(map_height * map_width / CAFFE_CUDA_NUM_THREADS + 1);
  int thread_num = CAFFE_CUDA_NUM_THREADS;

  add_maps_with_offset_gpu_kernel<<<block_num, thread_num>>> (src_1, src_2, dst,
                                                              offset_x, offset_y,
                                                              map_height, map_width);

}

template void add_maps_with_offset_gpu<float>(const float *src_1, const float *src_2, float *dst,
                                              float offset_x, float offset_y,
                                              int map_height, int map_width);

template void add_maps_with_offset_gpu<double>(const double *src_1, const double *src_2, double *dst,
                                               double offset_x, double offset_y,
                                               int map_height, int map_width);

template<typename Dtype>
Dtype map_max_gpu(const Dtype *map, const int map_size) {
  int idx;
  Dtype res;
  caffe_gpu_amax(map_size, map, &idx);
  CUDA_CHECK(cudaMemcpy(&res, map + idx, sizeof(Dtype), cudaMemcpyDeviceToHost));
  return res;
}

template float map_max_gpu<float>(const float *map, const int map_size);

template double map_max_gpu<double>(const double *map, const int map_size);


template<typename Dtype>
Dtype map_max_gpu(const Dtype *map, int map_height, int map_width, int &max_x, int &max_y) {
  int idx;
  Dtype res;
  caffe_gpu_amax(map_height * map_width, map, &idx);

  max_x = idx % map_width;
  max_y = idx / map_width;

  CUDA_CHECK(cudaMemcpy(&res, map + idx, sizeof(Dtype), cudaMemcpyDeviceToHost));
  return res;
}

template float map_max_gpu<float>(const float *map, int map_height, int map_width, int &max_x, int &max_y);

template double map_max_gpu<double>(const double *map, int map_height, int map_width, int &max_x, int &max_y);

//template<typename Dtype>
//void add_layers_gpu(const Dtype *bottom, const Dtype *weight, Dtype *top,
//                    int num, int num_output, int channels, int height,
//                    int width) {
//
//  dim3 block_num(num, num_output, height * width / CAFFE_CUDA_NUM_THREADS + 1);
//  int thread_num = CAFFE_CUDA_NUM_THREADS;
//
//  add_layers_gpu_kernel<<<block_num, thread_num>>>(bottom, weight, top,
//                                                   channels, num_output, height,
//                                                   width);
//
//  // dim3 block_num(num, num_output, channels / CAFFE_CUDA_NUM_THREADS + 1);
//  // int thread_num = CAFFE_CUDA_NUM_THREADS;
//
//  // add_layers_gpu_kernel<<<block_num, thread_num>>>(bottom, weight, top,
//  //                                                  channels, num_output, height,
//  //                                                  width);
//
//}
//
//template void add_layers_gpu<float>(const float *bottom, const float *weight,
//                                    float *top, int num, int num_output,
//                                    int channels, int height, int width);
//
//template void add_layers_gpu<double>(const double *bottom, const double *weight,
//                                     double *top, int num, int num_output,
//                                     int channels, int height, int width);

}  // namespace caffe
