#include "caffe/util/resize_layer_util.hpp"
#include "caffe/util/device_alternate.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

// kernel methods

// code taken from opencv cuda resize method and modified
template<typename Dtype>
__global__ void resize_linear_kernel(const Dtype *bottom, Dtype *top,
                                     int height_in, int width_in,
                                     int height_out, int width_out,
                                     const float fy, const float fx,
                                     const int shift_y, const int shift_x) {

  const int n = blockIdx.z;
  int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
  int dst_y = blockDim.y * blockIdx.y + threadIdx.y;

  const float src_x = dst_x * fx;
  const float src_y = dst_y * fy;

  dst_x += shift_x;
  dst_y += shift_y;

  const Dtype *src = bottom + n * height_in * width_in;
  Dtype *dst = top + n * height_out * width_out;

  if (src_x >= 0 && src_y >= 0 && src_x < width_in && src_y < height_in && dst_x >= 0 && dst_y >= 0 && dst_x < width_out && dst_y < height_out) {

    Dtype out = 0.;  // = VecTraits < work_type > ::all(0);

    const int x1 = __float2int_rd(src_x);
    const int y1 = __float2int_rd(src_y);
    const int x2 = x1 + 1;
    const int y2 = y1 + 1;
    const int x2_read = ::min(x2, width_in - 1);
    const int y2_read = ::min(y2, height_in - 1);

    Dtype src_reg = src[y1 * width_in + x1];  //   src(y1, x1);
    out = out + src_reg * ((x2 - src_x) * (y2 - src_y));

    src_reg = src[y1 * width_in + x2_read];  //   src(y1, x2_read);
    out = out + src_reg * ((src_x - x1) * (y2 - src_y));

    src_reg = src[y2_read * width_in + x1];  //   src(y2_read, x1);
    out = out + src_reg * ((x2 - src_x) * (src_y - y1));

    src_reg = src[y2_read * width_in + x2_read];  //   src(y2_read, x2_read);
    out = out + src_reg * ((src_x - x1) * (src_y - y1));

    dst[dst_y * width_out + dst_x] = out;  //dst(dst_y, dst_x) = saturate_cast < T > (out);
  }
}

template __global__ void resize_linear_kernel<float>(const float *bottom,
                                                     float *top, int height_in,
                                                     int width_in,
                                                     int height_out,
                                                     int width_out,
                                                     const float fy,
                                                     const float fx,
                                                     const int shift_y, const int shift_x);

template __global__ void resize_linear_kernel<double>(const double *bottom,
                                                      double *top,
                                                      int height_in,
                                                      int width_in,
                                                      int height_out,
                                                      int width_out,
                                                      const float fy,
                                                      const float fx,
                                                      const int shift_y, const int shift_x);

// end of kernel methods

template<typename Dtype>
void resize_linear_gpu(const Dtype *bottom, Dtype *top, int num, int num_output,
                       int height, int width, int height_out, int width_out,
                       float fy, float fx, int shift_x, int shift_y) {
  const dim3 block(32, CAFFE_CUDA_NUM_THREADS / 32);
  const dim3 grid(divUp(width_out, block.x),
                  divUp(height_out, block.y), num * num_output);
  resize_linear_kernel<<<grid, block>>>(bottom, top, height, width, height_out,
                                        width_out, fy, fx, shift_y, shift_x);

}

template void resize_linear_gpu<float>(const float *bottom, float *top, int num,
                                       int num_output, int height, int width,
                                       int height_out, int width_out,
                                       float fy, float fx, int shift_x, int shift_y);

template void resize_linear_gpu<double>(const double *bottom, double *top,
                                        int num, int num_output, int height,
                                        int width, int height_out,
                                        int width_out,
                                        float fy, float fx, int shift_x, int shift_y);

}  // namespace caffe
