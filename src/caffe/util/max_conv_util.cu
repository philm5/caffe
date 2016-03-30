#include "caffe/util/max_conv_util.hpp"
#include "caffe/util/device_alternate.hpp"
#include <float.h>
#include <stdio.h>

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
 // TODO: implement double version...
}

template<typename Dtype>
__global__ void fast_max_convolution_gpu_kernel(const Dtype *bottom,
                                                const Dtype *weight, Dtype *top,
                                                int kernel_h, int kernel_w,
                                                int kernel_radius, int channels,
                                                int height, int width,
                                                int tile_dim, int shared_mem_length);


template<>
__global__ void fast_max_convolution_gpu_kernel<float>(const float *bottom,
                                                       const float *weight, float *top,
                                                       int kernel_h, int kernel_w,
                                                       int kernel_radius, int channels,
                                                       int height, int width,
                                                       int tile_dim, int shared_mem_length) {
  const int batch_idx = blockIdx.x;
  const int k = blockIdx.y;

  const float *bottom_data = bottom
      + ((batch_idx * channels + k) * height) * width;

  // tiles are quadratically... get initial offset...
  const int tile_idx = blockIdx.z; // * blockDim.x + threadIdx.x;
  const int tiles_per_row = width / tile_dim + 1; // calculate outside?
  const int y_offset = (tile_idx / tiles_per_row) * tile_dim;
  const int x_offset = (tile_idx % tiles_per_row) * tile_dim;

  const int y_start = y_offset - kernel_radius;
  const int x_start = x_offset - kernel_radius;

  // we have to copy the values from offset (X) into shared mem:
  // *************** (kernel_radius) **************|stride! = width|
  // (kernel_radius)X --- tile_dim -▶(kernel_radius)
  // ***************|                **************
  // ***************| tile_dim       **************
  // ***************▼               ◢**************
  // *************** (kernel_radius) **************

  // every thread has to copy some amount of values from global to shared memory
  __shared__ float shared_data[4356];

  const int shared_data_size = shared_mem_length * shared_mem_length;
  const int copy_count = (shared_data_size / blockDim.x) + 1; // calculated outside !? e.g. [(50*50)/512] + 1 = 5


  // offset of current thread inside the global memory...
  const int inner_offset = threadIdx.x * copy_count;


  //printf("BLOCK(%d, %d, %d) -- threadIdx: %d, xstart: %d, ystart: %d, inneroffset: %d\n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, x_start, y_start, inner_offset);

  for (int i = 0; i < copy_count; ++i) {
    const int copy_offset = (inner_offset + i);
    const int shared_mem_y_offset = copy_offset / shared_mem_length;
    const int shared_mem_x_offset = copy_offset % shared_mem_length;
    //printf("copy_offset %d, shared_mem_len %d -- y_off: %d, x_off: %d\n", copy_offset, shared_mem_length, shared_mem_y_offset, shared_mem_x_offset);

    // case III: we are at the end of a tile (including apron!) --> set x to x_start and current y to +=1
    const int src_y = y_start + shared_mem_y_offset; // handles case III
    const int src_x = x_start + shared_mem_x_offset; // handles case III



    // printf("copy_offset: %d, shared_data_size: %d, shared_data[0] = %f\n", copy_offset, shared_data_size, shared_data[0]);

   // don't copy data outside of the allocated space
    if (copy_offset < shared_data_size) {
      // find destination ptr
      //float *dst = shared_data + copy_offset;

      //printf("y, x -- shared_data[%d] = %f -- y_off: %d, x_off: %d\n", src_y, src_x, copy_offset, dst, shared_mem_y_offset, shared_mem_x_offset);

      // case I  : we are at the end of a row/column in the image? --> fill 0s into shared mem
      // case II : we are before the beginning of a row/column in a the image? --> fill 0s into shared mem
      if (src_y < 0 || src_x < 0 || src_x > width || src_y > height) {
        shared_data[copy_offset] = 0.;
      } else {
        float tmp = bottom_data[(src_y * width) + src_x];

        shared_data[copy_offset] = tmp;
      }
      //printf("copy_off: %d\n", copy_offset);

    }
  }

  // sync shared memory...
  __syncthreads();

  const int hw = threadIdx.x;

  if (hw < tile_dim * tile_dim) {
    int y = (hw / tile_dim); // + kernel_radius;
    int x = (hw % tile_dim); // + kernel_radius;

    // TODO: copy to constant memory?
    const float *kernel_weight_data = weight + k * kernel_h * kernel_w;

    float max_val = FLT_MIN;
    for (int q = 0; q < kernel_h; ++q) {
      for (int r = 0; r < kernel_w; ++r) {
        int y_bottom = y + q;
        int x_bottom = x + r;

          // - for max conv and + for min conv ???
          float tmp = shared_data[y_bottom * shared_mem_length + x_bottom]
              - kernel_weight_data[q * kernel_w + r];
          max_val = fmaxf(tmp, max_val);
      }
    }

    // make sure not to write outside top!!!
    float *top_data = top + ((batch_idx * channels + k) * height) * width;
    const int top_idx = (y + y_offset) * width + (x + x_offset);
    top_data[top_idx] = max_val;
  }
}


template<>
__global__ void fast_max_convolution_gpu_kernel<double>(const double *bottom,
                                                        const double *weight, double *top,
                                                        int kernel_h, int kernel_w,
                                                        int kernel_radius, int channels,
                                                        int height, int width,
                                                        int tile_dim, int shared_mem_length) {
  // TODO: implement double version...
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

template<typename Dtype>
void fast_max_convolution_gpu(const Dtype *bottom, const Dtype *weight, Dtype *top,
                         int kernel_h, int kernel_w, int num, int channels,
                         int height, int width) {

  int tile_dim = 32; //sqrt(CAFFE_CUDA_NUM_THREADS);
  int z_blocks = ((height / tile_dim) + 1)  * ((width / tile_dim) + 1);

  dim3 block_num(num, channels, z_blocks);
  int thread_num = 1024; //CAFFE_CUDA_NUM_THREADS;

  int k_radius = (kernel_h - 1) / 2;  // we only support quadratic kernels and assume uneven filter sizes...

  const int shared_mem_length = (tile_dim + 2 * k_radius);
  int shared_mem_size = shared_mem_length * shared_mem_length;

  fast_max_convolution_gpu_kernel<<<block_num, thread_num, shared_mem_size>>>(bottom, weight, top,
                                                            kernel_h, kernel_w,
                                                            k_radius, channels,
                                                            height, width, tile_dim, shared_mem_length);

}

template void fast_max_convolution_gpu<float>(const float *bottom,
                                              const float *weight, float *top,
                                              int kernel_h, int kernel_w, int num,
                                              int channels, int height, int width);

template void fast_max_convolution_gpu<double>(const double *bottom,
                                               const double *weight, double *top,
                                               int kernel_h, int kernel_w, int num,
                                               int channels, int height, int width);

}  // namespace caffe
