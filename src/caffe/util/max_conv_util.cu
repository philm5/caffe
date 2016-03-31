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
                                                int tile_dim, int tiles_per_row, int shared_mem_length);


template<>
__global__ void fast_max_convolution_gpu_kernel<float>(const float *bottom,
                                                       const float *weight, float *top,
                                                       int kernel_h, int kernel_w,
                                                       int kernel_radius, int channels,
                                                       int height, int width,
                                                       int tile_dim, int tiles_per_row, int shared_mem_length) {
  const int batch_idx = blockIdx.x;
  const int k = blockIdx.y;

  const float *bottom_data = bottom
      + ((batch_idx * channels + k) * height) * width;

  // tiles are quadratically... get initial offset...
  const int tile_idx = blockIdx.z; // * blockDim.x + threadIdx.x;
  const int y_tile_idx = (tile_idx / tiles_per_row);
  const int x_tile_idx = (tile_idx % tiles_per_row);


  printf("tile_idx %d, y %d, x %d\n", tile_idx, y_tile_idx, x_tile_idx);

  const int y_offset = y_tile_idx * tile_dim;
  const int x_offset = x_tile_idx * tile_dim;

  const int y_start = y_offset - kernel_radius;
  const int x_start = x_offset - kernel_radius;

  // we have to copy the values from offset (X) into shared mem:
  // *************** (kernel_radius) **************|stride! = width|
  // (kernel_radius)X --- tile_dim -▶(kernel_radius)
  // ***************|                **************
  // ***************| tile_dim       **************
  // ***************▼               ◢**************
  // *************** (kernel_radius) **************


  // We store a tile of the bottom data here + the kernel
  extern __shared__ float shared_data[];

  // every thread has to copy some amount of values from global to shared memory
  const int shared_data_size = shared_mem_length * shared_mem_length;
  const int copy_count = (shared_data_size / blockDim.x) + 1; // calculated outside !? e.g. [(50*50)/512] + 1 = 5

  // offset of current thread inside the shared memory...
  const int inner_offset = threadIdx.x * copy_count;

  for (int i = 0; i < copy_count; ++i) {
    const int copy_offset = (inner_offset + i);
    const int shared_mem_y_offset = copy_offset / shared_mem_length;
    const int shared_mem_x_offset = copy_offset % shared_mem_length;

    // case III: we are at the end of a tile (including apron!) --> set x to x_start and current y to +=1
    const int src_y = y_start + shared_mem_y_offset; // handles case III
    const int src_x = x_start + shared_mem_x_offset; // handles case III

   // don't copy data outside of the allocated space
    if (copy_offset < shared_data_size) {
      // find destination ptr
      float *dst = shared_data + copy_offset;
      // case I  : we are at the end of a row/column in the image? --> fill 0s into shared mem
      // case II : we are before the beginning of a row/column in a the image? --> fill 0s into shared mem
      if (src_y < 0 || src_x < 0 || src_x > width || src_y > height) {
        *dst = 0.;
      } else {
        *dst = bottom_data[(src_y * width) + src_x];
      }
    }
  }

  // write kernel to shared mem
  // ------------------------------
  // we use one array in the shared memory for both data and kernel (we only dynamically alloc one array from outside the kernel)
  float *kernel = shared_data + shared_data_size;
  const int k_shared_mem_size = kernel_h * kernel_w;
  const int k_copy_count = (k_shared_mem_size / blockDim.x) + 1; // calculated outside !? e.g. [(50*50)/512] + 1 = 5
  // offset of current thread inside the kernel shared memory...
  const int k_inner_offset = threadIdx.x * k_copy_count;

  // offset to kernel
  const float *kernel_weight_data = weight + k * kernel_h * kernel_w;

  for (int i = 0; i < k_copy_count; ++i) {
    int idx = k_inner_offset + i;
    if (idx < k_shared_mem_size) {
      kernel[idx] = kernel_weight_data[idx];
    }
  }

  // sync shared memory...
  __syncthreads();

  const int hw = threadIdx.x;

  if (hw < tile_dim * tile_dim) {
    int y = (hw / tile_dim);
    int x = (hw % tile_dim);

    float max_val = FLT_MIN;
    for (int q = 0; q < kernel_h; ++q) {
      for (int r = 0; r < kernel_w; ++r) {
        int y_bottom = y + q;
        int x_bottom = x + r;

          // - for max conv and + for min conv ???
          float tmp = shared_data[y_bottom * shared_mem_length + x_bottom]
              - kernel[q * kernel_w + r];
          max_val = fmaxf(tmp, max_val);
      }
    }

    // we write in global memory, because we only acces each location once
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
                                                        int tile_dim, int tiles_per_row, int shared_mem_length) {
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

#define TITAN_NUM_THREADS 1024

  int tile_dim = sqrt(TITAN_NUM_THREADS);
  const int tiles_per_row = width / tile_dim + 1;
  const int tiles_per_col = height / tile_dim + 1;
  printf("tpr %d, tpc %d\n", tiles_per_row, tiles_per_col);
  int z_blocks = (tiles_per_col  * tiles_per_row);

  dim3 block_num(num, channels, z_blocks);
  int thread_num = TITAN_NUM_THREADS;

  int k_radius = (kernel_h - 1) / 2;  // we only support quadratic kernels and assume uneven filter sizes...

  const int shared_mem_length = (tile_dim + 2 * k_radius);
  // size of bottom data region + kernel region
  int shared_mem_size = (shared_mem_length * shared_mem_length + kernel_h * kernel_w) * sizeof(Dtype);

  fast_max_convolution_gpu_kernel<<<block_num, thread_num, shared_mem_size>>>(bottom, weight, top,
                                                            kernel_h, kernel_w,
                                                            k_radius, channels,
                                                            height, width, tile_dim, tiles_per_row, shared_mem_length);

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
