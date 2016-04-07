#ifndef RESIZE_LAYER_UTIL_HPP_
#define RESIZE_LAYER_UTIL_HPP_

namespace caffe {

// copied from opencv cuda code
__host__ __device__ __forceinline__ int divUp(int total, int grain)
{
    return (total + grain - 1) / grain;
}

template<typename Dtype>
void resize_linear_gpu(const Dtype *bottom, Dtype *top,
                       int num, int num_output, int height,
                       int width, int height_out, int width_out,
                       float fy, float fx, int shift_x, int shift_y);

}

#endif /* RESIZE_LAYER_UTIL_HPP_ */
