#ifndef MAX_CONV_UTIL_HPP_
#define MAX_CONV_UTIL_HPP_

namespace caffe {

template <typename Dtype>
void max_convolution_gpu(const Dtype *bottom, const Dtype *weight, Dtype *top, Dtype *top_origin,
                         int kernel_h, int kernel_w, int num, int channels, int height, int width);
template <typename Dtype>
void fast_max_convolution_gpu(const Dtype *bottom, const Dtype *weight, Dtype *top, Dtype *top_origin,
                              int kernel_h, int kernel_w, int num, int channels, int height, int width);


}



#endif /* MAX_CONV_UTIL_HPP_ */
