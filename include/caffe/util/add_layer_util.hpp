#ifndef ADD_LAYER_UTIL_HPP_
#define ADD_LAYER_UTIL_HPP_

namespace caffe {

template<typename Dtype>
void add_layers_gpu(const Dtype *bottom, const Dtype *weight, Dtype *top,
                    int num, int num_output, int channels, int height,
                    int width);

}

#endif /* ADD_LAYER_UTIL_HPP_ */
