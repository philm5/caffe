#ifndef INFERENCE_LAYER_UTIL_HPP_
#define INFERENCE_LAYER_UTIL_HPP_

namespace caffe {


// GPU Methods

template<typename Dtype>
void add_maps_with_offset_gpu(const Dtype *src_1, const Dtype *src_2, Dtype *dst,
                              Dtype offset_x, Dtype offset_y,
                              int map_height, int map_width);

template<typename Dtype>
Dtype map_max_gpu(const Dtype *map, const int map_size);

template<typename Dtype>
Dtype map_max_gpu(const Dtype *map, int map_height, int map_width, int &max_x, int &max_y);

}

#endif /* INFERENCE_LAYER_UTIL_HPP_ */
