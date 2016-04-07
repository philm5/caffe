#include <vector>

#include "caffe/layers/resize_layer.hpp"
#include "caffe/util/resize_layer_util.hpp"

namespace caffe {

template <typename Dtype>
void ResizeLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype *bottom_ptr = bottom[0]->gpu_data();
  Dtype *top_ptr = top[0]->mutable_gpu_data();

  resize_linear_gpu(bottom_ptr, top_ptr, num_, num_output_, height_, width_, height_out_, width_out_,
                    scale_y_, scale_x_, shift_y_, shift_x_);
}

template <typename Dtype>
void ResizeLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

}

INSTANTIATE_LAYER_GPU_FUNCS(ResizeLayer);

}
