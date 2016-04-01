#include <vector>

#include "caffe/layers/add_layer.hpp"
#include "caffe/util/add_layer_util.hpp"

namespace caffe {

template <typename Dtype>
void AddLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype *bottom_ptr = bottom[0]->gpu_data();
  Dtype *top_ptr = top[0]->mutable_gpu_data();
  const Dtype *weight = this->blobs_[0]->gpu_data();

  add_layers_gpu(bottom_ptr, weight, top_ptr, num_, num_output_, channels_, height_, width_);
}

template <typename Dtype>
void AddLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

}

INSTANTIATE_LAYER_GPU_FUNCS(AddLayer);

}
