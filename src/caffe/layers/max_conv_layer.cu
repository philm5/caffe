#include <vector>

#include "caffe/layers/max_conv_layer.hpp"
#include "caffe/util/max_conv_util.hpp"

namespace caffe {

template <typename Dtype>
void MaxConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype *bottom_ptr = bottom[0]->gpu_data();
  const Dtype *kernel_weight_ptr = this->blobs_[0]->gpu_data();
  Dtype *top_ptr = top[0]->mutable_gpu_data();
  Dtype *top_origin = top[1]->mutable_gpu_data();

  fast_max_convolution_gpu(bottom_ptr, kernel_weight_ptr, top_ptr, top_origin, kernel_h_, kernel_w_,
                           num_, channels_, height_, width_);
}

template <typename Dtype>
void MaxConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

}

INSTANTIATE_LAYER_GPU_FUNCS(MaxConvolutionLayer);

}
