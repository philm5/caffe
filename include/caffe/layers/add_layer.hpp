#ifndef CAFFE_ADD_LAYER_HPP_
#define CAFFE_ADD_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/loss_layers.hpp"
#include "caffe/neuron_layers.hpp"
#include "caffe/proto/caffe.pb.h"
namespace caffe {


template <typename Dtype>
class AddLayer : public Layer<Dtype> {
 public:
  explicit AddLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "AddLayer"; }

 protected:
  virtual void  compute_output_shape();
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  // int kernel_h_, kernel_w_;
  // int stride_h_, stride_w_;
  int num_;
  int channels_;
  // int pad_h_, pad_w_;
  int height_, width_;
  // int group_;
  int num_output_;
  int height_out_, width_out_;
};
}
#endif // CAFFE_ADD_LAYER_HPP_
