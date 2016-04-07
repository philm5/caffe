#ifndef CAFFE_RESIZE_LAYER_HPP_
#define CAFFE_RESIZE_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
namespace caffe {


template <typename Dtype>
class ResizeLayer : public Layer<Dtype> {
 public:
  explicit ResizeLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "ResizeLayer"; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  // int kernel_h_, kernel_w_;
  // int stride_h_, stride_w_;
  float scale_x_, scale_y_;
  int shift_x_, shift_y_;
  int num_;
  int channels_;
  int scale_h_, scale_w_;
  int height_, width_;
  // int group_;
  int num_output_;
  int height_out_, width_out_;
};
}
#endif // CAFFE_ADD_LAYER_HPP_
