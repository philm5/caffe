#ifndef CAFFE_INFERENCE_LAYER_HPP_
#define CAFFE_INFERENCE_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
namespace caffe {


template <typename Dtype>
class InferenceLayer : public Layer<Dtype> {
 public:
  explicit InferenceLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "InferenceLayer"; }

 protected:
  virtual void  compute_output_shape();
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
//   virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
//       const vector<Blob<Dtype>*>& top);
//   virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
//       const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int num_;
  int num_appearance_maps_;
  int num_idpr_maps_;
  int height_, width_;
  int num_connections_;
  int num_output_;
  int height_out_, width_out_;
};
}
#endif // CAFFE_INFERENCE_LAYER_HPP_
