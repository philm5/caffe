#ifndef CAFFE_MAXCONV_LAYER_HPP_
#define CAFFE_MAXCONV_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"


namespace caffe {


template <typename Dtype>
class MaxConvolutionLayer : public Layer<Dtype> {
 public:
  explicit MaxConvolutionLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "MaxConvolution"; }

 protected:
  virtual void  compute_output_shape();
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  bool max_conv_; // do max or min conv?
  bool scale_term_; // should we scale inputs with a factor before doing the maxconv?
                    // corresponding weights are in blobs_[1].
  int kernel_h_, kernel_w_;
  int num_;
  int channels_;
  int height_, width_;
  int num_output_;
  int height_out_, width_out_;
};
}
#endif // CAFFE_MAXCONV_LAYER_HPP_
