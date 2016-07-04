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

#define INFERENCE_WEIGHT_OFFSET 3

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
  virtual void check_bottom_top(const vector<Blob<Dtype>*>& bottom,
                                const vector<Blob<Dtype>*>& top);
  virtual void compute_output_shape();
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
     const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
     const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  virtual void add_maps_with_offset(const Dtype *src_1, const Dtype *src_2, Dtype *dst,
                                    Dtype offset_x, Dtype offset_y);
  virtual Dtype map_max_cpu(const Dtype *map);
  virtual Dtype map_max_cpu(const Dtype *map, int &max_x, int &max_y);

  inline int get_child_idpr_map_idx_cpu(int which_map, const Dtype *connection_weights) {
    int idx = INFERENCE_WEIGHT_OFFSET + which_map;
    return static_cast<int>(connection_weights[idx]);
  }

  inline int get_parent_idpr_map_idx_cpu(int which_map, int num_clusters, const Dtype *connection_weights) {
    int idx = INFERENCE_WEIGHT_OFFSET + num_clusters + which_map;
    return static_cast<int>(connection_weights[idx]);
  }

  inline const Dtype *get_child_ipdr_map_offset(int which_map, int num_clusters, const Dtype *connection_weights) {
    int idx = INFERENCE_WEIGHT_OFFSET + 2 * num_clusters + 2 * which_map;
    return connection_weights + idx;

  }


  shared_ptr<Blob<Dtype> > tmp_map_;
  shared_ptr<Blob<Dtype> > tmp2_map_;
  shared_ptr<Blob<Dtype> > max_map_;
  int num_;
  int num_appearance_maps_;
  int num_idpr_maps_;
  int height_, width_;
  int num_connections_;
  int num_output_;
  int height_out_, width_out_;
  int map_size_;
};
}
#endif // CAFFE_INFERENCE_LAYER_HPP_
