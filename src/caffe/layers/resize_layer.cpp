#include "caffe/layers/resize_layer.hpp"

namespace caffe {

template<typename Dtype>
void ResizeLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                 const vector<Blob<Dtype>*>& top) {

  CHECK_EQ(4, bottom[0]->num_axes())<< "Input must have 4 axes, "
  << "corresponding to (num, channels, height, width)";
  num_ = bottom[0]->num();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  channels_ = bottom[0]->channels();

  ResizeParameter resize_param = this->layer_param_.resize_param();

  shift_x_ = resize_param.shift_x();
  shift_y_ = resize_param.shift_y();

  // Configure output size and groups.
  height_out_ = resize_param.height_out();
  width_out_ = resize_param.width_out();
  num_output_ = channels_;


  // Set scale to 0 in case it is not set, so we do NOT use the scale parameter.
  if (resize_param.has_scale_x()) {
    scale_x_ = 1. / resize_param.scale_x();
    CHECK_GT(scale_x_, 0) << "Scale factor (x) must be greater than 0";
    scale_w_ = width_ / scale_x_;
  } else {
    // calculate scale factor based on in and output sizes!
    scale_x_ = static_cast<float>(width_) / width_out_;
    scale_w_ = width_out_;
  }
  if (resize_param.has_scale_y()) {
    scale_y_ = 1. / resize_param.scale_y();
    CHECK_GT(scale_y_, 0) << "Scale factor (y) must be greater than 0";
    scale_h_ = height_ / scale_y_;
  } else {
    // calculate scale factor based on in and output sizes!
    scale_y_ = static_cast<float>(height_) / height_out_;
    scale_h_ = height_out_;
  }

  // no weights in this layer...
}

template <typename Dtype>
void ResizeLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
      << "corresponding to (num, channels, height, width)";
  num_ = bottom[0]->num();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  channels_ = bottom[0]->channels();
  // TODO: generalize to handle inputs of different shapes.
  for (int bottom_id = 1; bottom_id < bottom.size(); ++bottom_id) {
    CHECK_EQ(num_, bottom[bottom_id]->num()) << "Inputs must have same num.";
    CHECK_EQ(channels_, bottom[bottom_id]->channels())
        << "Inputs must have same channels.";
    CHECK_EQ(height_, bottom[bottom_id]->height())
        << "Inputs must have same height.";
    CHECK_EQ(width_, bottom[bottom_id]->width())
        << "Inputs must have same width.";
  }
  // Shape the tops.
  for (int top_id = 0; top_id < top.size(); ++top_id) {
    top[top_id]->Reshape(num_, num_output_, height_out_, width_out_);
  }
}

template<typename Dtype>
void ResizeLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                  const vector<Blob<Dtype>*>& top) {
  // implement resize for cpu...

}

template<typename Dtype>
void ResizeLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                   const vector<bool>& propagate_down,
                                   const vector<Blob<Dtype>*>& bottom) {

}

#ifdef CPU_ONLY
STUB_GPU(ResizeLayer);
#endif

INSTANTIATE_CLASS(ResizeLayer);
REGISTER_LAYER_CLASS(Resize);

}  // namespace caffe
