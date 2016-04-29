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

template<typename Dtype>
void ResizeLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                 const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(4, bottom[0]->num_axes())<< "Input must have 4 axes, "
  << "corresponding to (num, channels, height, width)";
  num_ = bottom[0]->num();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  channels_ = bottom[0]->channels();

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
  this->resize_cpu(bottom[0]->cpu_data(), top[0]->mutable_cpu_data());
}

template<typename Dtype>
void ResizeLayer<Dtype>::resize_cpu(const Dtype *bottom, Dtype *top) {

  for (int n = 0; n < num_ * num_output_; ++n) {
    for (int x = 0; x < width_out_; ++x) {
      for (int y = 0; y < height_out_; ++y) {

        const float src_x = x * scale_x_;
        const float src_y = y * scale_y_;

        int dst_x = x + shift_x_;
        int dst_y = y + shift_y_;

        const Dtype *src = bottom + n * height_ * width_;
        Dtype *dst = top + n * height_out_ * width_out_;

        if (src_x >= 0 && src_y >= 0 && src_x < width_ && src_y < height_
            && dst_x >= 0 && dst_y >= 0 && dst_x < width_out_
            && dst_y < height_out_) {
          Dtype out = 0.;  // = VecTraits < work_type > ::all(0);

          const int x1 = static_cast<int>(std::floor(src_x));
          const int y1 = static_cast<int>(std::floor(src_y));
          const int x2 = x1 + 1;
          const int y2 = y1 + 1;
          const int x2_read = std::min(x2, width_ - 1);
          const int y2_read = std::min(y2, height_ - 1);

          Dtype src_reg = src[y1 * width_ + x1];  //   src(y1, x1);
          out = out + src_reg * ((x2 - src_x) * (y2 - src_y));

          src_reg = src[y1 * width_ + x2_read];  //   src(y1, x2_read);
          out = out + src_reg * ((src_x - x1) * (y2 - src_y));

          src_reg = src[y2_read * width_ + x1];  //   src(y2_read, x1);
          out = out + src_reg * ((x2 - src_x) * (src_y - y1));

          src_reg = src[y2_read * width_ + x2_read];  //   src(y2_read, x2_read);
          out = out + src_reg * ((src_x - x1) * (src_y - y1));

          dst[dst_y * width_out_ + dst_x] = out;  //dst(dst_y, dst_x) = saturate_cast < T > (out);
        }
      }
    }
  }
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
