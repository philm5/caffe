#include "caffe/layers/max_conv_layer.hpp"

namespace caffe {

template<typename Dtype>
void MaxConvolutionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                            const vector<Blob<Dtype>*>& top) {

  CHECK_EQ(4, bottom[0]->num_axes())<< "Input must have 4 axes, "
  << "corresponding to (num, channels, height, width)";
  num_ = bottom[0]->num();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  channels_ = bottom[0]->channels();

  MaxConvolutionParameter max_param = this->layer_param_.max_convolution_param();

  // Configure output channels and groups.
  kernel_h_ = max_param.kernel_size();
  kernel_w_ = max_param.kernel_size();
  max_conv_ = max_param.max_conv();
  scale_term_ = max_param.scale_term();
  CHECK_GT(kernel_h_, 0) << "Kernel size cannot be zero.";

  // Handle the parameters: weights.
  // - blobs_[0] holds the filter weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (scale_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }

    // Initialize and fill the weights:
    // 1 x input channels x kernel_h x kernel_w
    this->blobs_[0].reset(new Blob<Dtype>(1, channels_, kernel_h_, kernel_w_));

    // If necessary, initialize and fill the scale terms.
    if (scale_term_) {
      vector<int> bias_shape(1, channels_);
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
    }
  }
}

template<typename Dtype>
void MaxConvolutionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                         const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(4, bottom[0]->num_axes())<< "Input must have 4 axes, "
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
  compute_output_shape();
  for (int top_id = 0; top_id < top.size(); ++top_id) {
    top[top_id]->Reshape(num_, num_output_, height_out_, width_out_);
  }
}

template<typename Dtype>
void MaxConvolutionLayer<Dtype>::compute_output_shape() {
  this->height_out_ = this->height_;
  this->width_out_ = this->width_;
  this->num_output_ = this->channels_;  // output size is same as input dims...
}

template<typename Dtype>
void MaxConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                             const vector<Blob<Dtype>*>& top) {
  const Dtype *bottom_ptr = bottom[0]->cpu_data();
  const Dtype *kernel_weight_ptr = this->blobs_[0]->cpu_data();
  Dtype *top_ptr = top[0]->mutable_cpu_data();

  int k_half = (kernel_h_ - 1 ) / 2; // we only support quadratic kernels and assume uneven filter sizes...

  for(int batch_idx = 0; batch_idx < num_; ++batch_idx) {
    for(int k = 0; k < channels_; ++k) {
      // ((n * K + k) * H + h) * W + w
      const Dtype *bottom_data = bottom_ptr + ((batch_idx * channels_ + k) * height_) * width_;
      const Dtype *kernel_weight_data = kernel_weight_ptr + k * kernel_h_ * kernel_w_;
      Dtype *top_data = top_ptr + ((batch_idx * channels_ + k) * height_) * width_;

      // max-convolution / dt
      for (int y = 0; y < height_; ++y) {
        for (int x = 0; x < width_; ++x) {
          Dtype max_val = boost::math::tools::min_value<Dtype>();
          for (int q = 0; q < kernel_h_; ++q) {
            for (int r = 0; r < kernel_w_; ++r) {
              int y_bottom = y + (q - k_half);
              int x_bottom = x + (r - k_half);

              // ignore borders...
              if (! (y_bottom < 0 || x_bottom < 0 || y_bottom >= height_ || x_bottom >= width_)) {
                // - for max conv and + for min conv ???
                Dtype tmp = bottom_data[y_bottom * width_ + x_bottom] - kernel_weight_data[q * kernel_w_ + r];
                max_val = std::max(tmp, max_val);
              }
            }
          }

          top_data[y * width_ + x] = max_val;
        }
      }

    }
  }
  LOG(ERROR) << "MaxConv Forward!";
}

template<typename Dtype>
void MaxConvolutionLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

}

#ifdef CPU_ONLY
STUB_GPU(MaxConvolutionLayer);
#endif

INSTANTIATE_CLASS(MaxConvolutionLayer);
REGISTER_LAYER_CLASS(MaxConvolution);

}  // namespace caffe
