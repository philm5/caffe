#include "caffe/layers/add_layer.hpp"

namespace caffe {

template<typename Dtype>
void AddLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                 const vector<Blob<Dtype>*>& top) {

  CHECK_EQ(4, bottom[0]->num_axes())<< "Input must have 4 axes, "
  << "corresponding to (num, channels, height, width)";
  num_ = bottom[0]->num();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  channels_ = bottom[0]->channels();

  AddParameter add_param = this->layer_param_.add_layer_param();

  // Configure output channels and groups.
  num_output_ = add_param.num_output();
  CHECK_GT(num_output_, 0) << "Number of outputs cannot be zero.";

  // Handle the parameters: weights.
  // - blobs_[0] holds the filter weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    this->blobs_.resize(1);
    // Initialize and fill the weights:
    // num_output_ x input channels x 1 x 1
    this->blobs_[0].reset(new Blob<Dtype>(num_output_, channels_, 1, 1));
//    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
//            this->layer_param_.convolution_param().weight_filler()));
//    weight_filler->Fill(this->blobs_[0].get());
  }
}

template <typename Dtype>
void AddLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
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
  compute_output_shape();
  for (int top_id = 0; top_id < top.size(); ++top_id) {
    top[top_id]->Reshape(num_, num_output_, height_out_, width_out_);
  }
}


template <typename Dtype>
void AddLayer<Dtype>::compute_output_shape() {
  this->height_out_ = this->height_;
  this->width_out_ = this->width_;
}

template<typename Dtype>
void AddLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                  const vector<Blob<Dtype>*>& top) {

  const Dtype *bottom_ptr = bottom[0]->cpu_data();
  Dtype *top_ptr = top[0]->mutable_cpu_data();
  // set mem to 0 before adding on top of it...
  //caffe_memset(num_ * num_output_ * height_out_  * width_out_ * sizeof(Dtype), 0., top_ptr);
  const Dtype *weight = this->blobs_[0]->cpu_data();

  // TODO: optmize loops?
  for(int batch_idx = 0; batch_idx < num_; ++batch_idx) {
    for(int n = 0; n < num_output_; ++n) {
      for (int k = 0; k < channels_; ++k) {
        const Dtype *bottom_data = bottom_ptr + ((batch_idx * channels_ + k) * height_) * width_;
        Dtype *top_data = top_ptr + ((batch_idx * num_output_ + n) * height_) * width_;

        // Add result on top of existing top (multiply with weight is 1 or 0)
        caffe_axpy(height_ * width_, weight[(n * channels_ + k)], bottom_data, top_data);
      }
    }
  }
}

template<typename Dtype>
void AddLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                   const vector<bool>& propagate_down,
                                   const vector<Blob<Dtype>*>& bottom) {

}

#ifdef CPU_ONLY
//STUB_GPU(AddLayer);
#endif

INSTANTIATE_CLASS(AddLayer);
REGISTER_LAYER_CLASS(Add);

}  // namespace caffe
