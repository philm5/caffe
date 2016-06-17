#include "caffe/layers/inference_layer.hpp"

namespace caffe {

template<typename Dtype>
void InferenceLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                 const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(2, bottom.size()) << "Must have two bottom blobs...";

  CHECK_EQ(bottom[0]->shape(0), bottom[1]->shape(0)) << "The two bottom blobs need to have the same batch size (dimension 0)";
  CHECK_EQ(bottom[0]->shape(2), bottom[1]->shape(2)) << "The two bottom blobs need to have the same height (dimension 2)";
  CHECK_EQ(bottom[0]->shape(3), bottom[1]->shape(3)) << "The two bottom blobs need to have the same width (dimension 3)";

  LOG(ERROR) << "bottom[0] shape: " << bottom[0]->shape(0) << " " << bottom[0]->shape(1) << " " << bottom[0]->shape(2) << " " << bottom[0]->shape(3);
  LOG(ERROR) << "bottom[1] shape: " << bottom[1]->shape(0) << " " << bottom[1]->shape(1) << " " << bottom[1]->shape(2) << " " << bottom[1]->shape(3);

  CHECK_EQ(4, bottom[0]->num_axes())<< "First input must have 4 axes, "
  << "corresponding to (num, channels, height, width)";
  CHECK_EQ(4, bottom[1]->num_axes())<< "Second input must have 4 axes, "
  << "corresponding to (num, channels, height, width)";

  num_ = bottom[0]->num();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();

  num_appearance_maps_ = bottom[0]->channels();
  num_idpr_maps_ = bottom[1]->channels();

  InferenceParameter inference_param = this->layer_param_.inference_param();
  num_connections_ = inference_param.num_connections();

  CHECK_GT(num_connections_, 0) << "Number of connections cannot be zero.";

  // Handle the parameters: weights.
  // - blobs_[0] holds the filter weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
   this->blobs_.resize(3);

   // 1st weights: appearance weights = num_appearance_maps_ * single float
   vector<int> appearance_weights_shape;
   appearance_weights_shape.push_back(num_appearance_maps_);
   this->blobs_[0].reset(new Blob<Dtype>(appearance_weights_shape));

   // 2nd weights: idpr weights       = num_idpr_maps_ * single float
   vector<int> idpr_weights_shape;
   idpr_weights_shape.push_back(num_idpr_maps_);
   this->blobs_[1].reset(new Blob<Dtype>(idpr_weights_shape));


   // 3rd weights: inference order weights = num_connections_ * (3 + 2 * num_idpr_maps_)
   //              for every connection: (x, y, num_clusters, <cluster indices of joint x>, <cluster indices of joint y>)
   vector<int> inference_weights_shape;
   inference_weights_shape.push_back(num_connections_);
   inference_weights_shape.push_back(3 + 2 * num_idpr_maps_);
   this->blobs_[2].reset(new Blob<Dtype>(inference_weights_shape));
  //    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
  //            this->layer_param_.convolution_param().weight_filler()));
  //    weight_filler->Fill(this->blobs_[0].get());
  }




//   num_ = bottom[0]->num();
//   height_ = bottom[0]->height();
//   width_ = bottom[0]->width();
//   channels_ = bottom[0]->channels();

//   AddParameter add_param = this->layer_param_.add_layer_param();

//   // Configure output channels and groups.
//   num_output_ = add_param.num_output();
//   CHECK_GT(num_output_, 0) << "Number of outputs cannot be zero.";

//   // Handle the parameters: weights.
//   // - blobs_[0] holds the filter weights
//   if (this->blobs_.size() > 0) {
//     LOG(INFO) << "Skipping parameter initialization";
//   } else {
//     this->blobs_.resize(1);
//     // Initialize and fill the weights:
//     // num_output_ x input channels x 1 x 1
//     this->blobs_[0].reset(new Blob<Dtype>(num_output_, channels_, 1, 1));
// //    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
// //            this->layer_param_.convolution_param().weight_filler()));
// //    weight_filler->Fill(this->blobs_[0].get());
//   }
}

template <typename Dtype>
void InferenceLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
//   CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
//       << "corresponding to (num, channels, height, width)";
//   num_ = bottom[0]->num();
//   height_ = bottom[0]->height();
//   width_ = bottom[0]->width();
//   channels_ = bottom[0]->channels();
//   // TODO: generalize to handle inputs of different shapes.
//   for (int bottom_id = 1; bottom_id < bottom.size(); ++bottom_id) {
//     CHECK_EQ(num_, bottom[bottom_id]->num()) << "Inputs must have same num.";
//     CHECK_EQ(channels_, bottom[bottom_id]->channels())
//         << "Inputs must have same channels.";
//     CHECK_EQ(height_, bottom[bottom_id]->height())
//         << "Inputs must have same height.";
//     CHECK_EQ(width_, bottom[bottom_id]->width())
//         << "Inputs must have same width.";
//   }
//   // Shape the tops.
//   compute_output_shape();
//   for (int top_id = 0; top_id < top.size(); ++top_id) {
//     top[top_id]->Reshape(num_, num_output_, height_out_, width_out_);
//   }
}


template <typename Dtype>
void InferenceLayer<Dtype>::compute_output_shape() {
//   this->height_out_ = this->height_;
//   this->width_out_ = this->width_;
}

template<typename Dtype>
void InferenceLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                  const vector<Blob<Dtype>*>& top) {

//   const Dtype *bottom_ptr = bottom[0]->cpu_data();
//   Dtype *top_ptr = top[0]->mutable_cpu_data();
//   // set mem to 0 before adding on top of it...
//   caffe_memset(num_ * num_output_ * height_out_  * width_out_ * sizeof(Dtype), 0., top_ptr);
//   const Dtype *weight = this->blobs_[0]->cpu_data();

//   // TODO: optmize loops?
//   for(int batch_idx = 0; batch_idx < num_; ++batch_idx) {
//     for(int n = 0; n < num_output_; ++n) {
//       for (int k = 0; k < channels_; ++k) {
//         const Dtype *bottom_data = bottom_ptr + ((batch_idx * channels_ + k) * height_) * width_;
//         Dtype *top_data = top_ptr + ((batch_idx * num_output_ + n) * height_) * width_;
        


//         // Add result on top of existing top (if alpha == 1.)
//         const Dtype *alpha = weight + (n * channels_ + k);
//         if (*alpha == 1.) {
//           for (int idx = 0; idx < height_ * width_; ++idx) {
//             top_data[idx] += bottom_data[idx];
//           }
//         }
// //
// //
// //        // Add result on top of existing top (multiply with weight is 1 or 0)
// //        caffe_axpy(height_ * width_, weight[(n * channels_ + k)], bottom_data, top_data);
//       }
//     }
//   }
}

template<typename Dtype>
void InferenceLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                   const vector<bool>& propagate_down,
                                   const vector<Blob<Dtype>*>& bottom) {

}

#ifdef CPU_ONLY
STUB_GPU(InferenceLayer);
#endif

INSTANTIATE_CLASS(InferenceLayer);
REGISTER_LAYER_CLASS(Inference);

}  // namespace caffe
