#include <vector>

#include "caffe/layers/add_layer.hpp"

namespace caffe {

template <typename Dtype>
void AddLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype *bottom_ptr = bottom[0]->gpu_data();

  Dtype *top_ptr = top[0]->mutable_gpu_data();
  // set mem to 0 before adding on top of it...
  //caffe_memset(num_ * num_output_ * height_out_  * width_out_ * sizeof(Dtype), 0., top_ptr);
  const Dtype *weight = this->blobs_[0]->gpu_data();


//  cudaStream_t *streams = new cudaStream_t[1024];

  // set gpu scalar pointer type to device
  cublasPointerMode_t old_pointer_mode;
  cublasGetPointerMode(Caffe::cublas_handle(), &old_pointer_mode);
  cublasSetPointerMode(Caffe::cublas_handle(), CUBLAS_POINTER_MODE_DEVICE);

//  for (int n = 0; n < 1024; ++n) {
//    CUDA_CHECK(cudaStreamCreate(&streams[n]));
//  }

  // TODO: optmize loops?
  for(int batch_idx = 0; batch_idx < num_; ++batch_idx) {
      for (int k = 0; k < channels_; ++k) {
        const Dtype *bottom_data = bottom_ptr + ((batch_idx * channels_ + k) * height_) * width_;
        for(int n = 0; n < num_output_; ++n) {
        Dtype *top_data = top_ptr + ((batch_idx * num_output_ + n) * height_) * width_;

        // Add result on top of existing top (multiply with weight is 1 or 0)
        const Dtype *alpha =  weight + (n * channels_ + k);

        // if (*alpha == 1.) {
          //int stream_idx = ((batch_idx * num_ + n) * channels_ + k) % 1024;
          //CUBLAS_CHECK(cublasSetStream(Caffe::cublas_handle(), streams[stream_idx]));
          caffe_gpu_axpy(height_ * width_, alpha, bottom_data, top_data);
        //}

      }
    }
  }

  // Reset gpu scalar pointer mode to old value:
  cublasSetPointerMode(Caffe::cublas_handle(), old_pointer_mode);
}

template <typename Dtype>
void AddLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

}

INSTANTIATE_LAYER_GPU_FUNCS(AddLayer);

}
