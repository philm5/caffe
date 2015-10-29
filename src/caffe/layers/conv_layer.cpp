#include <vector>
#include <caffe/util/fft_util.hpp>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

// #define WRITE_DEBUG_FW
#define WRITE_TOP_RES

template <typename Dtype>
void ConvolutionLayer<Dtype>::compute_output_shape() {
  this->height_out_ = (this->height_ + 2 * this->pad_h_ - this->kernel_h_)
      / this->stride_h_ + 1;
  this->width_out_ = (this->width_ + 2 * this->pad_w_ - this->kernel_w_)
      / this->stride_w_ + 1;
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  const Dtype* weight = this->blobs_[0]->cpu_data();

  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = top[i]->mutable_cpu_data();


    for (int n = 0; n < this->num_; ++n) {

#ifdef WRITE_DEBUG_FW
      double begin_clock = cpu_time();
#endif
      this->forward_cpu_gemm(bottom_data + bottom[i]->offset(n), weight,
                             top_data + top[i]->offset(n));
 #ifdef WRITE_DEBUG_FW
      double end_clock = cpu_time();
      LOG(ERROR) << this->layer_param().name() << ": " << 1000.0 * (end_clock - begin_clock) << " ms.";
 #endif

      //#define WRITE_TOP_RES
      #ifdef WRITE_TOP_RES
        std::stringstream ss;
        ss << "res_top_nor_" << this->layer_param_.name() << ".txt";
        const char *s = ss.str().c_str();
        this->write_simple_arr_to_disk(s, top[i]->count() , top[i]->cpu_data());
      #endif

      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->cpu_data();
        this->forward_cpu_bias(top_data + top[i]->offset(n), bias);
      }
    }
  }



  // this->write_simple_arr_to_disk("top_data_non_fft.txt", this->num_output_ * this->height_out_ * this->width_out_, top[0]->mutable_cpu_data());
}

  template <typename Dtype>
  void ConvolutionLayer<Dtype>::write_simple_arr_to_disk(const char* output_name, int size, const Dtype *arr)
  {
    std::ofstream fout(output_name); //opening an output stream for file test.txt
    if(fout.is_open())
    {
      //file opened successfully so we are here
      std::cout << "File Opened successfully!!!. Writing data from array to file" << std::endl;

      for(int i = 0; i < size; i++)
      {
        fout << *(arr + i) << "\n";
      }
      std::cout << "Array data successfully saved into the file " << output_name << std::endl;
    }

    fout.close();
  }

template <typename Dtype>
void ConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->cpu_diff();
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_cpu_bias(bias_diff, top_diff + top[i]->offset(n));
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_cpu_gemm(bottom_data + bottom[i]->offset(n),
              top_diff + top[i]->offset(n), weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_cpu_gemm(top_diff + top[i]->offset(n), weight,
              bottom_diff + bottom[i]->offset(n));
        }
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(ConvolutionLayer);
#endif

INSTANTIATE_CLASS(ConvolutionLayer);

}  // namespace caffe
