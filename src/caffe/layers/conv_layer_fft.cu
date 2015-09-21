#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

// CUDA Header includes
#include <cuda.h>

namespace caffe {

template <typename Dtype>
void ConvolutionLayerFFT<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                             const vector<Blob<Dtype>*>& top)
{
  size_t totalGlobalMem;
  size_t freeMem;

  cuMemGetInfo(&freeMem,&totalGlobalMem);
  printf("  Free memory:     %4.4f MB\n", (float)freeMem/(1024*1024));
}


template <typename Dtype>
void ConvolutionLayerFFT<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                              const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

}

INSTANTIATE_LAYER_GPU_FUNCS(ConvolutionLayerFFT);

}  // namespace caffe
