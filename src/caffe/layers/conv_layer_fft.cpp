#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
#include "fftw3.h"

namespace caffe 
{
	template <typename Dtype>
	void ConvolutionLayerFFT<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top)
	{
            ConvolutionLayer<Dtype>::LayerSetUp(bottom, top);
            LOG(ERROR) << "FFT Layer performing set up..."; 

	}
        
        template <typename Dtype>
        void ConvolutionLayerFFT<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
        {            
            ConvolutionLayer<Dtype>::Reshape(bottom, top);
            const Dtype* weight = this->blobs_[0]->cpu_data();
            LOG(ERROR) << *weight;

//		fftw_complex *in, *out;
//		fftw_plan p;
//
//		in = (fftw_complex *)fftw_malloc(sizeof(fftw_complex)* 1000);
//		out = (fftw_complex *)fftw_malloc(sizeof(fftw_complex)* 1000);
//
//		p = fftw_plan_dft_1d(1000, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
//
//		fftw_execute(p);
//		fftw_destroy_plan(p);
//
//		fftw_free(in);
//		fftw_free(out);
        }


//#ifdef CPU_ONLY
//	STUB_GPU(ConvolutionLayerFFT);
//#endif

	INSTANTIATE_CLASS(ConvolutionLayerFFT);

}  // namespace caffe
