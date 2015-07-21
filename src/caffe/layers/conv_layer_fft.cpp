#include <vector>
#include <complex>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/fft_util.hpp"
#include "caffe/vision_layers.hpp"
#include "fftw3.h"

#include <iostream> // library that contain basic input/output functions
#include <fstream>  // library that contains file input/output functions
#include <ctime>

namespace caffe 
{
	template <typename Dtype>
	void ConvolutionLayerFFT<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top)
	{
            ConvolutionLayer<Dtype>::LayerSetUp(bottom, top);
            LOG(ERROR) << "called base\nConvolutionLayerFFT::LayerSetUp"; 

	}
        
        template <typename Dtype>
        void ConvolutionLayerFFT<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
        {            
            ConvolutionLayer<Dtype>::Reshape(bottom, top);
            LOG(ERROR) << "called base\nConvolutionLayerFFT::Reshape"; 
            this->fft_set_up();
        }    
        
        template <typename Dtype>
        void ConvolutionLayerFFT<Dtype>::fft_set_up()
        {            
            // here only the memory should be reserved... The weight data will 
            // either be trained or loaded from disk. So there is no data available
            // yet...
            LOG(ERROR) << "ConvolutionLayerFFT::fft_set_up"; 
            
            // set fft width and height to be the image width and height (for now without padding...)
            // Check if width and height is a power of 2. If not, round up to the next power of 2.
            if(!check_power_of_2(this->width_))
            {
                this->fft_width_ = next_power_of_2(this->width_);
            }
            else
            {
                this->fft_width_ = this->width_;
            }

            if(!check_power_of_2(this->height_))
            {
                this->fft_height_ = next_power_of_2(this->height_);
            }
            else
            {
                this->fft_height_ = this->height_;
            }
            
            // for sizes see: http://www.fftw.org/doc/Multi_002dDimensional-DFTs-of-Real-Data.html
            this->fft_real_size_ = this->fft_height_ * this->fft_width_;
            this->fft_complex_size_ = this->fft_height_ * (this->fft_width_ / 2 + 1);

            // Allocations & plan for weights
            int num_weights = this->num_output_ * (this->channels_ / this->group_);
            
            int weight_alloc_size_in = this->fft_real_size_ * num_weights * sizeof(Dtype);
            fft_weights_in_real_ = reinterpret_cast<Dtype *>(fft_cpu_malloc<Dtype>(weight_alloc_size_in));
            
            int weight_alloc_size_out = this->fft_complex_size_ * num_weights * sizeof(std::complex<Dtype>);
            fft_weights_out_complex_ = reinterpret_cast<std::complex<Dtype> *>(fft_cpu_malloc<Dtype>(weight_alloc_size_out));

            // The plan. Is a plan for the actual conversion. Conversion will be done when weights are rdy...
            fft_weight_plan_ = fft_plan_many_dft_r2c_2d<Dtype>(fft_height_, fft_width_, num_weights, fft_weights_in_real_, fft_weights_out_complex_, FFTW_ESTIMATE);

            // ---------------------------------------------------
            // Allocations & plan for input values (bottom values)
            int K = (this->channels_ / this->group_); // the size of channels, second dim...

            int alloc_size_input_real = this->fft_real_size_ * 1 *  K  * sizeof(Dtype);
            fft_input_real_ = reinterpret_cast<Dtype *>(fft_cpu_malloc<Dtype>(alloc_size_input_real));

            int alloc_size_input_complex = this->fft_complex_size_ * 1 * K * sizeof(std::complex<Dtype>);
            fft_input_complex_ = reinterpret_cast<std::complex<Dtype> *>(fft_cpu_malloc<Dtype>(alloc_size_input_complex));

            // The plan to compute the input values to complex
            fft_input_plan_ = fft_plan_many_dft_r2c_2d<Dtype>(fft_height_, fft_width_, K, fft_input_real_, fft_input_complex_, FFTW_ESTIMATE);
        }
        
        template <typename Dtype>        
        void ConvolutionLayerFFT<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
        {
            // my custom fft code    
            LOG(ERROR) << "ConvolutionLayerFFT::Forward_cpu. now ffting the weights to complex....";
            this->convert_weights_fft();
            this->convert_bottom(bottom);

            LOG(ERROR) << "converted...?";



            LOG(ERROR) << "calling base: ConvolutionLayer::Forward_cpu";       
            // inherited things to do...
            ConvolutionLayer<Dtype>::Forward_cpu(bottom, top);
        }

        template <typename Dtype>
        void ConvolutionLayerFFT<Dtype>::convert_bottom(const vector<Blob<Dtype>*>& bottom)
        {
            auto input_data_blob = bottom[0]->cpu_data();
            auto shape =  bottom[0]->shape();

            LOG(ERROR) << "start of conversion of bottom data...";
            this->transform_blob_to_real_array(shape[0], shape[1], shape[2], shape[3], input_data_blob, fft_input_real_);

            caffe_memset(this->fft_complex_size_* shape[0] * shape[1] * sizeof(std::complex<Dtype>), 0., fft_input_complex_);

            clock_t begin_clock = std::clock();
            fft_execute_plan<Dtype>(this->fft_input_plan_);
            clock_t end_clock = std::clock();

            LOG(ERROR) << "fft for one layer took " << 1000.0 * (end_clock-begin_clock) / CLOCKS_PER_SEC << " ms.";

        }

        template <typename Dtype>
        void ConvolutionLayerFFT<Dtype>::transform_blob_to_real_array(int N, int K, int H, int W, const Dtype *blob_data, Dtype *padded_real_data)
        {
            int num_arr = N * K; // # of arrays (for weights it is num_weights [96 x 3]
                                 //              for input data it is channels [ 1 x 3]


            // set everything to 0 before --> so not set weights are 0-padded :)
            caffe_memset(this->fft_real_size_ * num_arr * sizeof(Dtype), 0., padded_real_data);

            for (int n = 0; n < N; n++)
            {
                for (int k = 0; k < K; k++)
                {
                    // ((n * ch_gr + c) * fft_height_ + h)* 2 * (fft_width_ / 2 + 1) + w
                    int offset_weight_real = (n * K + k) * this->fft_real_size_ ;
                    for (int h = 0; h < H; h++)
                    {
                        for (int w = 0; w < W; w++)
                        {
                            // e.g. a 3x3 filter should fit into a 5x5 because the image size is 5x5
                            // <--W-->
                            // ^ f f f 0 0
                            // H f f f 0 0
                            // _ f f f 0 0
                            //   0 0 0 0 0
                            //   0 0 0 0 0

                            int idx_weight_real =  offset_weight_real + h * this->fft_width_ + w;
                            // copy each weight into the fft_weights_in_real_
                            // get ptr to blob data. indexing see: http://caffe.berkeleyvision.org/tutorial/net_layer_blob.html
                            // Blob memory is row-major in layout, so the last / rightmost dimension changes fastest. For example,
                            // in a 4D blob, the value at index (n, k, h, w) is physically located at index ((n * K + k) * H + h) * W + w.
                            // 96 x 3 x 11 x 11 (num_output_, channels_ / group_, kernel_height, width_)
                            int idx_weight_in_blob = ((n * K + k) * H + h) * W + w;

                            padded_real_data[idx_weight_real] = blob_data[idx_weight_in_blob];
                        }
                    }
                }
            }
        }

        template <typename Dtype>
        void ConvolutionLayerFFT<Dtype>::convert_weights_fft() {
            if (weights_converted) {
                // if weights were converted alrdy don't do that again :)
                return;
            }

            // the data location of the weights before the conversion...
            auto cpu_data = this->blobs_[0]->cpu_data();

            // transform data to real
            int N = this->num_output_;
            int K = (this->channels_ / this->group_);
            int H = this->kernel_h_;
            int W = this->kernel_w_;
            this->transform_blob_to_real_array(N, K, H, W, cpu_data, fft_weights_in_real_);

            // set complex mem to 0
            caffe_memset(this->fft_complex_size_ * N * K * sizeof(std::complex<Dtype>), 0.,
                         this->fft_weights_out_complex_);

            clock_t begin_clock = std::clock();
            fft_execute_plan<Dtype>(this->fft_weight_plan_);
            clock_t end_clock = std::clock();

            LOG(ERROR) << "fft for one layer took " << 1000.0 * (end_clock - begin_clock) / CLOCKS_PER_SEC << " ms.";
            weights_converted = true;

            // output the weights to txt:
            // this->write_arr_to_disk("real_in.txt", N * K, this->fft_weights_in_real_);
            // this->write_arr_to_disk("complex_out.txt", N * K, this->fft_weights_out_complex_, true);
        }

        template <typename Dtype>
        void ConvolutionLayerFFT<Dtype>::write_arr_to_disk(const char* output_name, int size, void *arr, bool is_complex)
        {
            std::ofstream fout(output_name); //opening an output stream for file test.txt
            if(fout.is_open())
            {
                //file opened successfully so we are here
                std::cout << "File Opened successfully!!!. Writing data from array to file" << std::endl;

                int size_multiplier = is_complex ? this->fft_complex_size_ : this->fft_real_size_;
                for(int i = 0; i < size_multiplier * size; i++)
                {
                    if (is_complex)
                    {
                        auto arr_conv = reinterpret_cast<std::complex<Dtype> *>(arr);
                        fout << arr_conv[i].real() << ";" << arr_conv[i].imag() << "\n";
                    }
                    else
                    {
                        auto arr_conv = reinterpret_cast<Dtype *>(arr);
                        fout << arr_conv << "\n";
                    }
                }
                std::cout << "Array data successfully saved into the file " << output_name << std::endl;
            }

            fout.close();
        }

//#ifdef CPU_ONLY
//	STUB_GPU(ConvolutionLayerFFT);
//#endif

	INSTANTIATE_CLASS(ConvolutionLayerFFT);

}  // namespace caffe
