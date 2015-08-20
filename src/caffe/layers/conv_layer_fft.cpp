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

#ifdef _OPENMP
#include <omp.h>
#endif

namespace caffe 
{
    template <typename  Dtype>
    ConvolutionLayerFFT<Dtype>::~ConvolutionLayerFFT()
    {
        LOG(ERROR) << "called ~ConvolutionLayerFFT()... ";

        fft_cpu_free<Dtype>(fft_weights_out_complex_);

        // TODO: delete all plans? all memory?

        LOG(ERROR) << "complex weights freed.";
    }

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

            if (!this->weights_converted_)
            {
                this->fft_set_up();
            }
        }    
        
        template <typename Dtype>
        void ConvolutionLayerFFT<Dtype>::fft_set_up()
        {            
            // here only the memory should be reserved... The weight data will 
            // either be trained or loaded from disk. So there is no data available
            // yet...
            LOG(ERROR) << "ConvolutionLayerFFT::fft_set_up";

            // ---- openmp ------------------------------------------
            num_of_threads_ = 1;
#ifdef _OPENMP
  num_of_threads_ = omp_get_max_threads();
  if (num_of_threads_ < 1)
  {
     LOG(WARNING) << "FFT Convolution Layer: omp_get_max_threads() =" << num_of_threads_;
     num_of_threads_ = 1;
  }
#endif
            
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
            
            weight_alloc_size_in = this->fft_real_size_ * num_weights * sizeof(Dtype);
            fft_weights_in_real_ = reinterpret_cast<Dtype *>(fft_cpu_malloc<Dtype>(weight_alloc_size_in));
            
            weight_alloc_size_out = this->fft_complex_size_ * num_weights * sizeof(std::complex<Dtype>);
            fft_weights_out_complex_ = reinterpret_cast<std::complex<Dtype> *>(fft_cpu_malloc<Dtype>(weight_alloc_size_out));

            // The plan. Is a plan for the actual conversion. Conversion will be done when weights are rdy...
            fft_weight_plan_ = fft_plan_many_dft_r2c_2d<Dtype>(fft_height_, fft_width_, num_weights, fft_weights_in_real_, fft_weights_out_complex_, FFTW_ESTIMATE);
        }
        
        template <typename Dtype>        
        void ConvolutionLayerFFT<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
        {
            LOG(ERROR) << "ConvolutionLayerFFT::Forward_cpu";

            // Allocations & plan for input values (bottom values)
            int alloc_size_input_real = this->fft_real_size_ * 1 *  this->channels_ * sizeof(Dtype);
            int alloc_size_input_complex = this->fft_complex_size_ * 1 * this->channels_ * sizeof(std::complex<Dtype>);
            fft_input_real_ = reinterpret_cast<Dtype *>(fft_cpu_malloc<Dtype>(alloc_size_input_real));
            fft_input_complex_ = reinterpret_cast<std::complex<Dtype> *>(fft_cpu_malloc<Dtype>(alloc_size_input_complex));

            // ---------------------------------------------------
            // The plan to compute the input values to complex
            int num_weights = this->num_output_ * (this->channels_ / this->group_);
            int K = (this->channels_ / this->group_); // the size of channels, second dim...
            fft_input_plan_ = fft_plan_many_dft_r2c_2d<Dtype>(fft_height_, fft_width_, K, fft_input_real_, fft_input_complex_, FFTW_ESTIMATE);

            this->convert_weights_fft();
            this->convert_bottom(bottom);

            // alloc data for result
            fft_conv_result_complex_ = reinterpret_cast<std::complex<Dtype> *>(fft_cpu_malloc<Dtype>(weight_alloc_size_out));

            clock_t begin_clock = std::clock();
            this->convolve_fft();
            clock_t end_clock = std::clock();
            LOG(ERROR) << "fft convolve took " << 1000.0 * (end_clock-begin_clock) / CLOCKS_PER_SEC << " ms.";

            // free the memory used only once per forward call (the input memory was already used in convolve_fft)
            fft_cpu_free<Dtype>(fft_input_complex_);

            // alloc data for real result
            fft_conv_result_real_ = reinterpret_cast<Dtype *>(fft_cpu_malloc<Dtype>(weight_alloc_size_in));

            // The ifft plan; the backward conversion from c2r
            ifft_plan_ = fft_plan_many_dft_c2r_2d<Dtype>(fft_height_, fft_width_, num_weights, fft_conv_result_complex_, fft_conv_result_real_, FFTW_ESTIMATE);

            begin_clock = std::clock();
            // execute the actual ifft
            fft_execute_plan<Dtype>(this->ifft_plan_);
            end_clock = std::clock();

            // free the complex result, because it was already ifft-ed to real.
            fft_cpu_free<Dtype>(fft_conv_result_complex_);

            LOG(ERROR) << "ifft took " << 1000.0 * (end_clock-begin_clock) / CLOCKS_PER_SEC << " ms.";

            this->normalize_ifft_result(top);

            // free the real result memory
            fft_cpu_free<Dtype>(fft_conv_result_real_);

            //  bias
            for (int i = 0; i < bottom.size(); ++i)
            {
                Dtype* top_data = top[i]->mutable_cpu_data();
                for (int n = 0; n < this->num_; ++n)
                {
                    if (this->bias_term_) {
                        const Dtype* bias = this->blobs_[1]->cpu_data();
                        this->forward_cpu_bias(top_data + top[i]->offset(n), bias);
                    }
                }
            }
#ifdef WRITE_ARRAYS_T0_DISK
            int N = this->num_output_;
            int K = (this->channels_ / this->group_);
            this->write_arr_to_disk("conv_res_real.txt", N*K , this->fft_conv_result_real_);
            this->write_simple_arr_to_disk("top_data.txt", this->num_output_ * this->height_out_ * this->width_out_, top[0]->mutable_cpu_data());
#endif
        }

        template <typename Dtype>
        void ConvolutionLayerFFT<Dtype>::convert_bottom(const vector<Blob<Dtype>*>& bottom)
        {
            const Dtype *input_data_blob = bottom[0]->cpu_data();
            vector<int> shape = bottom[0]->shape();

            LOG(ERROR) << "start of conversion of bottom data...";
            this->transform_blob_to_real_array(shape[0], shape[1], shape[2], shape[3], input_data_blob, this->fft_input_real_);

            caffe_memset(this->fft_complex_size_* shape[0] * shape[1] * sizeof(std::complex<Dtype>), 0., this->fft_input_complex_);

            clock_t begin_clock = std::clock();
            fft_execute_plan<Dtype>(this->fft_input_plan_);
            clock_t end_clock = std::clock();

            fft_cpu_free<Dtype>(fft_input_real_);

            LOG(ERROR) << "fft for bottom data took " << 1000.0 * (end_clock-begin_clock) / CLOCKS_PER_SEC << " ms.";

#ifdef WRITE_ARRAYS_T0_DISK
            this->write_arr_to_disk("input_real_in.txt", shape[0] * shape[1], this->fft_input_real_);
            this->write_arr_to_disk("input_complex_out.txt", shape[0] * shape[1], this->fft_input_complex_, true);
#endif
        }

        template <typename Dtype>
        void ConvolutionLayerFFT<Dtype>::transform_blob_to_real_array(int N, int K, int H, int W, const Dtype *blob_data, Dtype *padded_real_data)
        {
            int num_arr = N * K; // # of arrays (for weights it is num_weights [96 x 3]
                                 //              for input data it is channels [ 1 x 3]


            // set everything to 0 before --> so not set weights are 0-padded :)
            caffe_memset(this->fft_real_size_ * num_arr * sizeof(Dtype), 0., padded_real_data);

            #pragma omp parallel for collapse(4)
            for (int n = 0; n < N; n++)
            {
                for (int k = 0; k < K; k++)
                {
                    for (int h = 0; h < H; h++)
                    {
                        for (int w = 0; w < W; w++)
                        {
                            // ((n * ch_gr + c) * fft_height_ + h)* 2 * (fft_width_ / 2 + 1) + w
                            int offset_weight_real = (n * K + k) * this->fft_real_size_ ;
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
            if (this->weights_converted_) {
                // if weights were converted alrdy don't do that again :)
                return;
            }

//            // ===============================
//
//            // set all weights to 1.0 for testing purposes
//            int weight_size = this->height_ * this->width_;
//            // Allocations & plan for weights
//            int num_weights = this->num_output_ * (this->channels_ / this->group_);
//            int weight_alloc_size_in = weight_size * num_weights;
//            Dtype* cpu_data = reinterpret_cast<Dtype *>(malloc(weight_alloc_size_in * sizeof(Dtype)));
//            for (int j = 0; j < weight_alloc_size_in; ++j)
//            {
//                cpu_data[j] = 1.0;
//            }
//
//
//            // ===============================

            // the data location of the weights before the conversion...
            const Dtype *cpu_data = this->blobs_[0]->cpu_data();

            // transform data to real
            int N = this->num_output_;
            int K = (this->channels_ / this->group_);
            int H = this->kernel_h_;
            int W = this->kernel_w_;
            this->transform_blob_to_real_array(N, K, H, W, cpu_data, this->fft_weights_in_real_);

            // set complex mem to 0
            caffe_memset(this->fft_complex_size_ * N * K * sizeof(std::complex<Dtype>), 0.,
                         this->fft_weights_out_complex_);

            clock_t begin_clock = std::clock();
            fft_execute_plan<Dtype>(this->fft_weight_plan_);
            clock_t end_clock = std::clock();

            fft_cpu_free<Dtype>(fft_weights_in_real_);

            LOG(ERROR) << "fft for one layer took " << 1000.0 * (end_clock - begin_clock) / CLOCKS_PER_SEC << " ms.";
            this->weights_converted_ = true;

#ifdef WRITE_ARRAYS_T0_DISK
            this->write_arr_to_disk("weights_real_in.txt", N * K, this->fft_weights_in_real_);
            this->write_arr_to_disk("weights_complex_out.txt", N * K, this->fft_weights_out_complex_, true);
#endif
        }

        template <typename Dtype>
        void ConvolutionLayerFFT<Dtype>::convolve_fft()
        {
            // fft_input_complex_:           1x3x256x256
            // fft_weights_out_complex_:    96x3x256x256

            // weights and inputs are stored in the memory like this (each k is of size 256x256 [in 1st layer only])
            // n=0   |n=1   |n=2   |n=3   |n=4
            // k0k1k2|k0k1k2|k0k1k2|k0k1k2|k0k1k2

            // loop through channels.
            int N = this->num_output_ / this->group_;
            int K = this->channels_; //(this->channels_ / this->group_);
            int H = this->fft_height_;
            int W = (this->fft_width_ / 2) + 1;

            int k, n, h, w;
           //  #pragma omp parallel for collapse(4) private(k, n, h, w)
            for (n = 0; n < N; ++n)
            {
                for (k = 0; k < K; ++k)
                {
                    for (h = 0; h < H; h++)
                    {
                        for (w = 0; w < W; w++)
                        {
                            const int ch_gr = this->channels_ / this->group_;
                            // the channel group. e.g. g = [0;...;group_-1]
                            int g = k / ch_gr;
                            // the idx within the channel. e.g. k_idx = [0;...;ch_gr-1]
                            int k_idx = k % ch_gr;

                            // The actual idx of n. Because N = n_o_ / gr_ (see above) the idx of n is actual n + ...
                            int n_idx = n + g * N;

                            // Indexing: ((n * K + k) * H + h) * W + w
                            std::complex<Dtype> input = fft_input_complex_[((0 * K + k) * H + h) * W + w];
                            std::complex<Dtype> weight = fft_weights_out_complex_[((n_idx * ch_gr + k_idx) * H + h) * W + w];

                            // formula for complex mult from here: https://en.wikipedia.org/wiki/Complex_number#Multiplication_and_division
                            // (a+bi) (c+di) = (ac-bd) + (bc+ad)i.
                            Dtype a = std::real(weight);
                            Dtype b = std::imag(weight);
                            Dtype c = std::real(input);
                            Dtype d = std::imag(input);

                            std::complex<Dtype> res(a * c - b * d, b * c + a * d);
                            this->fft_conv_result_complex_[((n_idx * ch_gr + k_idx) * H + h) * W + w] = res;
                        }
                    }
                }
            }
#ifdef WRITE_ARRAYS_T0_DISK
            this->write_arr_to_disk("conv_res_complex.txt", N * K, this->fft_conv_result_complex_, true);
#endif
        }

        template <typename Dtype>
        void ConvolutionLayerFFT<Dtype>::normalize_ifft_result(const vector<Blob<Dtype>*>& top)
        {
            Dtype *top_data = top[0]->mutable_cpu_data();
            Dtype ifft_normalize_factor = 1. / (this->fft_width_ * this->fft_height_);

            // TODO: do the following right... :)
            int N = this->num_output_;
            int K = (this->channels_ / this->group_);
            for (int k = 0; k < K; ++k)
            {
                for (int n = 0; n < N; ++n)
                {
                    int offset_res_real = (n * K + k) * this->fft_real_size_ ;

                    for (int h = 0; h < this->height_out_; ++h) // =55 in 1st layer
                    {
                        // caffe does a valid convolution. fft is a full convolution. so the first 'valid' result is at
                        // idx (kernel_h_ - 1). The stride times the idx of the output pixel will be added onto this.
                        int h_idx = (this->kernel_h_ - 1) + h * this->stride_h_;
                        for (int w = 0; w < this->width_out_; ++w) // =55 in 1st layer
                        {
                            // caffe does a valid convolution. fft is a full convolution. so the first 'valid' result is at
                            // idx (kernel_w_ - 1). The stride times the idx of the output pixel will be added onto this.
                            int w_idx = (this->kernel_w_ - 1) + w * this->stride_w_;
                            //((n * K + k) * H + h) * W + w;
                            int top_data_idx = (n * this->height_out_ + h) * this->width_out_ + w;

                            // the index in the data of the convolution result array (the real one)
                            int res_data_idx = offset_res_real + h_idx * this->fft_width_ + w_idx;

                            // normalize fft and sum up everything from the input channels...
                            top_data[top_data_idx] += this->fft_conv_result_real_[res_data_idx] * ifft_normalize_factor;
                        }

                    }

                }
            }
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
                        std::complex<Dtype> *arr_conv = reinterpret_cast<std::complex<Dtype> *>(arr);
                        fout << arr_conv[i].real() << ";" << arr_conv[i].imag() << "\n";
                    }
                    else
                    {
                        Dtype *arr_conv = reinterpret_cast<Dtype *>(arr);
                        fout << *(arr_conv + i) << "\n";
                    }
                }
                std::cout << "Array data successfully saved into the file " << output_name << std::endl;
            }

            fout.close();
        }

        template <typename Dtype>
        void ConvolutionLayerFFT<Dtype>::write_simple_arr_to_disk(const char* output_name, int size, Dtype *arr)
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

//#ifdef CPU_ONLY
//	STUB_GPU(ConvolutionLayerFFT);
//#endif

	INSTANTIATE_CLASS(ConvolutionLayerFFT);

}  // namespace caffe
