#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "caffe/util/upgrade_proto.hpp"
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <math.h>

// resize factor of image. Might be replaced by 1.0 iff the training of smaller
// parts works better in the future
#define IM_RESIZE       2.5
// resize factor and shift of DCNN result
#define RES_RESIZE        12.8
#define RES_SHIFT       34

using namespace caffe;  // NOLINT(build/namespaces)
using std::string;

/* Pair (label, confidence) representing a prediction. */
typedef std::pair<string, float> Prediction;

class ClassifierSimple {
 public:
  ClassifierSimple(const string& model_file,
             const string& trained_file);

  cv::Mat *Forward(const cv::Mat& img);

  void SaveNetwork(const string& out_file);
  shared_ptr<Net<float> > net_;

 private:
  void SetMean(const string& mean_file);

  Blob<float> *Predict(const cv::Mat& img);

  void WrapInputLayer(std::vector<cv::Mat>* input_channels);

  void Preprocess(const cv::Mat& img,
                  std::vector<cv::Mat>* input_channels);

 private:
  cv::Size input_geometry_;
  int num_channels_;
  cv::Mat mean_;
  std::vector<string> labels_;
  NetParameter param_;
};

ClassifierSimple::ClassifierSimple(const string& model_file,
                       const string& trained_file) {
#ifdef CPU_ONLY
  LOG(ERROR) << "CPU";
  Caffe::set_mode(Caffe::CPU);
#else
  Caffe::set_mode(Caffe::GPU);
#endif

  /* Load the network. */

  ReadNetParamsFromTextFileOrDie(model_file, &param_);
  param_.mutable_state()->set_phase(TEST);

  //net_.reset(new Net<float>(model_file, TEST));
  net_.reset(new Net<float>(param_));
  net_->CopyTrainedLayersFrom(trained_file);

  CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
  // CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

  Blob<float>* input_layer = net_->input_blobs()[0];
  num_channels_ = input_layer->channels();
  CHECK(num_channels_ == 3 || num_channels_ == 1)
    << "Input layer should have 1 or 3 channels.";

  Blob<float>* output_layer = net_->output_blobs()[0];
}


void ClassifierSimple::SaveNetwork(const string& out_file) {
   // For intermediate results, we will also dump the gradient values.
   net_->ToProto(&param_, false);
   string filename(out_file);
   LOG(INFO) << "Saving to " << out_file;
   WriteProtoToBinaryFile(param_, out_file.c_str());
}

static bool PairCompare(const std::pair<float, int>& lhs,
                        const std::pair<float, int>& rhs) {
  return lhs.first > rhs.first;
}

/* Return the indices of the top N values of vector v. */
static std::vector<int> Argmax(const std::vector<float>& v, int N) {
  std::vector<std::pair<float, int> > pairs;
  for (size_t i = 0; i < v.size(); ++i)
    pairs.push_back(std::make_pair(v[i], i));
  std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

  std::vector<int> result;
  for (int i = 0; i < N; ++i)
    result.push_back(pairs[i].second);
  return result;
}

/* Return the top N predictions. */
cv::Mat *ClassifierSimple::Forward(const cv::Mat& img) {
  Blob<float> *output = Predict(img);

  int num_outputs = output->shape(1);

  cv::Mat *res = new cv::Mat[num_outputs];



  for (int n = 0; n < num_outputs; ++n) {
    cv::Mat tmp = cv::Mat(output->shape(2), output->shape(3), CV_32FC1);
    const void *src = reinterpret_cast<const void *>(output->cpu_data() + output->offset(0, n, 0, 0));
    memcpy(tmp.ptr(0), src, output->shape(2) * output->shape(3) * sizeof(float));
    res[n] = tmp;
  }

  return res;
}

/* Load the mean file in binaryproto format. */
void ClassifierSimple::SetMean(const string& mean_file) {
  BlobProto blob_proto;
  ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

  /* Convert from BlobProto to Blob<float> */
  Blob<float> mean_blob;
  mean_blob.FromProto(blob_proto);
  CHECK_EQ(mean_blob.channels(), num_channels_)
    << "Number of channels of mean file doesn't match input layer.";

  /* The format of the mean file is planar 32-bit float BGR or grayscale. */
  std::vector<cv::Mat> channels;
  float* data = mean_blob.mutable_cpu_data();
  for (int i = 0; i < num_channels_; ++i) {
    /* Extract an individual channel. */
    cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
    channels.push_back(channel);
    data += mean_blob.height() * mean_blob.width();
  }

  /* Merge the separate channels into a single image. */
  cv::Mat mean;
  cv::merge(channels, mean);

  /* Compute the global mean pixel value and create a mean image
   * filled with this value. */
  cv::Scalar channel_mean = cv::mean(mean);
  mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
}

Blob<float> *ClassifierSimple::Predict(const cv::Mat& img) {
  Blob<float>* input_layer = net_->input_blobs()[0];
  // Reshape net to image size * scale
  input_layer->Reshape(1, num_channels_,
                       img.rows * IM_RESIZE, img.cols * IM_RESIZE);

  input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
  /* Forward dimension change to all layers. */
  net_->Reshape();

  std::vector<cv::Mat> input_channels;
  WrapInputLayer(&input_channels);

  Preprocess(img, &input_channels);

  net_->ForwardPrefilled();

  /* Copy the output layer to a std::vector */
  Blob<float>* output_layer = net_->output_blobs()[1];
//  const float* begin = output_layer->cpu_data();
//  const float* end = begin + output_layer->channels();
//  return std::vector<float>(begin, end);
  return output_layer;
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void ClassifierSimple::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
  Blob<float>* input_layer = net_->input_blobs()[0];

  int width = input_layer->width();
  int height = input_layer->height();
  float* input_data = input_layer->mutable_cpu_data();
  for (int i = 0; i < input_layer->channels(); ++i) {
    cv::Mat channel(height, width, CV_32FC1, input_data);
    input_channels->push_back(channel);
    input_data += width * height;
  }
}
//// convert image to float
//cv::Mat img_float;
//img_in.convertTo( img_float, CV_32FC3 );
//
//// split channels
//img_channels.clear();
//cv::split( img_float, img_channels );
//
//// subtract channelwise mean values
//for ( size_t channel_id = 0; channel_id < 3; ++channel_id )
//{
//  img_channels[channel_id] -= pixel_means[channel_id];
//  img_channels[channel_id] /= pixel_stdev[channel_id];
//}


void ClassifierSimple::Preprocess(const cv::Mat& img,
                            std::vector<cv::Mat>* input_channels) {
  /* Convert the input image to the input image format of the network. */
  cv::Mat sample;
  if (img.channels() == 3 && num_channels_ == 1)
    cv::cvtColor(img, sample, CV_BGR2GRAY);
  else if (img.channels() == 4 && num_channels_ == 1)
    cv::cvtColor(img, sample, CV_BGRA2GRAY);
  else if (img.channels() == 4 && num_channels_ == 3)
    cv::cvtColor(img, sample, CV_BGRA2BGR);
  else if (img.channels() == 1 && num_channels_ == 3)
    cv::cvtColor(img, sample, CV_GRAY2BGR);
  else
    sample = img;

  cv::Mat sample_resized;

  if (sample.size() != input_geometry_)
    cv::resize(sample, sample_resized, cv::Size(0,0), IM_RESIZE, IM_RESIZE);
  else
    sample_resized = sample;

  cv::Mat sample_float;
  if (num_channels_ == 3)
    sample_resized.convertTo(sample_float, CV_32FC3);
  else
    sample_resized.convertTo(sample_float, CV_32FC1);

  cv::Mat sample_normalized;

  // means
  std::vector<float> pixel_means;  //  = { 103.9f, 116.8f, 123.68f };  // pixels means in BGR order
  pixel_means.push_back(103.9f);
  pixel_means.push_back(116.8f);
  pixel_means.push_back(123.68f);
  std::vector<float> pixel_stdev; // = { 1.0f, 1.0f, 1.0f };
  pixel_stdev.push_back(1.0f);
  pixel_stdev.push_back(1.0f);
  pixel_stdev.push_back(1.0f);

  /* This operation will write the separate BGR planes directly to the
   * input layer of the network because it is wrapped by the cv::Mat
   * objects in input_channels. */
  cv::split(sample_float, *input_channels);

  // subtract channelwise mean values
  for ( size_t channel_id = 0; channel_id < num_channels_; ++channel_id )
  {
    (*input_channels)[channel_id] -= pixel_means[channel_id];
    (*input_channels)[channel_id] /= pixel_stdev[channel_id];
  }

  CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
        == net_->input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";
}

cv::Mat heatmap(cv::Mat in) {
  cv::Mat tmp, result;
  double min, max;
  cv::minMaxIdx(in, &min, &max);

  in.convertTo(tmp, CV_8UC1, 255.0 / (max - min), -min);
  cv::applyColorMap(tmp, result, cv::COLORMAP_JET);
  return result;
}

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cerr << "Usage: " << argv[0]
              << " deploy.prototxt network.caffemodel"
              << " img.jpg" << std::endl;
    return 1;
  }

  ::google::InitGoogleLogging(argv[0]);

  string model_file   = argv[1];
  string trained_file = argv[2];
  ClassifierSimple classifier(model_file, trained_file);

  string file = argv[3];

  std::cout << "---------- Prediction for "
            << file << " ----------" << std::endl;

  cv::Mat img = cv::imread(file, -1);
  CHECK(!img.empty()) << "Unable to decode image " << file;
  cv::Mat *res = classifier.Forward(img);



  cv::imshow("a", img);
  cv::waitKey(0);


  cv::Mat resized;
  cv::Mat result;
  cv::Mat roi;
  // get added idpr stuff
  shared_ptr<Blob<float> > output = classifier.net_->blob_by_name("idprclasses");
  int num_outputs = output->shape(1);
  cv::Mat *residpr = new cv::Mat[num_outputs];
  for (int n = 0; n < num_outputs; ++n) {
    cv::Mat tmp = cv::Mat(output->shape(2), output->shape(3), CV_32FC1);
    const void *src = reinterpret_cast<const void *>(output->cpu_data() + output->offset(0, n, 0, 0));
    memcpy(tmp.ptr(0), src, output->shape(2) * output->shape(3) * sizeof(float));
    residpr[n] = tmp;
  }

  result = residpr[42];
  resized = cv::Mat::zeros(576, 720, CV_32FC1);
  roi = resized( cv::Rect( RES_SHIFT, RES_SHIFT, round(result.cols * RES_RESIZE), round(result.rows * RES_RESIZE) ) );
  cv::resize( result, roi, cv::Size(roi.cols, roi.rows) );
  cv::imshow("before-dt", heatmap(resized));
  cv::waitKey(0);


  if (classifier.net_->blob_by_name("upscale2") != 0) {
    // get added scaled stuff
    output = classifier.net_->blob_by_name("upscale2");
    num_outputs = output->shape(1);
    cv::Mat *upscale = new cv::Mat[num_outputs];
    for (int n = 0; n < num_outputs; ++n) {
      cv::Mat tmp = cv::Mat(output->shape(2), output->shape(3), CV_32FC1);
      const void *src = reinterpret_cast<const void *>(output->cpu_data() + output->offset(0, n, 0, 0));
      memcpy(tmp.ptr(0), src, output->shape(2) * output->shape(3) * sizeof(float));
      upscale[n] = tmp;
    }

    result = upscale[42];
    cv::imshow("upscale", heatmap(result));
    cv::waitKey(0);
  }



  // show dt stuff
  result = res[42];

  cv::imshow("out", heatmap(result));
  cv::waitKey(0);
  cv::Mat tmp;
  result.convertTo(tmp, CV_8UC3, 255.0);
  cv::imwrite("out_fast.png", tmp);

  // resize if small image:

  cv::Size res_size = result.size();

  if (res_size.width < 720) {
      resized = cv::Mat::zeros(576, 720, CV_32FC1);
      roi = resized( cv::Rect( RES_SHIFT, RES_SHIFT, round(result.cols * RES_RESIZE), round(result.rows * RES_RESIZE) ) );
      cv::resize( result, roi, cv::Size(roi.cols, roi.rows) );
      cv::imshow("resized", heatmap(resized));
      cv::waitKey(0);
  }


  // save network
  // classifier.SaveNetwork("models/swimmers_fullconv/out_new.binaryproto");


//
//
//  /* Print the top N predictions. */
//  for (size_t i = 0; i < predictions.size(); ++i) {
//    Prediction p = predictions[i];
//    std::cout << std::fixed << std::setprecision(4) << p.second << " - \""
//              << p.first << "\"" << std::endl;
//  }
}
