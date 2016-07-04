#include "caffe/layers/inference_layer.hpp"
#include "boost/math/tools/precision.hpp"
#include "caffe/util/benchmark.hpp"

namespace caffe {

template<typename Dtype>
void InferenceLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                       const vector<Blob<Dtype>*>& top) {
  // Check blob dimensions...
  check_bottom_top(bottom, top);

  num_ = bottom[0]->num();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  map_size_ = height_ * width_;

  vector<int> map_shape;
  map_shape.push_back(1);
  map_shape.push_back(1);
  map_shape.push_back(height_);
  map_shape.push_back(width_);

  // Alloc temporary maps
  tmp_map_ = shared_ptr<Blob<Dtype> >(new Blob<Dtype>(map_shape));
  tmp2_map_ = shared_ptr<Blob<Dtype> >(new Blob<Dtype>(map_shape));
  max_map_ = shared_ptr<Blob<Dtype> >(new Blob<Dtype>(map_shape));

  num_appearance_maps_ = bottom[0]->channels();
  num_idpr_maps_ = bottom[1]->channels();

  InferenceParameter inference_param = this->layer_param_.inference_param();
  num_connections_ = inference_param.num_connections();

  CHECK_GT(num_connections_, 0) << "Number of connections cannot be zero.";

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

    // 3rd weights: inference order weights = num_connections_ * (3 + 6 * num_idpr_maps_)
    //              [num_idpr_maps_ is the max possible number of idpr maps associated with a joint, since there are only that much idpr maps as input to the layer]
    //              x = child joint, y = parent joint, n = num_clusters (= actual num idpr maps for each joint in the connection),
    //              o_x0 = offset of child idpr map zero (there are n idpr maps per joint),
    //              o_y0 = offset of parent idpr map zero (there are n idpr maps per joint) [an offset is a pair of floats (x-coord, y-coord)]
    //              for every connection: (x, y, n, <cluster indices of joint x>, <cluster indices of joint y>, <o_x0, o_x1, ... o_x(n-1)>, <o_y0, o_y1, ... o_y(n-1)>, 0 ... 0 )
    // There are many zeros at the end, because theoretically a higher number of idpr maps per joint is possible.
    vector<int> inference_weights_shape;
    inference_weights_shape.push_back(num_connections_);
    inference_weights_shape.push_back(3 + 6 * num_idpr_maps_);
    this->blobs_[2].reset(new Blob<Dtype>(inference_weights_shape));
  }
}

template<typename Dtype>
void InferenceLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                    const vector<Blob<Dtype>*>& top) {
  // Check blob dimensions...
  check_bottom_top(bottom, top);

  num_ = bottom[0]->num();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  map_size_ = height_ * width_;

  vector<int> map_shape;
  map_shape.push_back(1);
  map_shape.push_back(1);
  map_shape.push_back(height_);
  map_shape.push_back(width_);

  tmp_map_->Reshape(map_shape);
  tmp2_map_->Reshape(map_shape);
  max_map_->Reshape(map_shape);

  num_appearance_maps_ = bottom[0]->channels();
  num_idpr_maps_ = bottom[1]->channels();

  this->compute_output_shape();

  top[0]->Reshape(num_, num_output_, height_out_, width_out_);

  vector<int> offset_shape;
  offset_shape.push_back(num_);
  offset_shape.push_back(num_output_);
  offset_shape.push_back(3);
  top[1]->Reshape(offset_shape); // for offset {x, y} and idpr map idx

  std::vector<int> backtrack_shape; // actual result
  backtrack_shape.push_back(num_);
  backtrack_shape.push_back(num_output_);
  backtrack_shape.push_back(2); // for x and y
  top[2]->Reshape(backtrack_shape);
}

template<typename Dtype>
void InferenceLayer<Dtype>::check_bottom_top(const vector<Blob<Dtype>*>& bottom,
                                             const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom.size(), 3) << "InferenceLayer needs to have three input blobs: Appearance Maps, DT of IDPR Maps and Max Origins of IDPR Maps!";
  CHECK_EQ(top.size(), 3) << "InferenceLayer needs to have three output blobs: Inference Result (max-added maps), Offsets of each joint and final results (coordinates for each joint)!";

  CHECK_EQ(4, bottom[0]->num_axes())<< "First input must have 4 axes, "
  << "corresponding to (num, channels, height, width)";
  CHECK_EQ(4, bottom[1]->num_axes())<< "Second input must have 4 axes, "
  << "corresponding to (num, channels, height, width)";
  CHECK_EQ(5, bottom[2]->num_axes())<< "Third input must have 5 axes, "
  << "corresponding to (num, channels, height, width, 2)";

  CHECK_EQ(bottom[0]->shape(0), bottom[1]->shape(0))<< "The three bottom blobs need to have the same batch size (dimension 0)";
  CHECK_EQ(bottom[0]->shape(2), bottom[1]->shape(2)) << "The three bottom blobs need to have the same height (dimension 2)";
  CHECK_EQ(bottom[0]->shape(3), bottom[1]->shape(3)) << "The three bottom blobs need to have the same width (dimension 3)";
  CHECK_EQ(bottom[0]->shape(0), bottom[2]->shape(0))<< "The three bottom blobs need to have the same batch size (dimension 0)";
  CHECK_EQ(bottom[0]->shape(2), bottom[2]->shape(2)) << "The three bottom blobs need to have the same height (dimension 2)";
  CHECK_EQ(bottom[0]->shape(3), bottom[2]->shape(3)) << "The three bottom blobs need to have the same width (dimension 3)";
  CHECK_EQ(bottom[2]->shape(4), 2) << "The third blob's 5th dimension has to be 2 (one dim for x values and one for y values)";

  // LOG(ERROR) << "bottom[0] shape: " << bottom[0]->shape(0) << " " << bottom[0]->shape(1) << " " << bottom[0]->shape(2) << " " << bottom[0]->shape(3);
  // LOG(ERROR) << "bottom[1] shape: " << bottom[1]->shape(0) << " " << bottom[1]->shape(1) << " " << bottom[1]->shape(2) << " " << bottom[1]->shape(3);
  // LOG(ERROR) << "bottom[2] shape: " << bottom[2]->shape(0) << " " << bottom[2]->shape(1) << " " << bottom[2]->shape(2) << " " << bottom[2]->shape(3) << " " << bottom[2]->shape(4);
}

template<typename Dtype>
void InferenceLayer<Dtype>::compute_output_shape() {
  this->num_output_ = this->num_appearance_maps_;
  this->height_out_ = this->height_;
  this->width_out_ = this->width_;
}

template<typename Dtype>
void InferenceLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                        const vector<Blob<Dtype>*>& top) {

  Timer timer, total_timer;
  total_timer.Start();
  timer.Start();


  const Dtype* appearance_data = bottom[0]->cpu_data();
  const Dtype *appearance_weights = this->blobs_[0]->cpu_data();
  Dtype *top_data = top[0]->mutable_cpu_data();
  //LOG(ERROR) << "top count: " << top[0]->count();
  caffe_set(top[0]->count(), static_cast<Dtype>(0.), top_data);

//  // Scale appearance weights once with weights. TODO: Do this before? But after upscale?
//  // TODO: can parallelize here...
  for (int i = 0; i < num_appearance_maps_; ++i) {
    const Dtype *app_joint = appearance_data + i * map_size_;
    Dtype *top_joint = top_data + i * map_size_;
    caffe_axpy(map_size_, appearance_weights[i], app_joint, top_joint);
  }

  LOG(ERROR) << "13 app_map * wa took: " << timer.MilliSeconds() << " ms.";

  const Dtype* idpr_data = bottom[1]->cpu_data();

  const Dtype *idpr_weights = this->blobs_[1]->cpu_data();
  const Dtype *inference_weights = this->blobs_[2]->cpu_data();

  std::vector<bool> app_map_added(num_output_, false);

  for (int i = 0; i < num_connections_; ++i) {
    timer.Start();

    // first connection is 3 - 2 (child - parent)
    const Dtype *connection_weights = inference_weights + i * this->blobs_[2]->shape(1);
    int child_joint = static_cast<int>(connection_weights[0]);
    int parent_joint = static_cast<int>(connection_weights[1]); // = current joint in idpr code
    int num_clusters = static_cast<int>(connection_weights[2]);

    //LOG(ERROR) << "-------------------------------------------------";
    //LOG(ERROR) << child_joint << "-" << parent_joint << ": " << num_clusters;

    // the child appearance maps get saved into the top blob.
    // if we are at a leaf the childappearance maps equals to the appearance map
    Dtype *child_app_map = top_data + child_joint * map_size_;

    Dtype max_offset_x = 0.;
    Dtype max_offset_y = 0.;
    Dtype max_val = boost::math::tools::min_value<Dtype>();
    int max_child_idpr_map_idx = 0;

    for (int pp = 0; pp < num_clusters; ++pp) {
      int parent_id = get_parent_idpr_map_idx_cpu(pp, num_clusters, connection_weights);
      const Dtype *parent_idpr_map = idpr_data + parent_id * map_size_;
      for (int cc = 0; cc < num_clusters; ++cc) {

        Timer map_shift;
        map_shift.Start();

        // set tmp map to zero...
        caffe_set(map_size_, static_cast<Dtype>(0.), tmp_map_->mutable_cpu_data());
        caffe_set(map_size_, static_cast<Dtype>(0.), tmp2_map_->mutable_cpu_data());

        int child_id = get_child_idpr_map_idx_cpu(cc, connection_weights);

        //LOG(ERROR) << "parent id - child id :" << parent_id << " - " << child_id;

        // multiply input idpr map with corresponding idpr weight
        const Dtype wi = idpr_weights[child_id];
        const Dtype *idpr_map = idpr_data + child_id * map_size_;

        Dtype *tmp_map = tmp_map_->mutable_cpu_data();
        caffe_axpy(map_size_, wi, idpr_map, tmp_map);

        //LOG(ERROR) << "idprMap (" << child_id << ") @: " << *(tmp_map + (173 * 720 + 300));

        // subtree = subtree = childAppearanceMap + idprMap
        caffe_axpy(map_size_, static_cast<Dtype>(1.), child_app_map, tmp_map);

        //LOG(ERROR) << "subtree (" << child_id << ") @: " << *(tmp_map + (173 * 720 + 300));
        // translate tmp_map / subtree by offset, how?
        const Dtype* offset = get_child_ipdr_map_offset(cc, num_clusters, connection_weights);
        add_maps_with_offset(tmp_map, parent_idpr_map, tmp2_map_->mutable_cpu_data(), offset[0], offset[1]);

        LOG(ERROR) << "offset_added (" << child_id << ") @: " << *(tmp2_map_->cpu_data() + (173 * 720 + 300)) << "offset: " << offset[0] << ", " << offset[1];

        Timer max_timer;
        max_timer.Start();
        Dtype maxim = map_max_cpu(tmp2_map_->cpu_data());
        max_timer.Stop();

        if (maxim > max_val) {
          max_val = maxim;
          max_offset_x = offset[0];
          max_offset_y = offset[1];
          max_child_idpr_map_idx = child_id; // TODO: save real def x,y here?
          caffe_copy(map_size_, tmp2_map_->cpu_data(), max_map_->mutable_cpu_data());
        }

        // LOG(ERROR) << "Mapshift shift[child_app_map] + idpr*wi took: " << map_shift.MilliSeconds() << " max took: " << max_timer.MilliSeconds();
      }
    }


    // LOG(ERROR) << child_joint << "-" << parent_joint << " with " << num_clusters << " clusters took " << timer.MilliSeconds() << " ms.";

    //LOG(ERROR) << "maxmap (" << parent_joint << ") @: " << *(max_map_->cpu_data() + (173 * 720 + 300));

    Dtype *parent_app_map = top_data + parent_joint * map_size_;

    // Now add the top map on top of it...
    caffe_axpy(map_size_, static_cast<Dtype>(1.), max_map_->cpu_data(), parent_app_map);
    Dtype *offsets = top[1]->mutable_cpu_data() + child_joint * 3;
    offsets[0] = max_offset_x;
    offsets[1] = max_offset_y;
    offsets[2] = max_child_idpr_map_idx;
    //LOG(ERROR) << "Offsets and map idx for parent joint " << child_joint << ": " << max_offset_x << ", " << max_offset_y << ", " << max_child_idpr_map_idx;
  }

  // --------------------------------------- BACKTRACKING -------------------------------------------------


  Dtype *backtrack_data = top[2]->mutable_cpu_data();

  Dtype *result_map = top_data;
  int max_x = 0;
  int max_y = 0;

  map_max_cpu(result_map, max_x, max_y);
  backtrack_data[0] = max_x;
  backtrack_data[1] = max_y;

  for (int i = num_connections_ - 1; i >= 0; --i) {
      // first connection is 3 - 2 (child - parent)
      const Dtype *connection_weights = inference_weights + i * this->blobs_[2]->shape(1);
      int child_joint = static_cast<int>(connection_weights[0]);
      int parent_joint = static_cast<int>(connection_weights[1]);

      const Dtype *offsets = top[1]->cpu_data() + child_joint * 3;

      Dtype tmp_x = *(backtrack_data + parent_joint * 2 + 0) - offsets[0];
      Dtype tmp_y = *(backtrack_data + parent_joint * 2 + 1) - offsets[1];
      Dtype tmp_c = offsets[2];

      // TODO: now really backtrack... thereto use second blob from dt layer.
      int x_in_map = static_cast<int>(tmp_x);
      int y_in_map = static_cast<int>(tmp_y);
      int c_in_map = static_cast<int>(tmp_c);

      // LOG(ERROR) << "x, y, maxmap: " << x_in_map << ", " << y_in_map << ", " << c_in_map;

      const Dtype *max_coord = bottom[2]->cpu_data() + (((0 * num_idpr_maps_ + c_in_map) * height_+ y_in_map) * width_ + x_in_map) * 2;

      Dtype *child_location = backtrack_data + child_joint * 2;
      child_location[0] = max_coord[0];
      child_location[1] = max_coord[1];
  }

  LOG(ERROR) << "Total fwtime: " << total_timer.MilliSeconds() << " ms.";
//  for (int i = 0; i < 13; i++) {
//    Dtype *child_location = backtrack_data + i * 2;
//    LOG(ERROR) << "(" << i << ") " << child_location[0] << ", " << child_location[1];
//  }



//  // search for maximum
//  double mi, ma; cv::Point locma, locmi;
//  cv::minMaxLoc(result, &mi, &ma, &locmi, &locma);
//
//  instance.jointLocations[0].x = locma.x;
//  instance.jointLocations[0].y = locma.y;
//
//  // get the joint locations
//  for (int edge = 0; edge < model.swimmer.topDown.size(); ++edge){
//    int parent = model.swimmer.topDown[edge].x;
//    int child = model.swimmer.topDown[edge].y;
//
//    instance.jointLocations[child].x = instance.jointLocations[parent].x - instance.offsets[child].x;
//    instance.jointLocations[child].y = instance.jointLocations[parent].y - instance.offsets[child].y;
//
//    // we have to consider the distance transform too
//    Point2i reallocation = backtrackIndex(instance.deformationidx[child], instance.jointLocations[child].x, instance.jointLocations[child].y);
//
//    instance.jointLocations[child].x = reallocation.x;
//    instance.jointLocations[child].y = reallocation.y;


//  for (int i = 0; i < 13; ++i) {
//    const Dtype *offsets = top[1]->cpu_data() + i * 3;
//    LOG(ERROR) << "ME: " << offsets[0] << ", " << offsets[1] << "\n";
//  }
}

template<typename Dtype>
void InferenceLayer<Dtype>::add_maps_with_offset(const Dtype *src_1, const Dtype *src_2, Dtype *dst,
                                                 Dtype offset_x, Dtype offset_y) {
  for (int x = 0; x < width_; ++x) {
    for (int y = 0; y < height_; ++y) {
      //LOG(ERROR) << "round: " << x < " to " << offset_x << " and " << x < " to " << offset_x;
      Dtype src_x = x - offset_x;
      Dtype src_y = y - offset_y;
      Dtype *dst_px = dst + y * width_ + x;

      if (src_x < 0 || src_y < 0 || src_x >= width_ || src_y >= height_) {
        *dst_px = 0. + src_2[y * width_ + x];
      } else {
        Dtype out = 0.;
        // interpolate billinear pixel from src_1
        const int x1 = static_cast<int>(std::floor(src_x));
        const int y1 = static_cast<int>(std::floor(src_y));
        const int x2 = x1 + 1;
        const int y2 = y1 + 1;
        const int x2_read = std::min(x2, width_ - 1);
        const int y2_read = std::min(y2, height_ - 1);
        Dtype src_reg = src_1[y1 * width_ + x1];  //   src(y1, x1);
        out = out + src_reg * ((x2 - src_x) * (y2 - src_y));

        src_reg = src_1[y1 * width_ + x2_read];  //   src(y1, x2_read);
        out = out + src_reg * ((src_x - x1) * (y2 - src_y));

        src_reg = src_1[y2_read * width_ + x1];  //   src(y2_read, x1);
        out = out + src_reg * ((x2 - src_x) * (src_y - y1));

        src_reg = src_1[y2_read * width_ + x2_read];  //   src(y2_read, x2_read);
        out = out + src_reg * ((src_x - x1) * (src_y - y1));


        *dst_px = out + src_2[y * width_ + x];
      }
    }
  }
}

template<typename Dtype>
Dtype InferenceLayer<Dtype>::map_max_cpu(const Dtype *map) {
  Dtype max_val = boost::math::tools::min_value<Dtype>();

  for (int x = 0; x < width_; ++x) {
    for (int y = 0; y < height_; ++y) {
      Dtype val = map[y * width_ + x];
      max_val = std::max(val, max_val);
    }
  }

  return max_val;
}

template<typename Dtype>
Dtype InferenceLayer<Dtype>::map_max_cpu(const Dtype *map, int &max_x, int &max_y) {
  Dtype max_val = boost::math::tools::min_value<Dtype>();

  for (int x = 0; x < width_; ++x) {
    for (int y = 0; y < height_; ++y) {
      Dtype val = map[y * width_ + x];
      if (val > max_val) {
        max_val = val;
        max_x = x;
        max_y = y;
      }
    }
  }

  return max_val;
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
