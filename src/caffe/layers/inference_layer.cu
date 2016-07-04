#include "caffe/layers/inference_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/inference_layer_util.hpp"

namespace caffe {

template <typename Dtype>
void InferenceLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                          const vector<Blob<Dtype>*>& top) {
  // LOG(ERROR) << "We are in GPU forward mode! Hurray!";

//  Timer timer, total_timer;
//    total_timer.Start();
//    timer.Start();


    const Dtype* appearance_data = bottom[0]->gpu_data();
    const Dtype *appearance_weights = this->blobs_[0]->cpu_data(); // use cpu data, because alpha in gpu_saxpy expects host memory
    Dtype *top_data = top[0]->mutable_gpu_data();
    //LOG(ERROR) << "top count: " << top[0]->count();
    caffe_gpu_set(top[0]->count(), static_cast<Dtype>(0.), top_data);

    // TODO: can parallelize here...
    for (int i = 0; i < num_appearance_maps_; ++i) {
      const Dtype *app_joint = appearance_data + i * map_size_;
      Dtype *top_joint = top_data + i * map_size_;
      caffe_gpu_axpy(map_size_, appearance_weights[i], app_joint, top_joint);
    }

    // LOG(ERROR) << "13 app_map * wa took: " << timer.MilliSeconds() << " ms.";

    const Dtype* idpr_data = bottom[1]->gpu_data();
    const Dtype *idpr_weights = this->blobs_[1]->cpu_data();
    const Dtype *inference_weights = this->blobs_[2]->cpu_data(); // use cpu data, because alpha in gpu_saxpy expects host memory

    for (int i = 0; i < num_connections_; ++i) {
      // timer.Start();

      // first connection is 3 - 2 (child - parent)
      const Dtype *connection_weights = inference_weights + i * this->blobs_[2]->shape(1);
      int child_joint = static_cast<int>(connection_weights[0]);
      int parent_joint = static_cast<int>(connection_weights[1]); // = current joint in idpr code
      int num_clusters = static_cast<int>(connection_weights[2]);

      // the child appearance maps get saved into the top blob.
      // if we are at a leaf the childappearance maps equals to the appearance map
      Dtype *child_app_map = top_data + child_joint * map_size_;

      Dtype max_offset_x = 0.;
      Dtype max_offset_y = 0.;
      Dtype max_val = FLT_MIN;
      int max_child_idpr_map_idx = 0;
      for (int pp = 0; pp < num_clusters; ++pp) {
        int parent_id = get_parent_idpr_map_idx_cpu(pp, num_clusters, connection_weights);
        const Dtype *parent_idpr_map = idpr_data + parent_id * map_size_;
        for (int cc = 0; cc < num_clusters; ++cc) {

          Timer map_shift;
          map_shift.Start();

          // set tmp map to zero...
          caffe_gpu_set(map_size_, static_cast<Dtype>(0.), tmp_map_->mutable_gpu_data());
          caffe_gpu_set(map_size_, static_cast<Dtype>(0.), tmp2_map_->mutable_gpu_data());

          int child_id = get_child_idpr_map_idx_cpu(cc, connection_weights);

          //LOG(ERROR) << "parent id - child id :" << parent_id << " - " << child_id;

          // multiply input idpr map with corresponding idpr weight
          const Dtype wi = idpr_weights[child_id];
          const Dtype *idpr_map = idpr_data + child_id * map_size_;

          Dtype *tmp_map = tmp_map_->mutable_gpu_data();
          caffe_gpu_axpy(map_size_, wi, idpr_map, tmp_map);

          // subtree = subtree = childAppearanceMap + idprMap
          caffe_gpu_axpy(map_size_, static_cast<Dtype>(1.), child_app_map, tmp_map);

          //const Dtype *tmp_map_cpu = tmp_map_->cpu_data();
         // LOG(ERROR) << "subtree (" << child_id << ") @: " << *(tmp_map_cpu + (173 * 720 + 300));

          const Dtype* offset = get_child_ipdr_map_offset(cc, num_clusters, connection_weights);
          add_maps_with_offset_gpu(tmp_map, parent_idpr_map, tmp2_map_->mutable_gpu_data(), offset[0], offset[1], height_, width_);

          // LOG(ERROR) << "offset_added (" << child_id << ") @: " << *(tmp2_map_->cpu_data() + (173 * 720 + 300)) << "offset: " << offset[0] << ", " << offset[1];

          Timer max_timer;
          max_timer.Start();

          Dtype maxim = map_max_gpu(tmp2_map_->gpu_data(), map_size_);
          max_timer.Stop();

          if (maxim > max_val) {
            max_val = maxim;
            max_offset_x = offset[0];
            max_offset_y = offset[1];
            max_child_idpr_map_idx = child_id; // TODO: save real def x,y here?
            caffe_copy(map_size_, tmp2_map_->gpu_data(), max_map_->mutable_gpu_data());
          }

          // LOG(ERROR) << "Mapshift shift[child_app_map] + idpr*wi took: " << map_shift.MilliSeconds() << " max took: " << max_timer.MilliSeconds();
        }
      }
      Dtype *parent_app_map = top_data + parent_joint * map_size_;

      // Now add the top map on top of it...
      caffe_gpu_axpy(map_size_, static_cast<Dtype>(1.), max_map_->gpu_data(), parent_app_map);
      // TODO: GPU???
      Dtype *offsets = top[1]->mutable_cpu_data() + child_joint * 3;
      offsets[0] = max_offset_x;
      offsets[1] = max_offset_y;
      offsets[2] = max_child_idpr_map_idx;
    }


//

//
//
//        LOG(ERROR) << child_joint << "-" << parent_joint << " with " << num_clusters << " clusters took " << timer.MilliSeconds() << " ms.";
//
//        //LOG(ERROR) << "maxmap (" << parent_joint << ") @: " << *(max_map_->cpu_data() + (173 * 720 + 300));
//
//        Dtype *parent_app_map = top_data + parent_joint * map_size_;
//
//        // Now add the top map on top of it...
//        caffe_axpy(map_size_, static_cast<Dtype>(1.), max_map_->cpu_data(), parent_app_map);
//        Dtype *offsets = top[1]->mutable_cpu_data() + child_joint * 3;
//        offsets[0] = max_offset_x;
//        offsets[1] = max_offset_y;
//        offsets[2] = max_child_idpr_map_idx;
//        //LOG(ERROR) << "Offsets and map idx for parent joint " << child_joint << ": " << max_offset_x << ", " << max_offset_y << ", " << max_child_idpr_map_idx;
//      }
//
      // --------------------------------------- BACKTRACKING -------------------------------------------------

      // TODO: no gpu yet!
      Dtype *backtrack_data = top[2]->mutable_cpu_data();

      const Dtype *result_map = top[0]->cpu_data();
      //Dtype *result_map = top_data;
      int max_x = 0;
      int max_y = 0;

      map_max_cpu(result_map, max_x, max_y);
      //map_max_gpu(result_map, height_, width_, max_x, max_y);
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

      // LOG(ERROR) << "Total fwtime: " << total_timer.MilliSeconds() << " ms.";
}

template <typename Dtype>
void InferenceLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                         const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

}


INSTANTIATE_LAYER_GPU_FUNCS(InferenceLayer);

}
