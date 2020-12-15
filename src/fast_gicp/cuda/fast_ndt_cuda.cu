#include <fast_gicp/cuda/fast_ndt_cuda.cuh>

#include <thrust/pair.h>
#include <thrust/future.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <fast_gicp/gicp/neighbor_offsets.hpp>
#include <fast_gicp/cuda/ndt_hyper_params.cuh>
#include <fast_gicp/cuda/gaussian_voxelmap.cuh>
#include <fast_gicp/cuda/covariance_estimation.cuh>
#include <fast_gicp/cuda/find_voxel_correspondences.cuh>
#include <fast_gicp/cuda/compute_ndt_derivatives.cuh>

#include <glk/cuda_magic_headers.hpp>
#include <glk/thin_lines.hpp>
#include <glk/pointcloud_buffer.hpp>
#include <glk/pointcloud_buffer_cuda.hpp>
#include <glk/primitives/primitives.hpp>
#include <guik/viewer/light_viewer.hpp>

namespace fast_gicp {
namespace cuda {

FastNDTCudaCore::FastNDTCudaCore() {
  cudaDeviceSynchronize();

  distance_mode = NDTDistanceMode::P2D;
  host_offsets = fast_gicp::neighbor_offsets(NeighborSearchMethod::DIRECT7);
  offsets.reset(new thrust::device_vector<Eigen::Vector3i>);
  *offsets = host_offsets;

  hyper_params.reset(new NDTHyperParams(2.0f, 0.55f, 10));
  hyper_params_ptr.reset(new thrust::device_vector<NDTHyperParams>(1));
  (*hyper_params_ptr)[0] = *hyper_params;

  source_points.reset(new Points());
  target_points.reset(new Points());
  source_voxelmap.reset(new GaussianVoxelMap(hyper_params->resolution));
  target_voxelmap.reset(new GaussianVoxelMap(hyper_params->resolution));
  source_voxel_inv_covs.reset(new Matrices());
  target_voxel_inv_covs.reset(new Matrices());

  linearized_x.setIdentity();
  correspondence_weights.reset(new thrust::device_vector<float>);
  correspondences.reset(new thrust::device_vector<thrust::pair<int, int>>);
}

FastNDTCudaCore::~FastNDTCudaCore() {}

void FastNDTCudaCore::set_resolution(double resolution) {
  hyper_params->resolution = resolution;
  (*hyper_params_ptr)[0] = hyper_params->compute_constants();

  source_voxelmap->set_resolution(resolution);
  target_voxelmap->set_resolution(resolution);
}

void FastNDTCudaCore::set_min_points_per_voxel(int min_points) {
  hyper_params->min_points_per_voxel = min_points;
  (*hyper_params_ptr)[0] = hyper_params->compute_constants();
}

void FastNDTCudaCore::set_distance_mode(NDTDistanceMode mode) {
  this->distance_mode = mode;
}

void FastNDTCudaCore::set_neighbor_search_method(NeighborSearchMethod search_method) {
  host_offsets = fast_gicp::neighbor_offsets(search_method);
  *offsets = host_offsets;
}

void FastNDTCudaCore::swap_source_and_target() {
  source_points.swap(target_points);
  source_voxelmap.swap(target_voxelmap);

  if (target_points->size() != target_voxelmap->voxel_means.size()) {
    target_voxelmap->create_voxelmap(*target_points);

    thrust::device_ptr<const int> min_points(&thrust::raw_pointer_cast(hyper_params_ptr->data())->min_points_per_voxel);
    regularize_covariances_ndt(min_points, target_voxelmap->num_points, target_voxelmap->voxel_covs);
    calc_inverse_covariances(target_voxelmap->voxel_covs, *target_voxel_inv_covs);
  }

  if (distance_mode == NDTDistanceMode::D2D && source_points->size() != source_voxelmap->voxel_means.size()) {
    source_voxelmap->create_voxelmap(*source_points);

    thrust::device_ptr<const int> min_points(&thrust::raw_pointer_cast(hyper_params_ptr->data())->min_points_per_voxel);
    regularize_covariances_ndt(min_points, source_voxelmap->num_points, source_voxelmap->voxel_covs);
    calc_inverse_covariances(source_voxelmap->voxel_covs, *source_voxel_inv_covs);
  }
}

void FastNDTCudaCore::set_source_cloud(const std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& cloud) {
  *source_points = cloud;

  if (distance_mode == NDTDistanceMode::D2D) {
    source_voxelmap->create_voxelmap(*source_points);

    thrust::device_ptr<const int> min_points(&thrust::raw_pointer_cast(hyper_params_ptr->data())->min_points_per_voxel);
    regularize_covariances_ndt(min_points, source_voxelmap->num_points, source_voxelmap->voxel_covs);
    calc_inverse_covariances(source_voxelmap->voxel_covs, *source_voxel_inv_covs);
  }
}

void FastNDTCudaCore::set_target_cloud(const std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& cloud) {
  *target_points = cloud;
  target_voxelmap->create_voxelmap(*target_points);
  thrust::device_ptr<const int> min_points(&thrust::raw_pointer_cast(hyper_params_ptr->data())->min_points_per_voxel);
  regularize_covariances_ndt(min_points, target_voxelmap->num_points, target_voxelmap->voxel_covs);
  calc_inverse_covariances(target_voxelmap->voxel_covs, *target_voxel_inv_covs);
}

void FastNDTCudaCore::update_correspondences(const thrust::device_ptr<const Eigen::Isometry3f>& trans_ptr) {
  if (distance_mode == NDTDistanceMode::P2D) {
    find_voxel_correspondences_wifh_offsets(*source_points, *target_voxelmap, trans_ptr, *offsets, *correspondences);
  } else {
    find_voxel_correspondences_wifh_offsets(source_voxelmap->voxel_means, *target_voxelmap, trans_ptr, *offsets, *correspondences);
  }

  int corrs_per_scan = correspondences->size() / host_offsets.size();
  correspondence_weights->resize(correspondences->size());
  for (int i = 0; i < host_offsets.size(); i++) {
    double w = std::exp(-2.0 * host_offsets[i].cast<double>().norm());
    thrust::fill(correspondence_weights->begin() + corrs_per_scan * i, correspondence_weights->begin() + corrs_per_scan * (i + 1), w);
  }
}

double FastNDTCudaCore::linearize(const Eigen::Isometry3d& trans, Eigen::Matrix<double, 6, 6>* H, Eigen::Matrix<double, 6, 1>* b) {
  linearized_x = trans.cast<float>();
  thrust::device_vector<Eigen::Isometry3f> trans_ptr(1);
  trans_ptr[0] = linearized_x;

  update_correspondences(trans_ptr.data());

  thrust::device_vector<LinearizedSystem> outputs(1);

  auto event = compute_ndt_derivatives(
    hyper_params_ptr->data(),
    *source_points,
    *target_voxelmap,
    *target_voxel_inv_covs,
    *correspondence_weights,
    *correspondences,
    trans_ptr.data(),
    outputs.data());
  event.wait();

  LinearizedSystem result = outputs[0];

  if (H && b) {
    *H = result.H.cast<double>();
    *b = result.b.cast<double>();
  }

  return result.error;
}

double FastNDTCudaCore::compute_error(const Eigen::Isometry3d& trans) {
  thrust::device_vector<Eigen::Isometry3f> trans_ptr(1);
  trans_ptr[0] = trans.cast<float>();

  thrust::device_vector<LinearizedSystem> outputs(1);

  auto event = compute_ndt_derivatives(
    hyper_params_ptr->data(),
    *source_points,
    *target_voxelmap,
    *target_voxel_inv_covs,
    *correspondence_weights,
    *correspondences,
    trans_ptr.data(),
    outputs.data());
  event.wait();

  LinearizedSystem result = outputs[0];

  return result.error;
}

}  // namespace cuda
}  // namespace fast_gicp