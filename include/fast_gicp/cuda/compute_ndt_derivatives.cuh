#ifndef FAST_GICP_CUDA_COMPUTE_NDT_DERIVATIVES_CUH
#define FAST_GICP_CUDA_COMPUTE_NDT_DERIVATIVES_CUH

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <thrust/future.h>
#include <fast_gicp/cuda/linearized_system.cuh>
#include <fast_gicp/cuda/gaussian_voxelmap.cuh>
#include <fast_gicp/cuda/ndt_hyper_params.cuh>

namespace fast_gicp {
namespace cuda {

thrust::system::cuda::unique_eager_event compute_ndt_derivatives(
  const thrust::device_ptr<const NDTHyperParams>& ndt_params_ptr,
  const thrust::device_vector<Eigen::Vector3f>& src_points,
  const GaussianVoxelMap& target_voxelmap,
  const thrust::device_vector<Eigen::Matrix3f>& target_voxel_inv_covs,
  const thrust::device_vector<float>& correspondence_weights,
  const thrust::device_vector<thrust::pair<int, int>>& correspondences,
  const thrust::device_ptr<const Eigen::Isometry3f>& x_ptr,
  const thrust::device_ptr<LinearizedSystem>& output_ptr);
}
}  // namespace fast_gicp

#endif