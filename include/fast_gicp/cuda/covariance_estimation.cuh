#ifndef FAST_GICP_CUDA_COVARIANCE_ESTIMATION_CUH
#define FAST_GICP_CUDA_COVARIANCE_ESTIMATION_CUH

#include <Eigen/Core>
#include <thrust/device_vector.h>
#include <fast_gicp/gicp/gicp_settings.hpp>

namespace fast_gicp {
namespace cuda {

void covariance_estimation(
  const thrust::device_vector<Eigen::Vector3f>& points,
  int k,
  const thrust::device_vector<int>& k_neighbors,
  thrust::device_vector<Eigen::Matrix3f>& covariances,
  RegularizationMethod method);

void regularize_covariances_ndt(
  const thrust::device_ptr<const int>& min_points_per_voxel_ptr,
  const thrust::device_vector<int>& num_points,
  thrust::device_vector<Eigen::Matrix3f>& covariances);

void calc_inverse_covariances(const thrust::device_vector<Eigen::Matrix3f>& covariances, thrust::device_vector<Eigen::Matrix3f>& inv_covariances);

}  // namespace cuda
}  // namespace fast_gicp

#endif