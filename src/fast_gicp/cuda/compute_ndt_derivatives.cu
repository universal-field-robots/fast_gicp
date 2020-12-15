#include <fast_gicp/cuda/compute_ndt_derivatives.cuh>

#include <thrust/async/reduce.h>
#include <thrust/iterator/transform_iterator.h>

#include <fast_gicp/cuda/so3.cuh>

namespace fast_gicp {
namespace cuda {

struct compute_ndt_derivatives_kernel {
  compute_ndt_derivatives_kernel(
    const thrust::device_ptr<const NDTHyperParams>& ndt_params_ptr,
    const thrust::device_vector<Eigen::Vector3f>& src_points,
    const GaussianVoxelMap& target_voxelmap,
    const thrust::device_vector<Eigen::Matrix3f>& target_voxel_inv_covs,
    const thrust::device_ptr<const Eigen::Isometry3f>& trans_ptr)
  : ndt_params_ptr(ndt_params_ptr),
    x_ptr(trans_ptr),
    src_points_ptr(src_points.data()),
    voxel_num_points_ptr(target_voxelmap.num_points.data()),
    voxel_means_ptr(target_voxelmap.voxel_means.data()),
    voxel_inv_covs_ptr(target_voxel_inv_covs.data()) {}

  __device__ LinearizedSystem operator()(const thrust::tuple<float, thrust::pair<int, int>>& input) const {
    const float offset_w = thrust::get<0>(input);
    const auto& correspondence = thrust::get<1>(input);

    const auto& hyper_params = *thrust::raw_pointer_cast(ndt_params_ptr);
    const int src_index = correspondence.first;
    const int target_index = correspondence.second;

    if (src_index < 0 || target_index < 0 || thrust::raw_pointer_cast(voxel_num_points_ptr)[target_index] < hyper_params.min_points_per_voxel) {
      return LinearizedSystem::zero();
    }

    const auto& trans = *thrust::raw_pointer_cast(x_ptr);
    const auto& src_pt = thrust::raw_pointer_cast(src_points_ptr)[src_index];
    const Eigen::Vector3f transed_src_pt = trans * src_pt;

    const auto& target_num_points = thrust::raw_pointer_cast(voxel_num_points_ptr)[target_index];
    const auto& target_mean = thrust::raw_pointer_cast(voxel_means_ptr)[target_index];
    const auto& target_cov_inv = thrust::raw_pointer_cast(voxel_inv_covs_ptr)[target_index];

    const float w = sqrtf(hyper_params.gauss_d2) * sqrtf(offset_w);
    Eigen::Vector3f error = w * target_cov_inv * (target_mean - transed_src_pt);

    Eigen::Matrix<float, 3, 6> dtdx0;
    dtdx0.block<3, 3>(0, 0) = skew_symmetric(transed_src_pt);
    dtdx0.block<3, 3>(0, 3) = -Eigen::Matrix3f::Identity();

    Eigen::Matrix<float, 3, 6> J = w * target_cov_inv * dtdx0;

    LinearizedSystem result;
    result.error = error.squaredNorm();
    result.H = J.transpose() * J;
    result.b = J.transpose() * error;

    return result;
  }

  thrust::device_ptr<const NDTHyperParams> ndt_params_ptr;
  thrust::device_ptr<const Eigen::Isometry3f> x_ptr;
  thrust::device_ptr<const Eigen::Vector3f> src_points_ptr;
  thrust::device_ptr<const int> voxel_num_points_ptr;
  thrust::device_ptr<const Eigen::Vector3f> voxel_means_ptr;
  thrust::device_ptr<const Eigen::Matrix3f> voxel_inv_covs_ptr;
};

thrust::system::cuda::unique_eager_event compute_ndt_derivatives(
  const thrust::device_ptr<const NDTHyperParams>& ndt_params_ptr,
  const thrust::device_vector<Eigen::Vector3f>& src_points,
  const GaussianVoxelMap& target_voxelmap,
  const thrust::device_vector<Eigen::Matrix3f>& target_voxel_inv_covs,
  const thrust::device_vector<float>& correspondence_weights,
  const thrust::device_vector<thrust::pair<int, int>>& correspondences,
  const thrust::device_ptr<const Eigen::Isometry3f>& x_ptr,
  const thrust::device_ptr<LinearizedSystem>& output_ptr) {
  compute_ndt_derivatives_kernel kernel(ndt_params_ptr, src_points, target_voxelmap, target_voxel_inv_covs, x_ptr);
  auto first = thrust::make_transform_iterator(thrust::make_zip_iterator(thrust::make_tuple(correspondence_weights.begin(), correspondences.begin())), kernel);
  auto last = thrust::make_transform_iterator(thrust::make_zip_iterator(thrust::make_tuple(correspondence_weights.end(), correspondences.end())), kernel);

  return thrust::async::reduce_into(first, last, output_ptr, LinearizedSystem::zero(), thrust::plus<LinearizedSystem>());
}

}  // namespace cuda
}  // namespace fast_gicp