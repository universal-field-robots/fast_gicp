#ifndef FAST_GICP_FAST_NDT_CUDA_CORE_CUH
#define FAST_GICP_FAST_NDT_CUDA_CORE_CUH

#include <memory>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include <fast_gicp/gicp/gicp_settings.hpp>

namespace thrust {

template <typename T1, typename T2>
struct pair;

template <typename T>
class device_ptr;

template <typename T>
class device_allocator;

template <typename T, typename Alloc>
class device_vector;
}  // namespace thrust

namespace fast_gicp {
namespace cuda {

struct NDTHyperParams;
class GaussianVoxelMap;

class FastNDTCudaCore {
public:
  using Scalars = thrust::device_vector<float, thrust::device_allocator<float>>;
  using Points = thrust::device_vector<Eigen::Vector3f, thrust::device_allocator<Eigen::Vector3f>>;
  using Matrices = thrust::device_vector<Eigen::Matrix3f, thrust::device_allocator<Eigen::Matrix3f>>;
  using Poses = thrust::device_vector<Eigen::Isometry3f, thrust::device_allocator<Eigen::Isometry3f>>;
  using Offsets = thrust::device_vector<Eigen::Vector3i, thrust::device_allocator<Eigen::Vector3i>>;
  using Correspondences = thrust::device_vector<thrust::pair<int, int>, thrust::device_allocator<thrust::pair<int, int>>>;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  FastNDTCudaCore();
  ~FastNDTCudaCore();

  void set_resolution(double resolution);
  void set_min_points_per_voxel(int min_points);
  void set_distance_mode(NDTDistanceMode mode);
  void set_neighbor_search_method(NeighborSearchMethod search_method);

  void swap_source_and_target();
  void set_source_cloud(const std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& cloud);
  void set_target_cloud(const std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& cloud);

  void update_correspondences(const thrust::device_ptr<const Eigen::Isometry3f>& trans_ptr);

  double linearize(const Eigen::Isometry3d& trans, Eigen::Matrix<double, 6, 6>* H, Eigen::Matrix<double, 6, 1>* b);

  double compute_error(const Eigen::Isometry3d& trans);

public:
  NDTDistanceMode distance_mode;
  std::vector<Eigen::Vector3i, Eigen::aligned_allocator<Eigen::Vector3i>> host_offsets;
  std::unique_ptr<Offsets> offsets;

  std::unique_ptr<NDTHyperParams> hyper_params;
  std::unique_ptr<thrust::device_vector<NDTHyperParams, thrust::device_allocator<NDTHyperParams>>> hyper_params_ptr;

  std::unique_ptr<Points> source_points;
  std::unique_ptr<Points> target_points;

  std::unique_ptr<GaussianVoxelMap> source_voxelmap;
  std::unique_ptr<GaussianVoxelMap> target_voxelmap;
  std::unique_ptr<Matrices> source_voxel_inv_covs;
  std::unique_ptr<Matrices> target_voxel_inv_covs;

  Eigen::Isometry3f linearized_x;
  std::unique_ptr<Scalars> correspondence_weights;
  std::unique_ptr<Correspondences> correspondences;
};

}  // namespace cuda
}  // namespace fast_gicp

#endif