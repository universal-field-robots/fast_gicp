#include <Eigen/Core>
#include <Eigen/Geometry>

#include <thrust/device_vector.h>
#include <thrust/async/transform.h>
#include <fast_gicp/cuda/vector3_hash.cuh>
#include <fast_gicp/cuda/gaussian_voxelmap.cuh>
#include <fast_gicp/cuda/find_voxel_correspondences.cuh>

namespace fast_gicp {
namespace cuda {

namespace {

// lookup voxel
__host__ __device__ int lookup_voxel(
  thrust::device_ptr<const VoxelMapInfo> voxelmap_info_ptr,
  thrust::device_ptr<const thrust::pair<Eigen::Vector3i, int>> buckets_ptr,
  const Eigen::Vector3f& x,
  const Eigen::Vector3i& offset = Eigen::Vector3i(0, 0, 0)) {
  const VoxelMapInfo& voxelmap_info = *thrust::raw_pointer_cast(voxelmap_info_ptr);

  Eigen::Vector3i coord = calc_voxel_coord(x, voxelmap_info.voxel_resolution) + offset;
  uint64_t hash = vector3i_hash(coord);

  for (int i = 0; i < voxelmap_info.max_bucket_scan_count; i++) {
    uint64_t bucket_index = (hash + i) % voxelmap_info.num_buckets;
    const thrust::pair<Eigen::Vector3i, int>& bucket = thrust::raw_pointer_cast(buckets_ptr)[bucket_index];

    if (bucket.second < 0) {
      return -1;
    }

    if (bucket.first == coord) {
      return bucket.second;
    }
  }

  return -1;
}

struct find_voxel_correspondences_kernel {
  find_voxel_correspondences_kernel(const GaussianVoxelMap& voxelmap, const Eigen::Isometry3f& x)
  : R(x.linear()),
    t(x.translation()),
    voxelmap_info_ptr(voxelmap.voxelmap_info_ptr.data()),
    buckets_ptr(voxelmap.buckets.data()) {}

  __host__ __device__ int operator()(const Eigen::Vector3f& pt) const { return lookup_voxel(voxelmap_info_ptr, buckets_ptr, R * pt + t); }

  const Eigen::Matrix3f R;
  const Eigen::Vector3f t;

  thrust::device_ptr<const VoxelMapInfo> voxelmap_info_ptr;
  thrust::device_ptr<const thrust::pair<Eigen::Vector3i, int>> buckets_ptr;

  thrust::device_ptr<const int> voxel_num_points_ptr;
};

struct find_voxel_correspondences_with_offset_kernel {
  find_voxel_correspondences_with_offset_kernel(
    const GaussianVoxelMap& voxelmap,
    thrust::device_ptr<const Eigen::Isometry3f> trans_ptr,
    thrust::device_ptr<const Eigen::Vector3i> offset_ptr)
  : trans_ptr(trans_ptr),
    offset_ptr(offset_ptr),
    voxelmap_info_ptr(voxelmap.voxelmap_info_ptr.data()),
    buckets_ptr(voxelmap.buckets.data()) {}

  __host__ __device__ thrust::pair<int, int> operator()(const thrust::tuple<int, Eigen::Vector3f>& input) const {
    const int src_index = thrust::get<0>(input);
    const auto& src_pt = thrust::get<1>(input);
    const auto& trans = *thrust::raw_pointer_cast(trans_ptr);
    const auto& offset = *thrust::raw_pointer_cast(offset_ptr);

    const int target_index = lookup_voxel(voxelmap_info_ptr, buckets_ptr, trans * src_pt, offset);
    return thrust::make_pair(src_index, target_index);
  }

  const Eigen::Matrix3f R;
  const Eigen::Vector3f t;
  thrust::device_ptr<const Eigen::Isometry3f> trans_ptr;
  thrust::device_ptr<const Eigen::Vector3i> offset_ptr;

  thrust::device_ptr<const VoxelMapInfo> voxelmap_info_ptr;
  thrust::device_ptr<const thrust::pair<Eigen::Vector3i, int>> buckets_ptr;
};
}  // namespace

void find_voxel_correspondences(
  const thrust::device_vector<Eigen::Vector3f>& src_points,
  const GaussianVoxelMap& voxelmap,
  const Eigen::Isometry3f& x,
  thrust::device_vector<int>& correspondences) {
  correspondences.resize(src_points.size());
  thrust::transform(src_points.begin(), src_points.end(), correspondences.begin(), find_voxel_correspondences_kernel(voxelmap, x));
}

void find_voxel_correspondences_wifh_offsets(
  const thrust::device_vector<Eigen::Vector3f>& src_points,
  const GaussianVoxelMap& voxelmap,
  const thrust::device_ptr<const Eigen::Isometry3f>& x_ptr,
  const thrust::device_vector<Eigen::Vector3i>& offsets,
  thrust::device_vector<thrust::pair<int, int>>& correspondences) {
  correspondences.resize(src_points.size() * offsets.size());

  for (int i = 0; i < offsets.size(); i++) {
    thrust::transform(
      thrust::make_zip_iterator(thrust::make_tuple(thrust::counting_iterator<int>(0), src_points.begin())),
      thrust::make_zip_iterator(thrust::make_tuple(thrust::counting_iterator<int>(src_points.size()), src_points.end())),
      correspondences.begin() + src_points.size() * i,
      find_voxel_correspondences_with_offset_kernel(voxelmap, x_ptr, offsets.data() + i));
  }
}

}  // namespace cuda
}  // namespace fast_gicp
