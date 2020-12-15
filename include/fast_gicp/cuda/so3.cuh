#ifndef FAST_GICP_CUDA_SO3_CUH
#define FAST_GICP_CUDA_SO3_CUH

#include <Eigen/Core>
#include <thrust/device_ptr.h>

namespace fast_gicp {
namespace cuda {

// skew symmetric matrix
inline __host__ __device__ Eigen::Matrix3f skew_symmetric(const Eigen::Vector3f& x) {
  Eigen::Matrix3f skew = Eigen::Matrix3f::Zero();
  skew(0, 1) = -x[2];
  skew(0, 2) = x[1];
  skew(1, 0) = x[2];
  skew(1, 2) = -x[0];
  skew(2, 0) = -x[1];
  skew(2, 1) = x[0];

  return skew;
}

}  // namespace cuda
}  // namespace fast_gicp
#endif