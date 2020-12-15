#ifndef FAST_GICP_CUDA_LINEARIZED_SYSTEM_CUH
#define FAST_GICP_CUDA_LINEARIZED_SYSTEM_CUH

#include <Eigen/Core>
#include <thrust/device_ptr.h>

namespace fast_gicp {
namespace cuda {

struct LinearizedSystem {
public:
  static __host__ __device__ LinearizedSystem zero() {
    LinearizedSystem system;
    system.error = 0.0f;
    system.H.setZero();
    system.b.setZero();
    return system;
  }

  __host__ __device__ LinearizedSystem& operator+=(const LinearizedSystem& rhs) {
    error += rhs.error;
    H += rhs.H;
    b += rhs.b;
    return *this;
  }

  __host__ __device__ LinearizedSystem operator+(const LinearizedSystem& rhs) const {
    LinearizedSystem sum;
    sum.error = error + rhs.error;
    sum.H = H + rhs.H;
    sum.b = b + rhs.b;
    return sum;
  }

  float error;
  Eigen::Matrix<float, 6, 6> H;
  Eigen::Matrix<float, 6, 1> b;
};

}  // namespace cuda
}  // namespace fast_gicp

#endif