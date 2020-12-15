#ifndef FAST_GICP_CUDA_NDT_HYPER_PARAMS_CUH
#define FAST_GICP_CUDA_NDT_HYPER_PARAMS_CUH

#include <cmath>

namespace fast_gicp {
namespace cuda {

struct NDTHyperParams {
  __host__ __device__ NDTHyperParams() {}

  NDTHyperParams(double resolution, double outlier_ratio, int min_points_per_voxel) {
    this->resolution = resolution;
    this->outlier_ratio = outlier_ratio;
    this->min_points_per_voxel = min_points_per_voxel;

    compute_constants();
  }

  __host__ NDTHyperParams& compute_constants() {
    double gauss_c1 = 10.0 * (1 - outlier_ratio);
    double gauss_c2 = outlier_ratio / std::pow(resolution, 3);
    double gauss_d3 = -std::log(gauss_c2);

    gauss_d1 = -std::log(gauss_c1 + gauss_c2) - gauss_d3;
    gauss_d2 = -2 * std::log((-std::log(gauss_c1 * std::exp(-0.5) + gauss_c2) - gauss_d3) / gauss_d1);

    return *this;
  }

  float resolution;
  float outlier_ratio;
  int min_points_per_voxel;

  float gauss_d1;
  float gauss_d2;
};

}  // namespace cuda
}  // namespace fast_gicp

#endif