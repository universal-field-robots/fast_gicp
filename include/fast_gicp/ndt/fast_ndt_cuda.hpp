#ifndef FAST_GICP_FAST_NDT_CUDA_HPP
#define FAST_GICP_FAST_NDT_CUDA_HPP

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/registration/registration.h>

#include <fast_gicp/gicp/lsq_registration.hpp>
#include <fast_gicp/gicp/gicp_settings.hpp>

namespace fast_gicp {
namespace cuda {
class FastNDTCudaCore;
}

/**
 * @brief Fast NDT algorithm with CUDA
 */
template<typename PointSource, typename PointTarget>
class FastNDTCuda : public LsqRegistration<PointSource, PointTarget> {
public:
  using Scalar = float;
  using Matrix4 = typename pcl::Registration<PointSource, PointTarget, Scalar>::Matrix4;

  using PointCloudSource = typename pcl::Registration<PointSource, PointTarget, Scalar>::PointCloudSource;
  using PointCloudSourcePtr = typename PointCloudSource::Ptr;
  using PointCloudSourceConstPtr = typename PointCloudSource::ConstPtr;

  using PointCloudTarget = typename pcl::Registration<PointSource, PointTarget, Scalar>::PointCloudTarget;
  using PointCloudTargetPtr = typename PointCloudTarget::Ptr;
  using PointCloudTargetConstPtr = typename PointCloudTarget::ConstPtr;

protected:
  using pcl::Registration<PointSource, PointTarget, Scalar>::reg_name_;
  using pcl::Registration<PointSource, PointTarget, Scalar>::input_;
  using pcl::Registration<PointSource, PointTarget, Scalar>::target_;

public:
  FastNDTCuda();
  virtual ~FastNDTCuda() override;

  void setDistanceMode(NDTDistanceMode mode);

  void setNeighborSearchMethod(NeighborSearchMethod method);

  void setResolution(double resolution);

  virtual void swapSourceAndTarget();

  virtual void clearSource();

  virtual void clearTarget();

  virtual void setInputSource(const PointCloudSourceConstPtr& cloud) override;

  virtual void setInputTarget(const PointCloudTargetConstPtr& cloud) override;

protected:
  virtual void computeTransformation(PointCloudSource& output, const Matrix4& guess) override;

  virtual double linearize(const Eigen::Isometry3d& trans, Eigen::Matrix<double, 6, 6>* H, Eigen::Matrix<double, 6, 1>* b) override;

  virtual double compute_error(const Eigen::Isometry3d& trans) override;

private:
  std::unique_ptr<cuda::FastNDTCudaCore> ndt_cuda_;
};

}  // namespace fast_gicp

#endif