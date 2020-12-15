#include <fast_gicp/ndt/fast_ndt_cuda.hpp>

#include <fast_gicp/cuda/fast_ndt_cuda.cuh>

namespace fast_gicp {

template <typename PointSource, typename PointTarget>
FastNDTCuda<PointSource, PointTarget>::FastNDTCuda() {
  ndt_cuda_.reset(new cuda::FastNDTCudaCore());
  ndt_cuda_->set_resolution(3.0);
  ndt_cuda_->set_distance_mode(NDTDistanceMode::P2D);
  ndt_cuda_->set_neighbor_search_method(NeighborSearchMethod::DIRECT1);
}

template <typename PointSource, typename PointTarget>
FastNDTCuda<PointSource, PointTarget>::~FastNDTCuda() {}

template <typename PointSource, typename PointTarget>
void FastNDTCuda<PointSource, PointTarget>::setDistanceMode(NDTDistanceMode mode) {
  ndt_cuda_->set_distance_mode(mode);
}

template <typename PointSource, typename PointTarget>
void FastNDTCuda<PointSource, PointTarget>::setNeighborSearchMethod(NeighborSearchMethod method) {
  ndt_cuda_->set_neighbor_search_method(method);
}

template <typename PointSource, typename PointTarget>
void FastNDTCuda<PointSource, PointTarget>::setResolution(double resolution) {
  ndt_cuda_->set_resolution(resolution);
}

template <typename PointSource, typename PointTarget>
void FastNDTCuda<PointSource, PointTarget>::swapSourceAndTarget() {
  ndt_cuda_->swap_source_and_target();
}

template <typename PointSource, typename PointTarget>
void FastNDTCuda<PointSource, PointTarget>::clearSource() {
  this->input_.reset();
}

template <typename PointSource, typename PointTarget>
void FastNDTCuda<PointSource, PointTarget>::clearTarget() {
  this->target_.reset();
}

template <typename PointSource, typename PointTarget>
void FastNDTCuda<PointSource, PointTarget>::setInputSource(const PointCloudSourceConstPtr& cloud) {
  // the input cloud is the same as the previous one
  if (cloud == this->input_) {
    return;
  }

  pcl::Registration<PointSource, PointTarget>::setInputSource(cloud);

  std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> points(cloud->size());
  std::transform(cloud->begin(), cloud->end(), points.begin(), [=](const PointSource& pt) { return pt.getVector3fMap(); });

  ndt_cuda_->set_source_cloud(points);
}

template <typename PointSource, typename PointTarget>
void FastNDTCuda<PointSource, PointTarget>::setInputTarget(const PointCloudTargetConstPtr& cloud) {
  if (cloud == this->target_) {
    return;
  }

  pcl::Registration<PointSource, PointTarget>::setInputTarget(cloud);

  std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> points(cloud->size());
  std::transform(cloud->begin(), cloud->end(), points.begin(), [=](const PointSource& pt) { return pt.getVector3fMap(); });

  ndt_cuda_->set_target_cloud(points);
}

template <typename PointSource, typename PointTarget>
void FastNDTCuda<PointSource, PointTarget>::computeTransformation(PointCloudSource& output, const Matrix4& guess) {
  LsqRegistration<PointSource, PointTarget>::computeTransformation(output, guess);
}

template <typename PointSource, typename PointTarget>
double FastNDTCuda<PointSource, PointTarget>::linearize(const Eigen::Isometry3d& trans, Eigen::Matrix<double, 6, 6>* H, Eigen::Matrix<double, 6, 1>* b) {
  return ndt_cuda_->linearize(trans, H, b);
}

template <typename PointSource, typename PointTarget>
double FastNDTCuda<PointSource, PointTarget>::compute_error(const Eigen::Isometry3d& trans) {
  return ndt_cuda_->compute_error(trans);
}
}  // namespace fast_gicp