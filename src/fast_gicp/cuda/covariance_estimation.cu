#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Eigenvalues>

#include <thrust/sequence.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <fast_gicp/gicp/gicp_settings.hpp>
#include <fast_gicp/cuda/covariance_estimation.cuh>

namespace fast_gicp {
namespace cuda {

namespace {
struct covariance_estimation_kernel {
  covariance_estimation_kernel(const thrust::device_vector<Eigen::Vector3f>& points, int k, const thrust::device_vector<int>& k_neighbors, thrust::device_vector<Eigen::Matrix3f>& covariances)
      : k(k), points_ptr(points.data()), k_neighbors_ptr(k_neighbors.data()), covariances_ptr(covariances.data()) {}

  __host__ __device__ void operator()(int idx) const {
    // target points buffer & nn output buffer
    const Eigen::Vector3f* points = thrust::raw_pointer_cast(points_ptr);
    const int* k_neighbors = thrust::raw_pointer_cast(k_neighbors_ptr) + idx * k;
    Eigen::Matrix3f* cov = thrust::raw_pointer_cast(covariances_ptr) + idx;

    Eigen::Vector3f mean(0.0f, 0.0f, 0.0f);
    cov->setZero();
    for(int i = 0; i < k; i++) {
      const auto& pt = points[k_neighbors[i]];
      mean += pt;
      (*cov) += pt * pt.transpose();
    }
    mean /= k;
    (*cov) = (*cov) / k - mean * mean.transpose();
  }

  const int k;
  thrust::device_ptr<const Eigen::Vector3f> points_ptr;
  thrust::device_ptr<const int> k_neighbors_ptr;

  thrust::device_ptr<Eigen::Matrix3f> covariances_ptr;
};

struct covariance_regularization_svd {
  __host__ __device__ void operator()(Eigen::Matrix3f& cov) const {
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eig;
    eig.computeDirect(cov);

    // why this doen't work...???
    // cov = eig.eigenvectors() * values.asDiagonal() * eig.eigenvectors().inverse();
    Eigen::Matrix3f values = Eigen::Vector3f(1e-3, 1, 1).asDiagonal();
    Eigen::Matrix3f v_inv = eig.eigenvectors().inverse();
    cov = eig.eigenvectors() * values * v_inv;

    // JacobiSVD is not supported on CUDA
    // Eigen::JacobiSVD(cov, Eigen::ComputeFullU | Eigen::ComputeFullV);
    // Eigen::Vector3f values(1, 1, 1e-3);
    // cov = svd.matrixU() * values.asDiagonal() * svd.matrixV().transpose();
  }
};

struct covariance_regularization_frobenius {
  __host__ __device__ void operator()(Eigen::Matrix3f& cov) const {
    float lambda = 1e-3;
    Eigen::Matrix3f C = cov + lambda * Eigen::Matrix3f::Identity();
    Eigen::Matrix3f C_inv = C.inverse();
    Eigen::Matrix3f C_norm = (C_inv / C_inv.norm()).inverse();
    cov = C_norm;
  }
};
}  // namespace

void covariance_estimation(const thrust::device_vector<Eigen::Vector3f>& points, int k, const thrust::device_vector<int>& k_neighbors, thrust::device_vector<Eigen::Matrix3f>& covariances, RegularizationMethod method) {
  thrust::device_vector<int> d_indices(points.size());
  thrust::sequence(d_indices.begin(), d_indices.end());

  covariances.resize(points.size());
  thrust::for_each(d_indices.begin(), d_indices.end(), covariance_estimation_kernel(points, k, k_neighbors, covariances));

  switch(method) {
    default:
      std::cerr << "unimplemented covariance regularization method was selected!!" << std::endl;
      abort();
    case RegularizationMethod::PLANE:
      thrust::for_each(covariances.begin(), covariances.end(), covariance_regularization_svd());
      break;
    case RegularizationMethod::FROBENIUS:
      thrust::for_each(covariances.begin(), covariances.end(), covariance_regularization_frobenius());
      break;
  }
}

/**
 * covariance regularization for NDT
 **/

struct covariance_regularization_ndt {
  covariance_regularization_ndt(
    const thrust::device_ptr<const int>& min_points_per_voxel_ptr,
    const thrust::device_vector<int>& num_points,
    thrust::device_vector<Eigen::Matrix3f>& covs)
  : min_points_per_voxel_ptr(min_points_per_voxel_ptr),
    num_points_ptr(num_points.data()),
    covs_ptr(covs.data()) {}

  __host__ __device__ void operator()(int index) const {
    const int num_points = thrust::raw_pointer_cast(num_points_ptr)[index];
    Eigen::Matrix3f& cov = thrust::raw_pointer_cast(covs_ptr)[index];

    if (num_points < *thrust::raw_pointer_cast(min_points_per_voxel_ptr)) {
      cov.setIdentity();
      return;
    }

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eig;
    eig.computeDirect(cov);

    if (eig.eigenvalues()[2] <= 0.0f) {
      cov.setIdentity();
      return;
    }

    Eigen::Vector3f values(1e-3, 1, 1);

    /*
    Eigen::Vector3f values = eig.eigenvalues();
    float min_value = values[2] * 0.01f;
    values[0] = fmaxf(min_value, values[0]);
    values[1] = fmaxf(min_value, values[1]);
    */

    Eigen::Matrix3f values_diagonal = values.asDiagonal();

    Eigen::Matrix3f v_inv = eig.eigenvectors().inverse();
    cov = eig.eigenvectors() * values_diagonal * v_inv;
  }

  thrust::device_ptr<const int> min_points_per_voxel_ptr;
  thrust::device_ptr<const int> num_points_ptr;
  thrust::device_ptr<Eigen::Matrix3f> covs_ptr;
};

void regularize_covariances_ndt(
  const thrust::device_ptr<const int>& min_points_per_voxel_ptr,
  const thrust::device_vector<int>& num_points,
  thrust::device_vector<Eigen::Matrix3f>& covariances) {
  thrust::for_each(
    thrust::counting_iterator<int>(0),
    thrust::counting_iterator<int>(num_points.size()),
    covariance_regularization_ndt(min_points_per_voxel_ptr, num_points, covariances));
}

/**
 * inverse covariance calculation
 **/

struct calc_inverse_covariances_kernel {
  __host__ __device__ Eigen::Matrix3f operator()(const Eigen::Matrix3f& cov) const { return cov.inverse(); }
};

void calc_inverse_covariances(const thrust::device_vector<Eigen::Matrix3f>& covariances, thrust::device_vector<Eigen::Matrix3f>& inv_covariances) {
  inv_covariances.resize(covariances.size());
  thrust::transform(covariances.begin(), covariances.end(), inv_covariances.begin(), calc_inverse_covariances_kernel());
}

}  // namespace cuda
}  // namespace fast_gicp
