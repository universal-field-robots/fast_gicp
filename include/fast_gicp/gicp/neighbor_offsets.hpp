#ifndef FAST_GICP_NEIGHBOR_OFFSETS_HPP
#define FAST_GICP_NEIGHBOR_OFFSETS_HPP

#include <vector>
#include <iostream>
#include <Eigen/Core>
#include <fast_gicp/gicp/gicp_settings.hpp>

namespace fast_gicp {

static std::vector<Eigen::Vector3i, Eigen::aligned_allocator<Eigen::Vector3i>> neighbor_offsets(NeighborSearchMethod search_method) {
  switch(search_method) {
      // clang-format off
    default:
      std::cerr << "here must not be reached" << std::endl;
      abort();
    case NeighborSearchMethod::DIRECT1:
      return std::vector<Eigen::Vector3i, Eigen::aligned_allocator<Eigen::Vector3i>>{
        Eigen::Vector3i(0, 0, 0)
      };
    case NeighborSearchMethod::DIRECT7:
      return std::vector<Eigen::Vector3i, Eigen::aligned_allocator<Eigen::Vector3i>>{
        Eigen::Vector3i(0, 0, 0),
        Eigen::Vector3i(1, 0, 0),
        Eigen::Vector3i(-1, 0, 0),
        Eigen::Vector3i(0, 1, 0),
        Eigen::Vector3i(0, -1, 0),
        Eigen::Vector3i(0, 0, 1),
        Eigen::Vector3i(0, 0, -1)
      };
      // clang-format on
  }

  std::vector<Eigen::Vector3i, Eigen::aligned_allocator<Eigen::Vector3i>> offsets27;
  for(int i = 0; i < 3; i++) {
    for(int j = 0; j < 3; j++) {
      for(int k = 0; k < 3; k++) {
        offsets27.push_back(Eigen::Vector3i(i, j, k));
      }
    }
  }
  return offsets27;
}

}  // namespace fast_gicp

#endif