#include <fast_gicp/ndt/fast_ndt_cuda.hpp>
#include <fast_gicp/ndt/impl/fast_ndt_cuda_impl.hpp>

template class fast_gicp::FastNDTCuda<pcl::PointXYZ, pcl::PointXYZ>;
template class fast_gicp::FastNDTCuda<pcl::PointXYZI, pcl::PointXYZI>;
