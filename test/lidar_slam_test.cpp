
#include <gtest/gtest.h>
#include <lidar_slam/lidar_slam.h>

// following includes come last
#include <fast_gicp/gicp/impl/fast_gicp_impl.hpp>
#include <fast_gicp/gicp/impl/fast_vgicp_impl.hpp>
#include <fast_gicp/gicp/impl/lsq_registration_impl.hpp>

template class fast_gicp::FastVGICP<pcl::PointXYZ, pcl::PointXYZ>;
template class fast_gicp::FastGICP<pcl::PointXYZ, pcl::PointXYZ>;
template class fast_gicp::LsqRegistration<pcl::PointXYZ, pcl::PointXYZ>;

using namespace lidar_slam;

class LidarSlamTest : public ::testing::Test
{
  public:
    LidarSlamTest() = default;
};

TEST_F(LidarSlamTest, BasicTest)
{
    LidarSlam slam;
    LidarSlam::PointCloudPtr cloud(new LidarSlam::PointCloud());
    ASSERT_NO_THROW(slam.AddPointCloud(cloud));
}

