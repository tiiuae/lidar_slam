
#include <gtest/gtest.h>
#include <lidar_slam/lidar_slam.h>

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