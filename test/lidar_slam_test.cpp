#include <gtest/gtest.h>
#include <lidar_slam/lidar_slam.h>
#include <random>

using namespace lidar_slam;

class LidarSlamTest : public ::testing::Test
{
  public:
    LidarSlamTest() : params_{}, slam_{params_}
    {
        //params_.debug_logging = true;
        slam_.SetOdometryCallback(
            [this](const Eigen::Matrix4d t, const std::uint64_t s) { this->PublishOdometry(t, s); });

        slam_.SetMappingCallback([this](const Eigen::Matrix4d t, const std::uint64_t s) { this->PublishMap(t, s); });
    }

  protected:
    LidarSlam slam_;
    LidarSlamParameters params_;

    std::uint32_t odometry_published{};
    std::uint32_t mapping_published{};

    Eigen::Matrix4d latest_odometry_{};
    Eigen::Matrix4d latest_mapping_{};

    std::uint64_t latest_odometry_stamp_;
    std::uint64_t latest_mapping_stamp_;

    virtual void PublishOdometry(const Eigen::Matrix4d odometry, const std::uint64_t stamp)
    {
        odometry_published ++;
        latest_odometry_ = odometry;
        latest_odometry_stamp_ = stamp;
    }

    virtual void PublishMap(const Eigen::Matrix4d transform, const std::uint64_t stamp)
    {
        mapping_published ++;
        latest_mapping_ = transform;
        latest_mapping_stamp_ = stamp;
    }

    /// Prepares "distinctive" 3D surface, with strong features (i.e. almost guaranteed to align)
    LidarSlam::PointCloudPtr DistinctiveSurface()
    {
        std::mt19937 gen(1);
        std::normal_distribution<float> rnd(0.F, 5.0F);

        LidarSlam::PointCloudPtr cloud1(new LidarSlam::PointCloud());
        for (std::size_t i = 0U; i < 100U; i++)
        {
            // points in Z=0 plane
            pcl::PointXYZ p{};
            p.x = rnd(gen);
            p.y = rnd(gen);
            p.z = 0.F;
            cloud1->push_back(p);

            p.x = rnd(gen);
            p.y = 0.F;
            p.z = 5.F + rnd(gen);
            cloud1->push_back(p);

            p.x = rnd(gen);
            p.y = rnd(gen);
            p.z = -5.F - p.x - p.y;
            cloud1->push_back(p);
        }

        return cloud1;
    }

    LidarSlam::PointCloudPtr TranslateSurface(const LidarSlam::PointCloudPtr cloud1, const float dx = 0.5F, const float dy = 0.5F, const float dz = 0.5F)
    {
        LidarSlam::PointCloudPtr cloud2(new LidarSlam::PointCloud());

        for (std::size_t i = 0U; i < cloud1->size(); i++)
        {
            pcl::PointXYZ p = cloud1->at(i);
            p.x += 0.5;
            p.y += 0.5;
            p.z += 0.5;
            cloud2->push_back(p);
        }
        return cloud2;
    }

};

/// Basic test to check that SLAM throws only when expected
TEST_F(LidarSlamTest, BasicTest)
{
    LidarSlam::PointCloudPtr empty_cloud;
    LidarSlam::PointCloudPtr cloud1(new LidarSlam::PointCloud());
    LidarSlam::PointCloudPtr cloud2(new LidarSlam::PointCloud());

    cloud1->header.stamp = 1U;
    cloud2->header.stamp = 2U;

    pcl::PointXYZ p(1.F, 2.F, 3.F);
    cloud1->push_back(p);
    cloud2->push_back(p);

    EXPECT_THROW(slam_.AddPointCloud(empty_cloud), std::runtime_error);

    EXPECT_NO_THROW(slam_.AddPointCloud(cloud1));
    EXPECT_THROW(slam_.AddPointCloud(cloud1), std::runtime_error);

    EXPECT_NO_THROW(slam_.AddPointCloud(cloud2));
    std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(10));
    slam_.Stop();

    EXPECT_EQ(0U, odometry_published);
    EXPECT_EQ(0U, mapping_published);
}

/// Basic Odometry test
TEST_F(LidarSlamTest, BasicOdometryTest)
{
    auto cloud1 = DistinctiveSurface();
    auto cloud2 = TranslateSurface(cloud1, 0.5, 0.5, 0.5);

    cloud1->header.stamp = 1U;
    cloud2->header.stamp = 2U;

    EXPECT_NO_THROW(slam_.AddPointCloud(cloud1));
    std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(10));
    ASSERT_EQ(1U, odometry_published);  // at first slam sends "initialization" odometry
    EXPECT_EQ(1U, latest_odometry_stamp_);

    EXPECT_NO_THROW(slam_.AddPointCloud(cloud2));
    std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(10));
    slam_.Stop();

    ASSERT_EQ(2U, odometry_published);
    EXPECT_EQ(0U, mapping_published);
    EXPECT_EQ(2U, latest_odometry_stamp_);

    auto res = LidarSlam::GetAbsoluteShiftAngle(latest_odometry_);

    EXPECT_NEAR(0.866, res.first, 0.01F);
    EXPECT_NEAR(0.F, res.second, 0.01F);

    EXPECT_NEAR(-0.5, latest_odometry_(0,3), 0.01F);
    EXPECT_NEAR(-0.5, latest_odometry_(1,3), 0.01F);
    EXPECT_NEAR(-0.5, latest_odometry_(2,3), 0.01F);
}

/// Basic Mapping Test
TEST_F(LidarSlamTest, BasicMappingTest)
{
    auto cloud1 = DistinctiveSurface();
    auto cloud2 = TranslateSurface(cloud1, 0.5, 0.5, 0.5);
    auto cloud3 = TranslateSurface(cloud2, 0.5, 0.5, 0.5);

    cloud1->header.stamp = 1U;
    cloud2->header.stamp = 2U;
    cloud3->header.stamp = 3U;

    EXPECT_NO_THROW(slam_.AddPointCloud(cloud1));
    std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(10));
    ASSERT_EQ(1U, odometry_published);  // at first slam sends "initialization" odometry
    EXPECT_EQ(1U, latest_odometry_stamp_);

    EXPECT_NO_THROW(slam_.AddPointCloud(cloud2));
    std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(300));

    EXPECT_NO_THROW(slam_.AddPointCloud(cloud3));
    std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(300));

    slam_.Stop();

    ASSERT_EQ(3U, odometry_published);
    EXPECT_EQ(1U, mapping_published);
    EXPECT_EQ(3U, latest_odometry_stamp_);
    EXPECT_EQ(2U, latest_mapping_stamp_);


    auto res = LidarSlam::GetAbsoluteShiftAngle(latest_odometry_);
    EXPECT_NEAR(0.F, res.second, 0.01F);
    EXPECT_NEAR(-1, latest_odometry_(0,3), 0.01F);
    EXPECT_NEAR(-1, latest_odometry_(1,3), 0.01F);
    EXPECT_NEAR(-1, latest_odometry_(2,3), 0.01F);


    auto res2 = LidarSlam::GetAbsoluteShiftAngle(latest_mapping_);
    EXPECT_NEAR(0.F, res2.second, 0.01F);
    EXPECT_NEAR(-0.5, latest_mapping_(0,3), 0.01F);
    EXPECT_NEAR(-0.5, latest_mapping_(1,3), 0.01F);
    EXPECT_NEAR(-0.5, latest_mapping_(2,3), 0.01F);
}
