#include <gtest/gtest.h>
#include <lidar_slam/lidar_slam.h>
#include <lidar_slam/helpers.h>
#include <random>
#include <thread>

using namespace lidar_slam;

class LidarSlamTest : public ::testing::Test
{
  public:
    LidarSlamTest() : params_{false}, slam_{params_}
    {
        params_.gicp_resolution = 0.25;
        slam_.SetOdometryCallback(
            [this](const Eigen::Isometry3d t, const std::uint64_t s) { this->PublishOdometry(t, s); });

        slam_.SetMappingCallback([this](const Eigen::Isometry3d t, const std::uint64_t s) { this->PublishMap(t, s); });
    }

  protected:
    LidarSlamParameters params_;
    LidarSlam slam_;

    std::uint32_t odometry_published{};
    std::uint32_t mapping_published{};

    Eigen::Isometry3d latest_odometry_{};
    Eigen::Isometry3d latest_mapping_{};

    std::uint64_t latest_odometry_stamp_;
    std::uint64_t latest_mapping_stamp_;

    void PublishOdometry(const Eigen::Isometry3d odometry, const std::uint64_t stamp)
    {
        odometry_published++;
        latest_odometry_ = odometry;
        latest_odometry_stamp_ = stamp;
    }

    void PublishMap(const Eigen::Isometry3d transform, const std::uint64_t stamp)
    {
        mapping_published++;
        latest_mapping_ = transform;
        latest_mapping_stamp_ = stamp;
    }

    /// Prepares "distinctive" 3D surface, with strong features (i.e. almost guaranteed to align)
    static LidarSlam::PointCloudPtr DistinctiveSurface()
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

    static LidarSlam::PointCloudPtr TranslateSurface(const LidarSlam::PointCloudPtr cloud1,
                                                     const float dx = 0.5F,
                                                     const float dy = 0.5F,
                                                     const float dz = 0.5F)
    {
        LidarSlam::PointCloudPtr cloud2(new LidarSlam::PointCloud());

        for (std::size_t i = 0U; i < cloud1->size(); i++)
        {
            pcl::PointXYZ p = cloud1->at(i);
            p.x += dx;
            p.y += dy;
            p.z += dz;
            cloud2->push_back(p);
        }
        return cloud2;
    }
};

/// Basic test to check that SLAM throws only when expected
TEST_F(LidarSlamTest, BasicTest)
{
    slam_.Start();
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
    slam_.Start();
    const float shift = 0.2;
    auto cloud1 = DistinctiveSurface();
    auto cloud2 = TranslateSurface(cloud1, shift, shift, shift);

    cloud1->header.stamp = 1U;
    cloud2->header.stamp = 2U;

    EXPECT_NO_THROW(slam_.AddPointCloud(cloud1));
    // delay needed to make sure that Odometry thread will catch first cloud
    std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(10));
    ASSERT_EQ(1U, odometry_published);  // at first slam sends "initialization" odometry
    EXPECT_EQ(1U, latest_odometry_stamp_);

    EXPECT_NO_THROW(slam_.AddPointCloud(cloud2));
    std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(10));
    // when calling Stop() all threads will finish their active tasks, thus last odometry will be published too
    slam_.Stop();

    ASSERT_EQ(2U, odometry_published);
    EXPECT_EQ(0U, mapping_published);
    EXPECT_EQ(2U, latest_odometry_stamp_);

    auto res = Helpers::GetAbsoluteShiftAngle(latest_odometry_.matrix());

    EXPECT_NEAR(0.346, res.first, 0.01F);
    EXPECT_NEAR(0.F, res.second, 0.01F);

    EXPECT_NEAR(-shift, latest_odometry_(0, 3), 0.01F);
    EXPECT_NEAR(-shift, latest_odometry_(1, 3), 0.01F);
    EXPECT_NEAR(-shift, latest_odometry_(2, 3), 0.01F);
}

/// Basic Mapping Test
TEST_F(LidarSlamTest, BasicMappingTest)
{
    slam_.Start();
    const float shift = 0.13;
    auto cloud1 = DistinctiveSurface();
    auto cloud2 = TranslateSurface(cloud1, shift, shift, shift);
    auto cloud3 = TranslateSurface(cloud2, shift, shift, shift);

    cloud1->header.stamp = 1U;
    cloud2->header.stamp = 2U;
    cloud3->header.stamp = 3U;

    EXPECT_NO_THROW(slam_.AddPointCloud(cloud1));
    std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(100));
    ASSERT_EQ(1U, odometry_published);  // at first slam sends "initialization" odometry
    EXPECT_EQ(1U, latest_odometry_stamp_);

    EXPECT_NO_THROW(slam_.AddPointCloud(cloud2));
    std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(500));
    ASSERT_EQ(2U, odometry_published);

    EXPECT_NO_THROW(slam_.AddPointCloud(cloud3));
    std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(500));

    slam_.Stop();

    ASSERT_EQ(3U, odometry_published);
    EXPECT_EQ(3U, latest_odometry_stamp_);

    auto res = Helpers::GetAbsoluteShiftAngle(latest_odometry_.matrix());
    EXPECT_NEAR(0.F, res.second, 0.01F);
    EXPECT_NEAR(-shift*2, latest_odometry_(0,3), 0.01F);
    EXPECT_NEAR(-shift*2, latest_odometry_(1,3), 0.01F);
    EXPECT_NEAR(-shift*2, latest_odometry_(2,3), 0.01F);

    ASSERT_EQ(1U, mapping_published);
    EXPECT_EQ(2U, latest_mapping_stamp_);
    auto res2 = Helpers::GetAbsoluteShiftAngle(latest_mapping_.matrix());
    EXPECT_NEAR(0.F, res2.second, 0.01F);
    EXPECT_NEAR(0, latest_mapping_(0,3), 0.01F);
    EXPECT_NEAR(0, latest_mapping_(1,3), 0.01F);
    EXPECT_NEAR(0, latest_mapping_(2,3), 0.01F);
}


TEST_F(LidarSlamTest, LoopClosuresTest)
{
    // "enable" loop-closures
    params_.loop_closure_min_vertices = 2;
    params_.loop_closure_edge_rato = 1;
    slam_.Start();


    const float shift = 0.13;
    auto cloud1 = DistinctiveSurface();
    auto cloud2 = TranslateSurface(cloud1, shift, shift, shift);
    auto cloud3 = TranslateSurface(cloud2, shift, shift, shift);
    auto cloud4 = TranslateSurface(cloud3, shift, shift, shift);

    cloud1->header.stamp = 1U;
    cloud2->header.stamp = 2U;
    cloud3->header.stamp = 3U;
    cloud4->header.stamp = 4U;

    EXPECT_NO_THROW(slam_.AddPointCloud(cloud1));
    std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(100));
    ASSERT_EQ(1U, odometry_published);  // at first slam sends "initialization" odometry
    EXPECT_EQ(1U, latest_odometry_stamp_);

    EXPECT_NO_THROW(slam_.AddPointCloud(cloud2));
    std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(500));
    ASSERT_EQ(2U, odometry_published);

    EXPECT_NO_THROW(slam_.AddPointCloud(cloud3));
    std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(500));

    EXPECT_NO_THROW(slam_.AddPointCloud(cloud4));
    std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(600));

    slam_.Stop();

    ASSERT_EQ(4U, odometry_published);
    EXPECT_EQ(4U, latest_odometry_stamp_);

    ASSERT_EQ(2U, mapping_published);
    EXPECT_EQ(3U, latest_mapping_stamp_);
    auto res2 = Helpers::GetAbsoluteShiftAngle(latest_mapping_.matrix());
    EXPECT_NEAR(0.F, res2.second, 0.01F);
    EXPECT_NEAR(0, latest_mapping_(0,3), 0.01F);
    EXPECT_NEAR(0, latest_mapping_(1,3), 0.01F);
    EXPECT_NEAR(0, latest_mapping_(2,3), 0.01F);
}
