#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl/registration/gicp.h>
#include <mutex>
#include <thread>

#ifndef FOG_SW_LIDAR_SLAM_H
#define FOG_SW_LIDAR_SLAM_H

namespace lidar_slam
{

/// (almost) Real-Time SLAM algorithm, works with PointClouds for now
class LidarSlam
{
  public:
    using Point = pcl::PointXYZ;
    using PointCloud = pcl::PointCloud<Point>;
    using PointCloudPtr = PointCloud::Ptr;
    using CallbackType = std::function<void(const Eigen::Matrix4f transform, const std::uint64_t stamp)>;

    explicit LidarSlam()
        : output_callback_{}, icp1_(), icp2_(), previous_cloud_{}, latest_cloud_{}, processing_thread_{}, processing_thread_running_{}
    {
    }

    void SetCallback(CallbackType && callback)
    {
        output_callback_ = callback;
    }

    ~LidarSlam()
    {
        processing_thread_running_ = false;
        processing_thread_.join();
    }

    void GrabPointCloud(const PointCloudPtr& msg)
    {
        const std::lock_guard<std::mutex> guard(buffer_mutex_);
        latest_cloud_ = msg;
    }

  private:

    void Run();

    CallbackType output_callback_;
    pcl::GeneralizedIterativeClosestPoint<Point, Point> icp1_;
    pcl::GeneralizedIterativeClosestPoint<Point, Point> icp2_;
    PointCloudPtr previous_cloud_;
    PointCloudPtr latest_cloud_;

    Eigen::Matrix4f previous_transform_;

    // static const std::size_t BufferSize = 5U;
    // std::array<CloudMsg::SharedPtr, BufferSize> buffer_;
    // std::size_t buffer_head_,buffer_tail_;
    std::mutex buffer_mutex_;
    std::thread processing_thread_;
    std::atomic_bool processing_thread_running_;
};

} // namespace lidar_slam

#endif  // FOG_SW_LIDAR_SLAM_H
