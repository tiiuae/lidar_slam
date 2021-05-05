#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <memory>

#ifndef FOG_SW_LIDAR_SLAM_H
#define FOG_SW_LIDAR_SLAM_H


namespace lidar_slam
{

/// Most of the algorithmic parameters are collected here, in one single struct
struct LidarSlamParameters
{
    bool automatic_start = true;
    bool optimizer_verbose = false;
    bool gicp_use_l2r_check = false;
    double gicp_resolution = 1.0;
    double gicp_translation_noise = 0.05; // [meters]
    double gicp_rotation_noise = 0.002;  // [radians]
    std::size_t minimal_cloud_size = 20U;
    double new_node_min_translation = 0.1; // [meters]
    double new_node_after_rotation = 0.05; // [radians]
    int loop_closure_min_vertices = 5;
    int loop_closure_edge_rato = 2;
};

/// Lidar-SLAM algorithm, for working with PointClouds
class LidarSlam
{
  public:

    using Point = pcl::PointXYZ;
    using PointCloud = pcl::PointCloud<Point>;
    using PointCloudPtr = PointCloud::Ptr;
    using CloudConstPtr = PointCloud::ConstPtr;

    /// Callback for SLAM outputs
    using CallbackType = std::function<void(const Eigen::Isometry3d transform, const std::uint64_t stamp)>;

    /// Main C-tor
    LidarSlam(const LidarSlamParameters& params = LidarSlamParameters());

    /// Set-up callback which publishes low-latency (pseudo real-time) transform, which may be used for odometry
    /// It contains "drifty" transform from the very-first frame into very last one
    void SetOdometryCallback(CallbackType&& callback);

    /// Set-up callback which publishes high-latency transform, suitable for mapping
    /// returns transform from odometry frame into mapping frame
    void SetMappingCallback(CallbackType&& callback);

    void Start();
    void Stop();

    ~LidarSlam();

    /// Use this function to add new (just arrived) point-cloud into the SLAM
    /// Each new cloud shall have later timestamp than previous
    void AddPointCloud(const PointCloudPtr& msg);

  private:

    struct Impl;
    std::unique_ptr<Impl> impl_;

};

}  // namespace lidar_slam

#endif  // FOG_SW_LIDAR_SLAM_H
