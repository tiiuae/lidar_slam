#include <boost/optional/optional.hpp>
#include <fast_gicp/gicp/fast_vgicp.hpp>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/types/slam3d/edge_se3.h>
#include <memory>
#include <mutex>
#include <thread>

#ifndef FOG_SW_LIDAR_SLAM_H
#define FOG_SW_LIDAR_SLAM_H

#ifdef USE_SLAM_LOGGING
#define SLAM_LOG std::cout
#else
#define SLAM_LOG \
    if (false)   \
    std::cout
#endif

namespace lidar_slam
{

/// Most of the algorithmic parameters are collected here, in one single struct
struct LidarSlamParameters
{
    bool automatic_start = true;
    bool optimizer_verbose = false;
    double gicp_resolution = 0.5;
    double gicp_translation_noise = 0.01;
    double gicp_rotation_noise = 0.001;
    std::size_t minimal_cloud_size = 20U;
    double new_node_min_translation = 0.1; // [meters]
    double new_node_after_rotation = 0.05; // [radians]

};

class G2O_TYPES_SLAM3D_API CloudVertexSE3 : public g2o::VertexSE3
{
  public:
    using PointCloudPtr = pcl::PointCloud<pcl::PointXYZ>::Ptr;
    PointCloudPtr cloud{};
    //std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> covariances{};
    //std::shared_ptr<pcl::search::KdTree<pcl::PointXYZ>> kdtree{};
    CloudVertexSE3() : VertexSE3(){}
    CloudVertexSE3(PointCloudPtr init_cloud) : VertexSE3() {cloud = init_cloud;}
};

/// Lidar-SLAM algorithm, for working with PointClouds
class LidarSlam
{
  public:

    using Point = pcl::PointXYZ;
    using PointCloud = pcl::PointCloud<Point>;
    using PointCloudPtr = PointCloud::Ptr;
    using CallbackType = std::function<void(const Eigen::Isometry3d transform, const std::uint64_t stamp)>;

    /// Main C-tor
    LidarSlam(const LidarSlamParameters& params = LidarSlamParameters());

    /// Set-up callback which publishes low-latency (pseudo real-time) transform, which may be used for odometry
    /// It contains "drifty" transform from the very-first frame into very last one
    void SetOdometryCallback(CallbackType&& callback) { odometry_callback_ = callback; }

    /// Set-up callback which publishes high-latency transform, suitable for mapping
    /// returns transform from odometry frame into mapping frame
    void SetMappingCallback(CallbackType&& callback) { mapping_callback_ = callback; }

    void Start();
    void Stop();

    ~LidarSlam();

    /// Use this function to add new (just arrived) point-cloud into the SLAM
    /// Each new cloud shall have later timestamp than previous
    void AddPointCloud(const PointCloudPtr& msg);

    /// @return pair of translation and rotation misalignment
    static std::pair<double, double> GetAbsoluteShiftAngle(const Eigen::Matrix4d& matrix);

  private:

    CallbackType odometry_callback_{}, mapping_callback_{};

    /// Cloud updated once AddPointCloud() is called
    PointCloudPtr latest_cloud_{};
    /// Corresponding mutex
    std::mutex latest_cloud_mutex_{};

    /// @brief List of vertices, waiting to be added into the Graph
    /// Every such vertex must be marginalized, in order not to ruin currently running optimization
    /// Mapping thread will set them correct when adding into the Graph
    std::vector<CloudVertexSE3*> vertices_{};
    std::vector<g2o::EdgeSE3*> odometryEdges_{};
    /// Mutex, controlling both @ref vertices_ and @ref odometryEdges_
    std::mutex vertices_mutex_{};

    /// Threads for independend SLAM parts
    std::thread odometry_thread_{}, mapping_thread_{};
    /// Respective bool atomics
    std::atomic_bool odometry_thread_running_{};
    std::atomic_bool mapping_thread_running_{};

    /// Main loop-functions for respective threads
    void OdometryThread();
    void MappingThread();

    /// read-only reference to parameters
    /// SLAM may not change it, but it still could be changed from outside
    const LidarSlamParameters& params_;

    int vertice_id_{}, edge_id_{};

    /// Information matrix (i.e. inverse covariance of the 6DoF measurement: X Y Z QX QY QZ)
    Eigen::Matrix<double, 6, 6> icp_information_{};

};

}  // namespace lidar_slam

#endif  // FOG_SW_LIDAR_SLAM_H
