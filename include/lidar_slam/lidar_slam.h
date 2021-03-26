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

struct LidarSlamParameters
{
    bool automatic_start = true;
    bool optimizer_verbose = false;
    double gicp_resolution = 0.5;
    double gicp_translation_noise = 0.01;
    double gicp_rotation_noise = 0.001;
};

class G2O_TYPES_SLAM3D_API CloudVertexSE3 : public g2o::VertexSE3
{
  public:
    using PointCloudPtr = pcl::PointCloud<pcl::PointXYZ>::Ptr;
    PointCloudPtr cloud;
    CloudVertexSE3(PointCloudPtr init_cloud) : VertexSE3() {cloud = init_cloud;}
};

/// Lidar-SLAM algorithm, for working with PointClouds
class LidarSlam
{
  public:
    // EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    using Point = pcl::PointXYZ;
    using PointCloud = pcl::PointCloud<Point>;
    using PointCloudPtr = PointCloud::Ptr;
    using CallbackType = std::function<void(const Eigen::Matrix4d transform, const std::uint64_t stamp)>;

    /// Main C-tor
    LidarSlam(const LidarSlamParameters& params = {});

    /// Set-up callback which publishes low-latency (pseudo real-time) transform, which may be used for odometry
    /// It contains "drifty" transform from the very-first frame into very last one
    void SetOdometryCallback(CallbackType&& callback) { odometry_callback_ = callback; }

    /// Set-up callback which publishes high-latency transform, suitable for mapping
    void SetMappingCallback(CallbackType&& callback) { mapping_callback_ = callback; }

    void Start();
    void Stop();

    ~LidarSlam();

    /// Use this function to add new (just arrived) point-cloud into the SLAM
    /// Each new cloud shall have later timestamp than previous
    void AddPointCloud(const PointCloudPtr& msg)
    {
        if (!msg)
        {
            throw std::runtime_error("LidarSlam::AddPointCloud() uninitialized cloud given");
        }
        if (latest_cloud_ && msg->header.stamp <= latest_cloud_->header.stamp)
        {
            throw std::runtime_error("LidarSlam::AddPointCloud() received clouds must have increasing timestamps");
        }
        {
            const std::lock_guard<std::mutex> guard(latest_cloud_mutex_);
            latest_cloud_ = msg;
        }
    }

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

    /// Loop Closure work in separate thread, thus it needs its own vector for found edges
    std::vector<g2o::EdgeSE3*> loopclosureEdges_{};
    /// as well as own mutex
    std::mutex loopclosure_mutex_{};



    /// Threads for independend SLAM parts
    std::thread odometry_thread_{}, mapping_thread_{}, loopclousures_thread_{};
    /// Respective bool atomics
    std::atomic_bool odometry_thread_running_{};
    std::atomic_bool loopclousures_thread_running_{};
    std::atomic_bool mapping_thread_running_{};

    /// Main loop-functions for respective threads
    void OdometryThread();
    void LoopClousuresThread();
    void MappingThread();

    const LidarSlamParameters& params_;

    int vertice_id_{}, edge_id_{};

    /// Information matrix (i.e. inverse covariance of the 6DoF measurement: X Y Z QX QY QZ)
    Eigen::Matrix<double, 6, 6> icp_information_{};

};

}  // namespace lidar_slam

#endif  // FOG_SW_LIDAR_SLAM_H
