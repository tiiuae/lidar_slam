#include "g2o/config.h"
#include "g2o/core/block_solver.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/core/robust_kernel_impl.h"
#include "g2o/core/solver.h"
#include "g2o/core/sparse_optimizer.h"
#include "g2o/solvers/cholmod/linear_solver_cholmod.h"
#include "g2o/solvers/dense/linear_solver_dense.h"
#include "g2o/solvers/structure_only/structure_only_solver.h"
#include "g2o/stuff/sampler.h"
#include "g2o/types/icp/types_icp.h"
#include <boost/optional/optional.hpp>
#include <fast_gicp/gicp/fast_vgicp.hpp>
#include <memory>
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
    using CallbackType = std::function<void(const Eigen::Matrix4d transform, const std::uint64_t stamp)>;

    /// Main Ctor
    LidarSlam();

    /// Set-up callback which publishes low-latency (pseudo real-time) transform, which may be used for odometry
    // It contains "drifty" transform from the very-first frame into very last one
    void SetOdometryCallback(CallbackType&& callback) { odometry_callback_ = callback; }

    /// Set-up callback which publishes high-latency transform, suitable for mapping
    void SetMappingCallback(CallbackType&& callback) { mapping_callback_ = callback; }

    ~LidarSlam()
    {
        local_thread_running_ = false;
        local_thread_.join();
    }

    /// Use this function to add new (just arrived) point-cloud into the SLAM
    /// Each new cloud shall have later timestamp than previous
    void AddPointCloud(const PointCloudPtr& msg)
    {
        if (latest_cloud_ && msg->header.stamp < latest_cloud_->header.stamp)
        {
            throw std::runtime_error("LidarSlam::AddPointCloud() received cloud with decreasing timestamp!");
        }
        const std::lock_guard<std::mutex> guard(buffer_mutex_);
        latest_cloud_ = msg;
    }

  private:
    boost::optional<Eigen::Matrix4d> TryAlignment(PointCloudPtr source,
                                                  PointCloudPtr target,
                                                  float max_shift_misalignment = 0.02F,
                                                  float max_angle_misalignment = 0.005F);

    void LocalThreadRun();

    /// @return pair of translation and rotation misalignment
    static std::pair<double, double> GetAbsoluteShiftAngle(const Eigen::Matrix4d& matrix);

    g2o::SparseOptimizer optimizer_;
    std::unique_ptr<g2o::OptimizationAlgorithmLevenberg> solver_;

    int vertice_id_ = 0;
    int edge_id_ = 0;
    std::vector<std::shared_ptr<g2o::VertexSE3>> vertices_;
    std::vector<std::shared_ptr<g2o::EdgeSE3>> odometryEdges_;
    std::vector<std::shared_ptr<g2o::EdgeSE3>> loopclosureEdges_;
    std::vector<PointCloudPtr> clouds_;

    /// Information matrix (i.e. covariance of the 6DoF measurement: X Y Z QX QY QZ)
    Eigen::Matrix<double, 6, 6> information_;

    CallbackType odometry_callback_, mapping_callback_;
    fast_gicp::FastVGICP<Point, Point> gicp1_;
    fast_gicp::FastVGICP<Point, Point> gicp2_;
    PointCloudPtr previous_cloud_;
    PointCloudPtr latest_cloud_;

    Eigen::Matrix4d accumulated_odometry_transform_;

    std::mutex buffer_mutex_;
    std::thread local_thread_;
    std::atomic_bool local_thread_running_;

#ifdef USE_OWN_FEATURES
  public:
    // List of "radius" (in meters) for feature calculation
    constexpr static std::array<float, 4> FeatureScales{0.1F, 0.5F, 1.F, 2.F};

  private:
    void static FeatureDetector(const PointCloud& cloud);
    static const std::size_t BufferSize = 5U;
    std::array<CloudMsg::SharedPtr, BufferSize> buffer_;
    std::size_t buffer_head_, buffer_tail_;
#endif
};

}  // namespace lidar_slam

#endif  // FOG_SW_LIDAR_SLAM_H
