#ifndef LIDAR_SLAM_LIDAR_SLAM_NODE_H
#define LIDAR_SLAM_LIDAR_SLAM_NODE_H

#include <lidar_slam/lidar_slam.h>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/nav_sat_fix.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <functional>
#include <memory>

namespace lidar_slam
{

/// @brief Interface class between ROS2 and LidarSlam
/// Helps to abstract out all ROS/PX4 stuff from LidarSlam
/// For instance, LidarSlam does not know about robot configuration (it only knows "sensor_frame")
/// While in the ROS we usually use following frame hierarchy:  map -> odom -> base_link -> "sensor_frame"
/// (sensor_frame may have different name)
class LidarSlamNode : public rclcpp::Node
{
  public:
    using PointCloudMsg = sensor_msgs::msg::PointCloud2;
    using OdometryMsg = nav_msgs::msg::Odometry;
    using GpsMsg = sensor_msgs::msg::NavSatFix;
    using Point = LidarSlam::Point;
    using PointCloud = LidarSlam::PointCloud;
    using TransformStamped = geometry_msgs::msg::TransformStamped;


    LidarSlamNode();

    /// @brief Outputs "Sensor Frame Odometry" transform, coming from the SLAM into "Base Frame Odometry"
    /// As SLAM does not know about robot configuration, it only may estimate odometry transform from current sensor
    /// pose into initial sensor pose. This, however, would be inconvinient for the robot itself,
    /// as odometry needs to be defined for "base_link" frame.
    ///
    /// @param sensor_odometry transform from current sernsor frame into initial sensor frame
    void PublishOdometry(const Eigen::Isometry3d sensor_odometry, const std::uint64_t stamp);


    /// @brief Outputs "Sensor Frame Mapping" transform, coming from the SLAM
    void PublishMap(const Eigen::Isometry3d sensor_map2odom, const std::uint64_t stamp);

  private:
    LidarSlamParameters params_;
    LidarSlam slam_;
    std::string base_frame_, map_frame_, odom_frame_, sensor_frame_;
    rclcpp::Subscription<PointCloudMsg>::SharedPtr cloud_subscriber_;
    rclcpp::Subscription<GpsMsg>::SharedPtr gps_subscriber_;
    rclcpp::Publisher<OdometryMsg>::SharedPtr odometry_publisher_;

    rclcpp::Clock clock_;
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

    ///
    void GrabPointCloud(PointCloudMsg::SharedPtr msg);

    void GrabGpsMsg(GpsMsg::SharedPtr msg);
};
}  // namespace lidar_slam

#endif  // LIDAR_SLAM_LIDAR_SLAM_NODE_H
