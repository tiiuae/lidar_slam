#ifndef LIDAR_SLAM_HELPERS_H
#define LIDAR_SLAM_HELPERS_H

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
namespace lidar_slam
{

class Helpers
{
  public:
    using TransformStamped = geometry_msgs::msg::TransformStamped;
    static constexpr std::uint64_t NanoFactor = 1000000000UL;

    static TransformStamped Convert(const Eigen::Isometry3d& transform, const std::uint64_t stamp)
    {
        builtin_interfaces::msg::Time stamp2;
        Eigen::Matrix3d rot = transform.matrix().block(0, 0, 3, 3);
        Eigen::Quaterniond q(rot);

        TransformStamped msg;
        msg.transform.translation.x = transform(0, 3);
        msg.transform.translation.y = transform(1, 3);
        msg.transform.translation.z = transform(2, 3);
        msg.transform.rotation.x = q.x();
        msg.transform.rotation.y = q.y();
        msg.transform.rotation.z = q.z();
        msg.transform.rotation.w = q.w();
        msg.header.stamp.sec = stamp / NanoFactor;
        msg.header.stamp.nanosec = stamp % NanoFactor;

        return msg;
    }

    static Eigen::Isometry3d Convert(const TransformStamped& msg)
    {
        Eigen::Matrix4d matrix = Eigen::Matrix4d::Identity();
        const auto qm = msg.transform.rotation;
        Eigen::Quaterniond q(qm.x, qm.y, qm.z, qm.w);
        matrix.block(0, 0, 3, 3) = q.matrix();
        matrix(0, 3) = msg.transform.translation.x;
        matrix(1, 3) = msg.transform.translation.y;
        matrix(2, 3) = msg.transform.translation.z;
        return Eigen::Isometry3d(matrix);
    }
};


} //namespace lidar_slam

#endif  // LIDAR_SLAM_HELPERS_H
