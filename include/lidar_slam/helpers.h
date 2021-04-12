#ifndef LIDAR_SLAM_HELPERS_H
#define LIDAR_SLAM_HELPERS_H

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
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


    template<typename type=float>
    static Eigen::Matrix<type, 4, 4> lookAt
        (
            Eigen::Matrix<type, 3, 1> const &eye,
            Eigen::Matrix<type, 3, 1> const &center,
            Eigen::Matrix<type, 3, 1> const &up
        )
    {
        typedef Eigen::Matrix<type, 4, 4> Matrix4;
        typedef Eigen::Matrix<type, 3, 1> Vector3;

        Vector3 f = (center - eye).normalized();
        Vector3 u = up.normalized();
        Vector3 s = f.cross(u).normalized();
        u = s.cross(f);

        Matrix4 res;
        res << s.x(), s.y(), s.z(), -s.dot(eye),
            u.x(), u.y(), u.z(), -u.dot(eye),
            -f.x(), -f.y(), -f.z(), f.dot(eye),
            0, 0, 0, 1;

        return res;
    }

    static Eigen::Quaterniond Convert(const geometry_msgs::msg::Quaternion& quat)
    {
        return Eigen::Quaterniond(quat.x, quat.y, quat.z, quat.w);
    }

    static Eigen::Vector3d Convert(const geometry_msgs::msg::Point& point)
    {
        return Eigen::Vector3d(point.x, point.y, point.z);
    }


};


} //namespace lidar_slam

#endif  // LIDAR_SLAM_HELPERS_H
