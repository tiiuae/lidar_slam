#include <lidar_slam/lidar_slam.h>
#include <nav_msgs/msg/odometry.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2/convert.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <functional>
#include <memory>
using TransformStamped = geometry_msgs::msg::TransformStamped;

using namespace lidar_slam;

constexpr std::uint64_t NanoFactor = 1000000000UL;

/// Interface class between ROS2 and LidarSlam
/// Helps to abstract out all ROS stuff from LidarSlams
class LidarSlamNode : public rclcpp::Node
{
  public:
    using PointCloudMsg = sensor_msgs::msg::PointCloud2;
    using OdometryMsg = nav_msgs::msg::Odometry;
    using Point = LidarSlam::Point;
    using PointCloud = LidarSlam::PointCloud;

    LidarSlamNode()
        : Node("lidar_slam"),
          slam_{},
          base_frame_{},
          map_frame_{},
          odom_frame_{},
          cloud_subscriber_{},
          clock_(RCL_SYSTEM_TIME),
          tf_buffer_(std::make_shared<rclcpp::Clock>(clock_)),
          tf_listener_(tf_buffer_, true),
          tf_broadcaster_{}
    {
        declare_parameter<std::string>("cloud_topic", "/camera/depth/color/points");
        declare_parameter<std::string>("base_frame_id", "base_link");
        declare_parameter<std::string>("map_frame_id", "map");
        declare_parameter<std::string>("odom_frame_id", "odom");
        declare_parameter<std::string>("sensor_frame_id", "camera_depth_optical_frame");

        const std::string cloud_topic = get_parameter("cloud_topic").as_string();
        base_frame_ = get_parameter("base_frame_id").as_string();
        map_frame_ = get_parameter("map_frame_id").as_string();
        odom_frame_ = get_parameter("odom_frame_id").as_string();
        const std::string sensor_frame = get_parameter("sensor_frame_id").as_string();

        tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);

        rclcpp::QoS qos(5);
        qos = qos.best_effort();
        cloud_subscriber_ = create_subscription<PointCloudMsg>(
            cloud_topic, qos, std::bind(&LidarSlamNode::GrabPointCloud, this, std::placeholders::_1));

        odometry_publisher_ = create_publisher<OdometryMsg>("/odom", qos);

        slam_.SetOdometryCallback(
            [this](const Eigen::Matrix4d t, const std::uint64_t s) { this->PublishOdometry(t, s); });

        slam_.SetMappingCallback([this](const Eigen::Matrix4d t, const std::uint64_t s) { this->PublishMap(t, s); });

        if (!sensor_frame.empty())
        {
            // try to find sensor-to-base transform and send it as odometry message
            try
            {
                geometry_msgs::msg::TransformStamped sensor2base_msg =
                    tf_buffer_.lookupTransform(sensor_frame, base_frame_, clock_.now(), tf2::durationFromSec(2.0));

                sensor2base_msg.header.frame_id = odom_frame_;
                sensor2base_msg.child_frame_id = base_frame_;
                tf_broadcaster_->sendTransform(sensor2base_msg);
            }
            catch (...)
            {
            }
        }

        RCLCPP_INFO(get_logger(), "LidarSlamNode successfully initialized");
    }

    TransformStamped ConvertMatrix4ToTransformMsg(const Eigen::Matrix4d& transform, const std::uint64_t stamp)
    {
        builtin_interfaces::msg::Time stamp2;
        Eigen::Matrix3d rot = transform.block(0, 0, 3, 3);
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

    /// Process Odometry transform, coming from SLAM
    /// @param transform shall map odom frame (i.e. very first frame) into latest one
    void PublishOdometry(const Eigen::Matrix4d transform, const std::uint64_t stamp)
    {
        TransformStamped msg = ConvertMatrix4ToTransformMsg(transform, stamp);
        msg.header.frame_id = odom_frame_;
        msg.child_frame_id = base_frame_;
        tf_broadcaster_->sendTransform(msg);

        const Eigen::Matrix3d rot = transform.block(0, 0, 3, 3);
        const Eigen::Vector3d trans = transform.block(0, 3, 3, 1);
        const Eigen::Quaterniond q(rot);

        //        const Eigen::Matrix3f rot_inv = rot.transpose();
        //        const Eigen::Vector3f trans_inv = -rot_inv * trans;
        //        const Eigen::Quaternionf q_inv(rot_inv);
        const Eigen::Vector3d trans_inv = trans;
        const Eigen::Quaterniond q_inv(rot);

        OdometryMsg omsg;
        omsg.header.frame_id = odom_frame_;
        omsg.header.stamp.sec = stamp / NanoFactor;
        omsg.header.stamp.nanosec = stamp % NanoFactor;
        omsg.child_frame_id = base_frame_;

        // because the input transform gives us a pose of the odom frame from the latest frame point of view,
        // we need to intern the transform to obtain "direct odometry"
        omsg.pose.pose.position.x = trans_inv[0];
        omsg.pose.pose.position.y = trans_inv[1];
        omsg.pose.pose.position.z = trans_inv[2];
        omsg.pose.pose.orientation.x = q_inv.x();
        omsg.pose.pose.orientation.y = q_inv.y();
        omsg.pose.pose.orientation.z = q_inv.z();
        omsg.pose.pose.orientation.w = q_inv.w();
        odometry_publisher_->publish(omsg);
    }

    void PublishMap(const Eigen::Matrix4d transform, const std::uint64_t stamp)
    {
        TransformStamped msg = ConvertMatrix4ToTransformMsg(transform, stamp);
        msg.header.frame_id = map_frame_;
        msg.child_frame_id = base_frame_;
        tf_broadcaster_->sendTransform(msg);
    }

  private:
    LidarSlam slam_;
    std::string base_frame_, map_frame_, odom_frame_;
    rclcpp::Subscription<PointCloudMsg>::SharedPtr cloud_subscriber_;
    rclcpp::Publisher<OdometryMsg>::SharedPtr odometry_publisher_;

    rclcpp::Clock clock_;
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

    void GrabPointCloud(const PointCloudMsg::SharedPtr msg)
    {

        PointCloud::Ptr cloud(new PointCloud());
        pcl::moveFromROSMsg<Point>(*msg, *cloud);
        const std::int32_t sec = msg->header.stamp.sec;
        const std::uint32_t nanosec = msg->header.stamp.nanosec;
        cloud->header.stamp = std::uint64_t(sec) * NanoFactor + std::uint64_t(nanosec);
        RCLCPP_INFO(get_logger(), "Recieved cloud at time " + std::to_string(sec) + "." + std::to_string(nanosec));
        slam_.AddPointCloud(cloud);
    }
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);

    auto node = std::make_shared<LidarSlamNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();

    return 0;
}
