#include <lidar_slam/lidar_slam.h>
#include <pcl_conversions/pcl_conversions.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2/convert.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <memory>
#include <functional>
using TransformStamped = geometry_msgs::msg::TransformStamped;

using namespace lidar_slam;

/// Interface class between ROS2 and LidarSlam
/// Helps to abstract out all ROS stuff from LidarSlams
class LidarSlamNode : public rclcpp::Node
{
  public:
    using PointCloudMsg = sensor_msgs::msg::PointCloud2;
    using Point = LidarSlam::Point;
    using PointCloud = LidarSlam::PointCloud;

    LidarSlamNode()
        : Node("lidar_slam"),
          slam_{},
          base_frame_{},
          map_frame_{},
          cloud_subscriber_{},
          clock_(RCL_SYSTEM_TIME),
          tf_buffer_(std::make_shared<rclcpp::Clock>(clock_)),
          tf_listener_(tf_buffer_, true),
          tf_broadcaster_{}
    {
        declare_parameter<std::string>("cloud_topic", "/camera/depth/color/points");
        declare_parameter<std::string>("base_frame_id", "base_link");
        declare_parameter<std::string>("map_frame_id", "map");

        const std::string cloud_topic = get_parameter("cloud_topic").as_string();
        base_frame_ = get_parameter("base_frame_id").as_string();
        map_frame_ = get_parameter("map_frame_id").as_string();

        tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);

        rclcpp::QoS qos(5);
        qos = qos.best_effort();
        cloud_subscriber_ = create_subscription<PointCloudMsg>(
            cloud_topic, qos, std::bind(&LidarSlamNode::GrabPointCloud, this, std::placeholders::_1));

        //slam_.SetCallback(std::bind(&LidarSlamNode::PublishTransform, this, std::placeholders::_2));
        slam_.SetCallback([this](const Eigen::Matrix4f t, const std::uint64_t s){this->PublishTransform(t, s);});

        RCLCPP_INFO(get_logger(), "Subscribed to topic '" + cloud_topic + "'");
    }

    void PublishTransform(const Eigen::Matrix4f transform, const std::uint64_t stamp)
    {
        builtin_interfaces::msg::Time stamp2;
        Eigen::Matrix3f rot = transform.block(0, 0, 3, 3);
        Eigen::Quaternionf q(rot);

        TransformStamped msg;
        msg.transform.translation.x = transform(0, 3);
        msg.transform.translation.y = transform(1, 3);
        msg.transform.translation.z = transform(2, 3);

        msg.transform.rotation.x = q.x();
        msg.transform.rotation.y = q.y();
        msg.transform.rotation.z = q.z();
        msg.transform.rotation.w = q.w();

        msg.header.stamp = stamp2;
        msg.header.frame_id = map_frame_;
        msg.child_frame_id = base_frame_;
    }

  private:
    LidarSlam slam_;
    std::string base_frame_, map_frame_;
    rclcpp::Subscription<PointCloudMsg>::SharedPtr cloud_subscriber_;
    rclcpp::Clock clock_;
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

    void GrabPointCloud(const PointCloudMsg::SharedPtr msg)
    {
        RCLCPP_INFO(get_logger(), "Recieved cloud at time " + std::to_string(msg->header.stamp.sec));

        PointCloud::Ptr cloud(new PointCloud());
        pcl::moveFromROSMsg<Point>(*msg, *cloud);
        slam_.GrabPointCloud(cloud);
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
