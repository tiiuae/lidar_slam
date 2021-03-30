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
          params_(),
          slam_(params_),
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
        sensor_frame_ = get_parameter("sensor_frame_id").as_string();

        tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);

        RCLCPP_INFO(get_logger(), "LidarSlamNode cloud_topic = " + cloud_topic);

        rclcpp::QoS qos(10);
        qos = qos.best_effort();
        cloud_subscriber_ = create_subscription<PointCloudMsg>(
            cloud_topic, qos, std::bind(&LidarSlamNode::GrabPointCloud, this, std::placeholders::_1));

        odometry_publisher_ = create_publisher<OdometryMsg>("/odom", qos);

        slam_.SetOdometryCallback(
            [this](const Eigen::Isometry3d t, const std::uint64_t s) { this->PublishOdometry(t, s); });

        slam_.SetMappingCallback([this](const Eigen::Isometry3d t, const std::uint64_t s) { this->PublishMap(t, s); });

        RCLCPP_INFO(get_logger(), "LidarSlamNode successfully initialized");
    }

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

    /// @brief Processes "Sensor Frame Odometry" transform, coming from the SLAM into "Base Frame Odometry"
    /// As SLAM does not know about robot configuration, it only may estimate odometry transform from current sensor
    /// pose into initial sensor pose. This, however, would be inconvinient for the robot itself, as odometry needs
    /// to be defined for "base_link" frame.
    /// @param sensor_odometry transform from current sernsor frame into initial sensor frame
    void PublishOdometry(const Eigen::Isometry3d sensor_odometry, const std::uint64_t stamp)
    {
//        // In order to re-map transform onto base_frame we need to look for the sensor-to-base transform
//        // As it does not deviates so fast, 10 secs delay is pretty OK
//        geometry_msgs::msg::TransformStamped sensor2base_msg =
//            tf_buffer_.lookupTransform(base_frame_, sensor_frame_, clock_.now(), tf2::durationFromSec(10.0));
//
//        const Eigen::Isometry3d sensor2base = Convert(sensor2base_msg);
//        const Eigen::Isometry3d base2sensor = sensor2base.inverse();
//        const Eigen::Isometry3d odometry_transform = sensor2base * sensor_odometry * base2sensor;
//
//        TransformStamped msg = Convert(odometry_transform, stamp);
        TransformStamped msg = Convert(sensor_odometry, stamp);
        msg.header.frame_id = odom_frame_;
        msg.child_frame_id = base_frame_;
        tf_broadcaster_->sendTransform(msg);

        OdometryMsg omsg;
        omsg.header.frame_id = odom_frame_;
        omsg.header.stamp.sec = stamp / NanoFactor;
        omsg.header.stamp.nanosec = stamp % NanoFactor;
        omsg.child_frame_id = base_frame_;

        // because the input transform gives us a pose of the odom frame from the latest frame point of view,
        // we need to intern the transform to obtain "direct odometry"
        const auto t = msg.transform.translation;
        omsg.pose.pose.position.x = t.x;
        omsg.pose.pose.position.y = t.y;
        omsg.pose.pose.position.z = t.z;
        omsg.pose.pose.orientation = msg.transform.rotation;
        odometry_publisher_->publish(omsg);

        const auto r = msg.transform.rotation;
        RCLCPP_INFO(get_logger(),
                    "Odometry translation: [" + std::to_string(t.x) + "," + std::to_string(t.y) + "," +
                        std::to_string(t.z) + "]");
        RCLCPP_INFO(get_logger(),
                    "Odometry rotation: [" + std::to_string(r.x) + "," + std::to_string(r.y) + "," +
                        std::to_string(r.z) + "," + std::to_string(r.w) + "]");
    }

    void PublishMap(const Eigen::Isometry3d transform, const std::uint64_t stamp)
    {
        TransformStamped msg = Convert(transform, stamp);
        msg.header.frame_id = map_frame_;
        msg.child_frame_id = odom_frame_;
        tf_broadcaster_->sendTransform(msg);

        const auto r = msg.transform.rotation;
        const auto t = msg.transform.translation;
        RCLCPP_INFO(get_logger(),
                    "Mapping translation: [" + std::to_string(t.x) + "," + std::to_string(t.y) + "," +
                    std::to_string(t.z) + "]");
        RCLCPP_INFO(get_logger(),
                    "Mapping rotation: [" + std::to_string(r.x) + "," + std::to_string(r.y) + "," +
                    std::to_string(r.z) + "," + std::to_string(r.w) + "]");

    }

  private:
    LidarSlamParameters params_;
    LidarSlam slam_;
    std::string base_frame_, map_frame_, odom_frame_, sensor_frame_;
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
