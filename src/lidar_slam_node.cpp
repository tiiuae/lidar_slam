#include <lidar_slam/lidar_slam_node.h>
#include <lidar_slam/ros_helpers.h>
#include <pcl_conversions/pcl_conversions.h>
#include <rclcpp/rclcpp.hpp>


using namespace lidar_slam;

LidarSlamNode::LidarSlamNode()
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
    declare_parameter<std::string>("gps_topic", "/gpsfix");
    declare_parameter<std::string>("base_frame_id", "base_link");
    declare_parameter<std::string>("map_frame_id", "map");
    declare_parameter<std::string>("odom_frame_id", "odom");
    declare_parameter<std::string>("sensor_frame_id", "camera_depth_optical_frame");

    const std::string cloud_topic = get_parameter("cloud_topic").as_string();
    const std::string gps_topic = get_parameter("gps_topic").as_string();
    base_frame_ = get_parameter("base_frame_id").as_string();
    map_frame_ = get_parameter("map_frame_id").as_string();
    odom_frame_ = get_parameter("odom_frame_id").as_string();
    sensor_frame_ = get_parameter("sensor_frame_id").as_string();

    tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);

    rclcpp::QoS qos(5);
    qos = qos.best_effort();
    cloud_subscriber_ = create_subscription<PointCloudMsg>(
        cloud_topic, qos, std::bind(&LidarSlamNode::GrabPointCloud, this, std::placeholders::_1));

    gps_subscriber_ =
        create_subscription<GpsMsg>(gps_topic, qos, std::bind(&LidarSlamNode::GrabGpsMsg, this, std::placeholders::_1));

    odometry_publisher_ = create_publisher<OdometryMsg>("/odom", qos);
    mapping_publisher_ = create_publisher<OdometryMsg>("/map", qos);

    slam_.SetOdometryCallback(
        [this](const Eigen::Isometry3d t, const std::uint64_t s) { this->PublishOdometry(t, s); });

    slam_.SetMappingCallback([this](const Eigen::Isometry3d t, const std::uint64_t s) { this->PublishMap(t, s); });

    RCLCPP_INFO(get_logger(), "LidarSlamNode successfully initialized");
}

void LidarSlamNode::PublishOdometry(const Eigen::Isometry3d sensor_odometry, const std::uint64_t stamp)
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
    TransformStamped msg = RosHelpers::Convert(sensor_odometry, stamp);
    msg.header.frame_id = odom_frame_;
    msg.child_frame_id = base_frame_;
    tf_broadcaster_->sendTransform(msg);
    OdometryMsg omsg = RosHelpers::Convert2Odometry(msg);
    odometry_publisher_->publish(omsg);

    const auto r = msg.transform.rotation;
    const auto t = msg.transform.translation;
    RCLCPP_INFO(
        get_logger(),
        "Odometry translation: [%f, %f, %f]",
        t.x, t.y, t.z);
    RCLCPP_INFO(get_logger(),
                "Odometry rotation: [%f, %f, %f, %f]",
        r.x, r.y, r.z, r.w);
}

void LidarSlamNode::PublishMap(const Eigen::Isometry3d transform, const std::uint64_t stamp)
{
    TransformStamped msg = RosHelpers::Convert(transform, stamp);
    msg.header.frame_id = map_frame_;
    msg.child_frame_id = odom_frame_;
    tf_broadcaster_->sendTransform(msg);

    OdometryMsg omsg = RosHelpers::Convert2Odometry(msg);
    mapping_publisher_->publish(omsg);


    const auto r = msg.transform.rotation;
    const auto t = msg.transform.translation;
    RCLCPP_INFO(
        get_logger(),
        "Mapping translation: [%f, %f, %f]",
        t.x, t.y, t.z);
    RCLCPP_INFO(get_logger(),
                "Mapping rotation: [%f, %f, %f, %f]",
        r.x, r.y, r.z, r.w);
}

void LidarSlamNode::GrabPointCloud(const PointCloudMsg::SharedPtr msg)
{
    PointCloud::Ptr cloud(new PointCloud());
    pcl::moveFromROSMsg<Point>(*msg, *cloud);
    const std::int32_t sec = msg->header.stamp.sec;
    const std::uint32_t nanosec = msg->header.stamp.nanosec;
    cloud->header.stamp = std::uint64_t(sec) * RosHelpers::NanoFactor + std::uint64_t(nanosec);
    slam_.AddPointCloud(cloud);
}

void LidarSlamNode::GrabGpsMsg(const GpsMsg::SharedPtr msg) {}

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);

    auto node = std::make_shared<lidar_slam::LidarSlamNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();

    return 0;
}
