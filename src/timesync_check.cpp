#include <px4_msgs/msg/timesync.hpp>
#include <px4_msgs/msg/vehicle_odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

using namespace std::chrono;

class TimeSyncCheckNode : public rclcpp::Node
{
  public:
    using PointCloud = sensor_msgs::msg::PointCloud2;
    using VehicleOdometry = px4_msgs::msg::VehicleOdometry;
    using Timesync = px4_msgs::msg::Timesync;

    TimeSyncCheckNode() : Node("timesync_check_node")
    {
        rclcpp::QoS qos(5);
        qos = qos.best_effort();
        cloud_subscriber_ =
            create_subscription<PointCloud>("/camera/depth/color/points",
                                            qos,
                                            std::bind(&TimeSyncCheckNode::GrabPointCloud, this, std::placeholders::_1));
        odometry_subscriber_ = create_subscription<VehicleOdometry>(
            "/default/fmu/vehicle_odometry/out",
            qos,
            std::bind(&TimeSyncCheckNode::GrabVehicleOdometry, this, std::placeholders::_1));
    }

    void GrabPointCloud(const PointCloud::SharedPtr msg)
    {
        const std::int32_t sec = msg->header.stamp.sec;
        const std::uint32_t nanosec = msg->header.stamp.nanosec;

        // const auto system_now = time_point_cast<nanoseconds>(system_clock::now()).time_since_epoch().count();
        const auto system_now = now().nanoseconds();
        const auto steady_now = time_point_cast<nanoseconds>(steady_clock::now()).time_since_epoch().count();
        const system_clock::rep clock_diff = system_now - steady_now;
        steady_clock::rep cloud_stamp = sec * 1000000000ULL + nanosec - clock_diff;
        const uint64_t steady_secs = cloud_stamp / 1000000000ULL;
        const uint64_t steady_nano = cloud_stamp % 1000000000ULL;

        const uint64_t local_time = time_point_cast<nanoseconds>(steady_clock::now()).time_since_epoch().count();
        const uint64_t local_secs = local_time / 1000000000ULL;
        const uint64_t local_nanos = local_time % 1000000000ULL;

        int64_t ros_time = now().nanoseconds();

        std::cout << "Realsense (steady): " << steady_secs << "." << steady_nano;
        std::cout << " ; Local: " << local_secs << "." << local_nanos << std::endl;
        // std::cout << " ; ROS: " << ros_time / 1000000000LL << "." << ros_time % 1000000000LL << std::endl;
    }

    void GrabVehicleOdometry(const VehicleOdometry::SharedPtr msg)
    {
        const std::uint32_t sec = msg->timestamp / 1000000UL;
        const std::uint32_t millisec = msg->timestamp % 1000000UL;

        const uint64_t local_time = time_point_cast<nanoseconds>(steady_clock::now()).time_since_epoch().count();
        const uint64_t local_secs = local_time / 1000000000ULL;
        const uint64_t local_nanos = local_time % 1000000000ULL;

        std::cout << "PX4: " << sec << "." << millisec << " ; Local: " << local_secs << "." << local_nanos << std::endl;
    }

  private:
    rclcpp::Subscription<PointCloud>::SharedPtr cloud_subscriber_;
    rclcpp::Subscription<VehicleOdometry>::SharedPtr odometry_subscriber_;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);

    auto node = std::make_shared<TimeSyncCheckNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();

    return 0;
}
