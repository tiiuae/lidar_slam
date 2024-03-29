#include "rosbag2_cpp/converter_interfaces/serialization_format_converter.hpp"
#include "rosbag2_cpp/types.hpp"
#include "rosbag2_cpp/typesupport_helpers.hpp"
//#include <lidar_slam/lidar_slam.h>
#include <lidar_slam/helpers.h>
#include <pcl_conversions/pcl_conversions.h>
#include <px4_msgs/msg/vehicle_odometry.hpp>
#include <rosbag2_cpp/readers/sequential_reader.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <iostream>
#include <map>

using namespace rosbag2_cpp;
using namespace lidar_slam;
using namespace std::chrono;
using rosbag2_cpp::converter_interfaces::SerializationFormatConverter;
using PointCloud2 = sensor_msgs::msg::PointCloud2;
using VehicleOdometry = px4_msgs::msg::VehicleOdometry;
using Point = pcl::PointXYZ;
using PointCloud = pcl::PointCloud<Point>;

/// Converter from Bag Message into a ROS2 msg type
template <typename MsgType>
MsgType Convert(const std::shared_ptr<rosbag2_storage::SerializedBagMessage>& bag_message)
{
    MsgType msg{};
    auto ros_message = std::make_shared<rosbag2_introspection_message_t>();
    ros_message->time_stamp = 0;
    ros_message->allocator = rcutils_get_default_allocator();
    ros_message->message = &msg;
    auto library =
        rosbag2_cpp::get_typesupport_library(rosidl_generator_traits::name<MsgType>(), "rosidl_typesupport_cpp");
    auto typesupport = rosbag2_cpp::get_typesupport_handle(
        rosidl_generator_traits::name<MsgType>(), "rosidl_typesupport_cpp", library);

    SerializationFormatConverterFactory factory;
    std::unique_ptr<converter_interfaces::SerializationFormatDeserializer> cdr_deserializer_;
    cdr_deserializer_ = factory.load_deserializer("cdr");
    cdr_deserializer_->deserialize(bag_message, typesupport, ros_message);
    return msg;
}

/// Converts secs and nanos given in "system_time" into nanosecs of "steady_time"
steady_clock::rep ConvertTime(std::int32_t sec, std::uint32_t nanosec)
{
    const auto system_now = time_point_cast<nanoseconds>(system_clock::now()).time_since_epoch().count();
    const auto steady_now = time_point_cast<nanoseconds>(steady_clock::now()).time_since_epoch().count();
    steady_clock::rep cloud_stamp = sec * 1000000000ULL + nanosec - system_now + steady_now;
    return cloud_stamp;
}

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::cout << "USAGE: " << argv[0] << " path_to/rosbag2_file.db" << std::endl;
        return 1;
    }

    // Recovered Poses are stored in these buffers
    std::map<std::uint64_t, Eigen::Isometry3d> camera_poses{};
    std::map<std::uint64_t, Eigen::Isometry3d> px4_poses{};
    std::atomic<bool> camera_pose_saved{false};  // need this one due-to multi-threaded nature of SLAM

    //    LidarSlamParameters params{};
    //    params.automatic_start = true;
    //    LidarSlam slam(params);
    //    LidarSlam::CallbackType lambda = [&] (const Eigen::Isometry3d transform, const std::uint64_t stamp){
    //      camera_poses[stamp] = transform;
    //      camera_pose_saved = true;
    //      std::cout << "PointCloud " << stamp << " : " << transform.translation().transpose() << std::endl;
    //    };
    //    slam.SetMappingCallback(std::move(lambda));

    readers::SequentialReader reader;
    reader.open({argv[1], "sqlite3"}, {"cdr", "cdr"});

    while (reader.has_next())
    {
        auto bag_message = reader.read_next();
        std::cout << bag_message->topic_name << std::endl;
        if (bag_message->topic_name == "/camera/depth/color/points")
        {
            PointCloud2 cloud_msg = Convert<PointCloud2>(bag_message);
            PointCloud::Ptr cloud(new PointCloud());
            pcl::moveFromROSMsg<Point>(cloud_msg, *cloud);
            const auto steady_stamp = ConvertTime(cloud_msg.header.stamp.sec, cloud_msg.header.stamp.nanosec);
            const uint64_t sec = steady_stamp / 1000000000ULL;
            const uint64_t nanosec = steady_stamp % 1000000000ULL;
            cloud->header.stamp = steady_stamp;
            std::cout << "PointCloud " << sec << "." << nanosec << " : " << cloud->points.size() << std::endl;
            //            camera_pose_saved = false;
            auto corner = Helpers::Detect3DCorner<Point>(cloud);
            if (corner.has_value())
            {
                Eigen::Isometry3d transform(corner.value().cast<double>());
                // inverse of 'corner' pose gives sensor pose in "corner" coordinate system
                camera_poses[steady_stamp] = transform.inverse();
            }

            //            slam.AddPointCloud(cloud);
            //            size_t iters = 0;
            //            while(!camera_pose_saved && iters < 10)
            //            {
            //                std::this_thread::sleep_for(milliseconds(50));
            //                iters ++;
            //            }
        }
        if (bag_message->topic_name == "/default/fmu/vehicle_odometry/out")
        {
            VehicleOdometry odom_msg = Convert<VehicleOdometry>(bag_message);
            const std::uint64_t steady_stamp = static_cast<std::uint64_t>(odom_msg.timestamp) * 1000ULL;
            Eigen::Isometry3d transform = Eigen::Isometry3d::Identity();
            transform.translate(Eigen::Vector3d(odom_msg.x, odom_msg.y, odom_msg.z));
            transform.rotate(Eigen::Quaterniond(odom_msg.q[0], odom_msg.q[1], odom_msg.q[2], odom_msg.q[3]));
            px4_poses[steady_stamp] = transform;

            const uint64_t sec = steady_stamp / 1000000000ULL;
            const uint64_t nanosec = steady_stamp % 1000000000ULL;
            std::cout << "VehicleOdometry " << sec << "." << nanosec << " : " << transform.translation().transpose()
                      << std::endl;
        }
    }

    auto camera_iter = camera_poses.begin();
    auto px4_iter = px4_poses.begin();

    // buffer containing matched pose pairs
    // correspondence is found by selecting temporally-closest pairs
    std::vector<std::pair<Eigen::Isometry3d, Eigen::Isometry3d>> correspondances{};

    while (camera_iter != camera_poses.end() && px4_iter != px4_poses.end())
    {
        auto camera_pose = camera_iter->second;
        auto px4_pose = px4_iter->second;
        std::int64_t current_difference = std::int64_t(px4_iter->first) - std::int64_t(camera_iter->first);

        if (camera_iter->first < px4_iter->first)
        {
            camera_iter++;
            if (camera_iter != camera_poses.end())
            {
                std::int64_t next_difference = std::int64_t(px4_iter->first) - std::int64_t(camera_iter->first);
                if (std::abs(current_difference) < std::abs(next_difference))
                {
                    correspondances.push_back({camera_pose, px4_pose});
                    px4_iter++;  // iterate also px4, such that each pose appear only once
                }
            }
        }
        else
        {
            px4_iter++;
            if (px4_iter != px4_poses.end())
            {
                std::int64_t next_difference = std::int64_t(px4_iter->first) - std::int64_t(camera_iter->first);
                if (std::abs(current_difference) < std::abs(next_difference))
                {
                    correspondances.push_back({camera_pose, px4_pose});
                    camera_iter++;  // iterate also camera_iter, such that each pose appear only once
                }
            }
        }
    }

    std::cout << "============================================" << std::endl;
    for (auto pair : correspondances)
    {
        std::cout << pair.first.translation().transpose() << ",";
        std::cout << pair.first.rotation().transpose() << ",";
        std::cout << pair.second.translation().transpose() << ",";
        std::cout << pair.second.rotation().transpose() << ",";
    }

    return 0;
}
